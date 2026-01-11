# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

import jax
import numpy as np
import torch
import openpi.models.model as _model
from rlinf.scheduler import Channel
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    concat_batch,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_dagger_worker import EmbodiedDAGGERFSDPPolicy


class AsyncEmbodiedDAGGERFSDPPolicy(EmbodiedDAGGERFSDPPolicy):
    async def start_replay_buffer(self, replay_channel: Channel):
        """Start the replay buffer in async mode."""
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        replay_buffer_task = asyncio.create_task(
            self.replay_buffer.run(
                self.cfg, data_channel=replay_channel, split_num=split_num
            )
        )
        await replay_buffer_task

    async def run_training(self):
        """DAgger SFT training using replay buffer (async version)"""
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                self.load_param_and_grad(self.device)
                self.load_optimizer(self.device)

            min_buffer_size = (
                self.cfg.algorithm.get("min_buffer_size", 100) // self._world_size
            )
            current_buffer_size = len(self.replay_buffer)
            if not (await self.replay_buffer.is_ready_async(min_buffer_size)):
                self.log_on_first_rank(
                    f"Replay buffer size {current_buffer_size} < {min_buffer_size}, skipping training. "
                    f"Buffer capacity: {self.replay_buffer.capacity if hasattr(self.replay_buffer, 'capacity') else 'N/A'}"
                )
                return False
            print(f"Training with {current_buffer_size} samples")
            self.model.train()
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()

            assert (
                self.cfg.actor.global_batch_size
                % (self.cfg.actor.micro_batch_size * self._world_size)
                == 0
                ), "global_batch_size is not divisible by micro_batch_size * world_size"

            self.gradient_accumulation = (
                self.cfg.actor.global_batch_size
                // self.cfg.actor.micro_batch_size
                // self._world_size
            )

            # Sample batch from replay buffer
            global_batch_size_per_rank = (
                self.cfg.actor.global_batch_size // self._world_size
            )
            if self.demo_buffer is not None:
                replay_batch = self.replay_buffer.sample(global_batch_size_per_rank // 2)
                demo_batch = self.demo_buffer.sample(global_batch_size_per_rank // 2)
                global_batch = concat_batch(replay_batch, demo_batch)
            else:
                global_batch = self.replay_buffer.sample(global_batch_size_per_rank)

            # Split into micro batches
            train_micro_batch_list = split_dict_to_chunk(
                global_batch,
                global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
            )

            metrics = {}

            for idx, batch in enumerate(train_micro_batch_list):
                # Yield control to event loop periodically for async operations
                await asyncio.sleep(0)

                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == len(train_micro_batch_list),
                )

                obs_dict = {}
                obs_prefix_keys = [k for k in batch.keys() if k.startswith("observation/")]
                for key in obs_prefix_keys:
                    obs_dict[key] = batch[key]
                if "tokenized_prompt" in batch:
                    obs_dict["tokenized_prompt"] = batch["tokenized_prompt"]
                if "tokenized_prompt_mask" in batch:
                    obs_dict["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"]
                processed_obs = self.model.input_transform(obs_dict, transpose=False)
                observation = _model.Observation.from_dict(processed_obs)
                
                if "model_action" in batch:
                    actions = batch["model_action"]
                elif "action" in batch:
                    actions = batch["action"]
                else:
                    raise KeyError(
                        f"Could not find 'model_action' or 'action' in batch. Available keys: {list(batch.keys())}"
                    )

                observation = jax.tree.map(
                    lambda x: torch.as_tensor(x, device=self.device)
                    .contiguous()
                    .clone(),
                    observation,
                )
                actions = actions.to(torch.float32)
                actions = actions.to(self.device)

                with self.amp_context:
                    losses = self.model(
                        forward_type="sft_forward",
                        data={"observation": observation, "actions": actions},
                    )
                    if isinstance(losses, (list, tuple)):
                        losses = torch.stack(losses)
                    elif not isinstance(losses, torch.Tensor):
                        losses = torch.tensor(
                            losses, device=self.device, dtype=torch.float32
                        )
                    loss = losses.mean()

                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            # Collect stats
            lr_value = (
                lr_list[0] if len(lr_list) > 0 else self.optimizer.param_groups[0]["lr"]
            )
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            append_to_dict(
                metrics,
                {
                    "loss": loss.item(),
                    "learning_rate": lr_value,
                    "grad_norm": grad_norm_value,
                },
            )

            self.lr_scheduler.step()

            clear_memory()
            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            # Add buffer stats
            replay_buffer_stats = self.replay_buffer.get_stats()
            replay_buffer_stats = {
                f"replay_buffer/{key}": value
                for key, value in replay_buffer_stats.items()
            }
            train_metrics.update(replay_buffer_stats)

            torch.cuda.synchronize()
            torch.distributed.barrier()
            torch.cuda.empty_cache()

            return train_metrics
