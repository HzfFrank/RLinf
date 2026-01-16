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

                # Check if replay buffer has enough samples (async version)
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

            # ========== START: Decide data source based on sampling probability ==========
            # Fixed sampling ratio: 1% from replay buffer, 99% from initial dataset
            sampling_ratio = self.cfg.algorithm.get("sampling_ratio", 0.9)
            use_initial_data = (
                self.initial_data_loader is not None
                and np.random.random() < sampling_ratio
            )
            # ========== END: Decide data source based on sampling probability ==========

            metrics = {}

            total_loss = 0.0

            # ========== START: Choose different loop based on data source ==========
            if use_initial_data:
                # Path 1: Sample from initial SFT dataset (similar to SFT worker)
                print("Sampling from initial SFT dataset")
                for idx in range(self.gradient_accumulation):
                    # Yield control to event loop periodically for async operations
                    await asyncio.sleep(0)

                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )

                    # Get data from initial data loader
                    try:
                        observation, actions = next(self.initial_data_iter)
                    except StopIteration:
                        # Reset iterator if exhausted
                        self.initial_data_iter = iter(self.initial_data_loader)
                        observation, actions = next(self.initial_data_iter)

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

                    total_loss += loss.item()
                    loss = loss / self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()
            else:
                # Path 2: Sample from replay buffer (existing logic)
                print("Sampling from replay buffer")
                global_batch_size_per_rank = (
                    self.cfg.actor.global_batch_size // self._world_size
                )
                if self.demo_buffer is not None:
                    print(f"Sampling from replay buffer and demo buffer")
                    replay_batch = self.replay_buffer.sample(global_batch_size_per_rank // 2)
                    demo_batch = self.demo_buffer.sample(global_batch_size_per_rank // 2)
                    global_batch = concat_batch(replay_batch, demo_batch)
                    print(f"Global batch size: {len(global_batch)}")
                else:
                    print(f"Sampling from replay buffer")
                    global_batch = self.replay_buffer.sample(global_batch_size_per_rank)
                    print(f"Global batch size: {len(global_batch)}")
                # Split into micro batches
                train_micro_batch_list = split_dict_to_chunk(
                    global_batch,
                    global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
                )

                for idx, batch in enumerate(train_micro_batch_list):
                    # Yield control to event loop periodically for async operations
                    await asyncio.sleep(0)

                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == len(train_micro_batch_list),
                    )

                    # Simple version: assume batch contains obs and action in correct format
                    # Extract observation fields from batch (from forward_inputs)
                    # These have keys like "observation/image", "observation/state", etc.
                    """hzf
                    obs_dict = {}
                    obs_prefix_keys = [k for k in batch.keys() if k.startswith("observation/")]
                    for key in obs_prefix_keys:
                        obs_dict[key] = batch[key]
                    # Also extract tokenized prompt fields if present
                    if "tokenized_prompt" in batch:
                        obs_dict["tokenized_prompt"] = batch["tokenized_prompt"]
                    if "tokenized_prompt_mask" in batch:
                        obs_dict["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"]
                    # Transform using model's input_transform to convert "observation/image" -> "image" etc.
                    # This matches what predict_action_batch does internally
                    processed_obs = self.model.input_transform(obs_dict, transpose=False)
                    observation = _model.Observation.from_dict(processed_obs)
                    """
                    if "model_observation" in batch:
                        observation = batch["model_observation"]
                        observation = _model.Observation.from_dict(observation)
                    else:
                        raise KeyError(
                            f"Could not find 'model_observation' or 'observation' in batch. Available keys: {list(batch.keys())}"
                        )
                    if "model_action" in batch:
                        actions = batch["model_action"]
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

                    total_loss += loss.item()
                    loss = loss / self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()
            # ========== END: Choose different loop based on data source ==========

            # Manual gradient sync (temporary fix)
            torch.cuda.synchronize()
            if torch.distributed.is_initialized():
                all_grads = []
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        all_grads.append((name, param.grad))
                process_group = getattr(self.model, "process_group", None)
                for name, grad in all_grads:
                    torch.distributed.all_reduce(
                        grad, op=torch.distributed.ReduceOp.AVG, group=process_group
                    )
                process_group = getattr(self.model, "process_group", None)
                torch.distributed.barrier(
                    group=process_group, device_ids=[torch.cuda.current_device()]
                )

            avg_loss = total_loss / self.gradient_accumulation
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
                    "loss": avg_loss,
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
