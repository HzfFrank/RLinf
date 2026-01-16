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


import os

import jax
import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.replay_buffer import SACReplayBuffer
from rlinf.scheduler import Channel
import openpi.models.model as _model
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_rollout_metrics,
)
from rlinf.utils.nested_dict_process import (
    concat_batch,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedDAGGERFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # DAgger buffer initialization
        self.replay_buffer = None
        self.demo_buffer = None
        # ========== START: Initial SFT data loader ==========
        self.initial_data_loader = None
        self.initial_data_iter = None
        # ========== END: Initial SFT data loader ==========

    def init_worker(self):
        # Use parent class method (FSDPModelManager.setup_model_and_optimizer)
        # This ensures training standard is the same as original SFT
        super().setup_model_and_optimizer()
        self.setup_buffer()
        # ========== START: Setup initial SFT data loader ==========
        self.setup_initial_data_loader()
        # ========== END: Setup initial SFT data loader ==========
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def setup_buffer(self):
        """Initialize replay buffer for DAgger."""
        seed = self.cfg.actor.get("seed", 1234)
        self.replay_buffer = SACReplayBuffer(
            capacity=self.cfg.algorithm.replay_buffer_capacity,
            device=self.device,
            seed=seed,
        )

    # ========== START: Setup initial SFT data loader ==========
    def setup_initial_data_loader(self):
        """Initialize initial SFT dataset data loader for mixed sampling."""
        # ========== START: Get initial_sft_data_path from config ==========
        initial_sft_data_path = self.cfg.get("initial_sft_data_path", None)
        if not initial_sft_data_path:
            self.initial_data_loader = None
            self.initial_data_iter = None
            print("No initial SFT data path provided")
            return
        # ========== END: Get initial_sft_data_path from config ==========

        try:
            import os
            os.environ["HF_LEROBOT_HOME"] = initial_sft_data_path

            from rlinf.config import SupportedModel
            if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI:
                import openpi.training.data_loader as openpi_data_loader
                from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

                config = get_openpi_config(
                    self.cfg.actor.model.openpi.config_name,
                    model_path=self.cfg.actor.model.model_path,
                    batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                )
                self.initial_data_loader = openpi_data_loader.create_data_loader(
                    config, framework="pytorch", shuffle=True
                )
                self.initial_data_iter = iter(self.initial_data_loader)
                self.log_on_first_rank(
                    f"Initial SFT data loader setup successfully from {initial_sft_data_path}"
                )
            else:
                self.log_on_first_rank(
                    f"Initial SFT data loader not supported for model type {self.cfg.actor.model.model_type}"
                )
                self.initial_data_loader = None
                self.initial_data_iter = None
        except Exception as e:
            self.log_on_first_rank(
                f"Failed to setup initial SFT data loader: {e}. Continuing without initial data."
            )
            self.initial_data_loader = None
            self.initial_data_iter = None
    # ========== END: Setup initial SFT data loader ==========

    def recv_rollout_batch(self, input_channel: Channel):
        super().recv_rollout_batch(input_channel)
        self.replay_buffer.add_rollout_batch(self.rollout_batch)

    async def recv_demo_data(self, input_channel: Channel):
        demo_data = await input_channel.get(async_op=True).async_wait()
        self.demo_buffer = SACReplayBuffer.create_from_buffer(
            demo_data, seed=self.cfg.actor.seed
        )

    def run_training(self):
        """DAgger SFT training using replay buffer"""
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                self.load_param_and_grad(self.device)
                self.load_optimizer(self.device)

            # Check if replay buffer has enough samples
            min_buffer_size = (
                self.cfg.algorithm.get("min_buffer_size", 100) // self._world_size
            )
            if not self.replay_buffer.is_ready(min_buffer_size):
                self.log_on_first_rank(
                    f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
                )
                return {}

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
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == len(train_micro_batch_list),
                )

                # Simple version: assume batch contains obs and action in correct format
                # Extract observation fields from batch (from forward_inputs)
                # These have keys like "observation/image", "observation/state", etc.
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
                
                # For OpenPI, use model_action (model-internal-space) instead of action (environment-space)
                # model_action is needed for sft_forward which expects model-internal-space actions
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

    def compute_advantages_and_returns(self):
        """
        SAC doesn't compute advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        # Just compute basic rollout metrics without advantages/returns
        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    def save_checkpoint(self, save_base_path, step):
        super().save_checkpoint(save_base_path, step)
        buffer_path = os.path.join(
            self.cfg.runner.logger.log_path, f"replay_buffer_{self._rank}.pkl"
        )
        self.replay_buffer.save(buffer_path)
