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

import copy
import gc
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.rollout.hf.utils import init_real_obs


class DaggerRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            #hzf
            # rollout_model_config.path = self.cfg.rollout.model.model_path
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model = get_model(rollout_model_config)

        # Start with actor.model as base to ensure all required fields are present
        # Then override with expert_model specific values
        expert_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(expert_model_config):
            # Override with expert_model specific values from actor.expert_model
            # hzf
            # expert_model_config.path = self.cfg.rollout.expert_model.model_path
            # Also set model_path for compatibility (get_model uses cfg.model_path)
            expert_model_config.model_path = self.cfg.rollout.expert_model.model_path

        self.expert_model = get_model(expert_model_config)

        self.hf_model.eval()
        self.expert_model.eval()

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def load_checkpoint(self, load_path):
        model_dict = torch.load(load_path)
        self.hf_model.load_state_dict(model_dict)

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
            "use_cache": True,
        }

        self._eval_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }
        
        # DAGGER: beta is the probability of using the learned policy (hf_model)
        # 1-beta is the probability of using the expert policy (expert_model)
        # Default to 0.5 if not specified
        if "expert_intervention_rate" in self.cfg.algorithm:
            self.initial_beta = self.cfg.algorithm.expert_intervention_rate
        else:
            # Default: 50% learned policy, 50% expert policy
            self.initial_beta = 0.5
        
        # Dynamic beta scheduling configuration
        self.beta_schedule = self.cfg.algorithm.get("expert_intervention_rate_schedule", "constant")
        self.beta_min = self.cfg.algorithm.get("expert_intervention_rate_min", 0.05)
        self.beta_decay_steps = self.cfg.algorithm.get("expert_intervention_rate_decay_steps", 1000)
        self.beta_decay_factor = self.cfg.algorithm.get("expert_intervention_rate_decay_factor", 0.9)
        
        # Initialize beta with initial value
        self.beta = self.initial_beta
        self.current_rollout_epoch = 0

    def update_beta(self, rollout_epoch: int):
        """
        Update expert intervention rate (beta) based on rollout epoch and schedule.
        
        Args:
            rollout_epoch: Current rollout epoch number (0-indexed)
        """
        old_beta = self.beta
        
        if self.beta_schedule == "constant":
            self.beta = self.initial_beta
        elif self.beta_schedule == "linear":
            progress = min(1.0, rollout_epoch / self.beta_decay_steps) if self.beta_decay_steps > 0 else 0.0
            self.beta = self.beta_min + (self.initial_beta - self.beta_min) * (1 - progress)
        elif self.beta_schedule == "exponential":
            # 每次rollout epoch都乘以decay_factor，持续衰减
            self.beta = self.initial_beta * (self.beta_decay_factor ** rollout_epoch)
        elif self.beta_schedule == "cosine":
            import math
            progress = min(1.0, rollout_epoch / self.beta_decay_steps) if self.beta_decay_steps > 0 else 0.0
            self.beta = self.beta_min + (self.initial_beta - self.beta_min) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            # Unknown schedule, use constant
            self.beta = self.initial_beta
        
        # Ensure beta is within valid range [0, 1]
        self.beta = max(0.0, min(1.0, self.beta))
        self.current_rollout_epoch = rollout_epoch
        
        # Log beta changes (only on rank 0 or 3, print every epoch)
        if hasattr(self, '_rank') and (self._rank == 0 or self._rank == 3):
            if self.beta_schedule == "exponential":
                print(f"[Beta Schedule] Rollout Epoch {rollout_epoch}: beta={self.beta:.4f} "
                      f"(schedule={self.beta_schedule}, decay_factor={self.beta_decay_factor:.3f}, "
                      f"initial_beta={self.initial_beta:.3f})")
            else:
                progress = min(1.0, rollout_epoch / self.beta_decay_steps) if self.beta_decay_steps > 0 else 0.0
                print(f"[Beta Schedule] Rollout Epoch {rollout_epoch}: beta={self.beta:.4f} "
                      f"(schedule={self.beta_schedule}, progress={progress:.2f})")

    def predict(self, env_obs, mode="train"):
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            kwargs = {"mode": mode}

        kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        use_expert = False
        has_expert_model = hasattr(self, "expert_model") and self.expert_model is not None
        if mode == "train" and has_expert_model:
            use_expert = np.random.random() < self.beta
        
        """hzf
        with torch.no_grad():
            if use_expert and has_expert_model:
                actions, result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
            else:
                if mode == "train":
                    actions, _ = self.hf_model.predict_action_batch(
                        env_obs=env_obs,
                        **kwargs,
                    )

                    _, result = self.expert_model.predict_action_batch(
                        env_obs=env_obs,
                        **kwargs,
                    )
                else:
                    actions, result = self.hf_model.predict_action_batch(
                        env_obs=env_obs,
                        **kwargs,
                    )
        """

        with torch.no_grad():
            if use_expert and has_expert_model:
                actions, result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
            else:
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )

        # Store whether expert was used in result for filtering (only relevant for simulation)
        # Convert to Python bool to ensure proper comparison
        result["use_expert"] = bool(use_expert)

        return actions, result

    def get_dones_and_rewards(
        self, env_output: dict[str, torch.Tensor], extracted_obs: dict[str, Any]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs

        Returns:
            Tuple of (dones, rewards, real_extracted_obs). dones and rewards are tensors.
        """
        # First step: no rewards yet, only dones
        real_extracted_obs = None
        if env_output["rewards"] is None:
            if hasattr(self.hf_model, "q_head"):
                real_extracted_obs = init_real_obs(extracted_obs)
            return (
                env_output["dones"].bool().cpu().contiguous(),
                None,
                real_extracted_obs,
            )

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()

        # Handle auto_reset: add bootstrap value to rewards for done episodes
        # Note: currently this is not correct for chunk-size>1 with partial reset
        if dones.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    final_extracted_obs = self.hf_model.preprocess_env_obs(final_obs)
                    if hasattr(self.hf_model, "q_head"):
                        real_extracted_obs = init_real_obs(final_extracted_obs)
                    actions, result = self.predict(final_extracted_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                # Add bootstrap value to the last step of done episodes
                rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        if real_extracted_obs is None and hasattr(self.hf_model, "q_head"):
            real_extracted_obs = init_real_obs(extracted_obs)
        return dones, rewards, real_extracted_obs

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name, src_rank=self.actor_weight_src_rank, async_op=True
        ).async_wait()

        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def update_intervene_actions(self, env_output, forward_inputs):
        """
        Update forward_inputs with intervene_actions (human intervention).
        
        For OpenPI models:
        - forward_inputs contains model_action (model-internal-space) and action (environment-space)
        - When intervene_actions is provided, we update both action and model_action
        - model_action is used for training, so we need to update it when action is updated
        - We directly reshape and assign the environment-space action to model_action
        - Note: This assumes the action format is compatible (output_transform mainly does Unnormalize)
        
        For other models:
        - forward_inputs contains action (environment-space)
        - We only update action
        """
        intervene_actions = env_output["intervene_actions"]
        intervene_flags = env_output["intervene_flags"]
        if intervene_actions is not None: 
            """           
            if "action" in forward_inputs:
                policy_action = forward_inputs["action"].to(intervene_actions.device)
                policy_action = policy_action.reshape(
                    policy_action.shape[0], self.hf_model.num_action_chunks, -1
                )
                intervene_actions = intervene_actions.reshape(
                    intervene_actions.shape[0], self.hf_model.num_action_chunks, -1
                )
                action = intervene_actions * intervene_flags[
                    ..., None
                ] + policy_action * (~intervene_flags[..., None])
                action = action.reshape(action.shape[0], -1)
                forward_inputs["action"] = action
            """
            if "model_action" in forward_inputs:
                model_action = forward_inputs["model_action"].to(intervene_actions.device)
                """
                model_action = model_action.reshape(
                    model_action.shape[0], self.hf_model.num_action_chunks, -1
                )
                """
                intervene_actions = intervene_actions.reshape(
                    intervene_actions.shape[0], self.hf_model.num_action_chunks, -1
                )
                model_action = intervene_actions * intervene_flags[
                    ..., None
                ] + model_action * (~intervene_flags[..., None])
                # model_action = model_action.reshape(model_action.shape[0], -1)
                forward_inputs["model_action"] = model_action
            else:
                raise NotImplementedError(
                    f"forward_inputs must contain 'model_action' for intervene_actions to work. "
                    f"Available keys: {list(forward_inputs.keys())}"
                )
        return forward_inputs

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        if self.enable_offload:
            self.reload_model()

        self.buffer_list = [
            EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            for _ in range(self.num_pipeline_stages)
        ]

        self.only_save_intervened = self.cfg.algorithm.get("only_save_intervened", False)

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        for epoch_idx in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            # Update beta (expert intervention rate) based on current epoch
            self.update_beta(epoch_idx)
            
            last_extracted_obs = [None for i in range(self.num_pipeline_stages)]
            last_forward_inputs = [
                None for i in range(self.num_pipeline_stages)
            ]  # save actions
            last_results = [None for i in range(self.num_pipeline_stages)]  # Store last step's predict result for only_save_intervened

            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)

                    if last_forward_inputs[stage_id] is not None:
                        last_forward_inputs[stage_id] = self.update_intervene_actions(
                            env_output, last_forward_inputs[stage_id]
                        )

                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                        env_output, extracted_obs
                    )
                    actions, result = self.predict(extracted_obs)
                    
                    # For OpenPI, forward_inputs contains model_action but not action (environment-space)
                    # We need to add action to forward_inputs so that update_intervene_actions can work
                    # Store the current step's forward_inputs with action added for next step's intervene handling
                    """hzf
                    current_forward_inputs = result["forward_inputs"].copy()
                    if "action" not in current_forward_inputs:
                        # Add environment-space action to forward_inputs for intervene_actions handling
                        # Convert actions from numpy to tensor if needed
                        if isinstance(actions, np.ndarray):
                            # Get device from extracted_obs
                            if isinstance(extracted_obs, dict):
                                sample_tensor = None
                                if "main_images" in extracted_obs:
                                    sample_tensor = extracted_obs["main_images"]
                                elif "states" in extracted_obs:
                                    sample_tensor = extracted_obs["states"]
                                elif len(extracted_obs) > 0:
                                    sample_tensor = list(extracted_obs.values())[0]
                                
                                if sample_tensor is not None and torch.is_tensor(sample_tensor):
                                    device = sample_tensor.device
                                else:
                                    device = "cpu"
                            else:
                                device = "cpu"
                            actions_tensor = torch.from_numpy(actions).to(device=device)
                        else:
                            actions_tensor = actions
                        current_forward_inputs["action"] = actions_tensor
                    """
                    # Check if the step we're saving (t-1) had intervention (like hil-serl: only save intervened steps)
                    # Note: We save last_forward_inputs[stage_id], which is the forward_inputs from step t-1
                    # So we need to check if step t-1 had intervention, not step t
                    should_save = True
                    if self.only_save_intervened:
                        # Check for intervention in step t-1 (the step we're saving):
                        # 1. Real-world: check intervene_flags from env_output (human intervention, no expert model needed)
                        # 2. Simulation: check if expert policy was used in step t-1 (last_results[stage_id]["use_expert"])
                        step_t_minus_1_intervened = False
                        
                        # Priority 1: Check real-world human intervention (from env_output, which is step t-1's output)
                        # In real-world setup, expert_model is removed, human intervention is handled by update_intervene_actions
                        if "intervene_flags" in env_output and env_output["intervene_flags"] is not None:
                            intervene_flags = env_output["intervene_flags"].bool()
                            step_t_minus_1_intervened = intervene_flags.any().item()
                        
                        # Priority 2: Check simulation: if expert policy was used in step t-1
                        # Only check this if no real-world intervention was detected
                        # last_results[stage_id] contains step t-1's prediction result
                        if not step_t_minus_1_intervened and last_results[stage_id] is not None:
                            if "use_expert" in last_results[stage_id]:
                                step_t_minus_1_intervened = bool(last_results[stage_id]["use_expert"])
                        
                        should_save = step_t_minus_1_intervened
                    
                    # Store last step's forward_inputs (which contains obs_{t-1} and action_{t-1})
                    # This is consistent with SAC version and allows proper handling of intervene_actions
                    # For SFT training, storing (obs_{t-1}, action_{t-1}) is mathematically equivalent to (obs_t, action_t)
                    # since we're learning the same mapping function f: obs -> action
                    if should_save:
                        chunk_step_result = ChunkStepResult(
                            prev_logprobs=result["prev_logprobs"],
                            prev_values=result["prev_values"],
                            dones=dones,
                            truncations=env_output["truncations"],
                            terminations=env_output["terminations"],
                            rewards=rewards,  # the first step is reset step, reward is none, which will not be appended to the buffer
                            forward_inputs=last_forward_inputs[stage_id],
                        )
                        self.buffer_list[stage_id].append_result(chunk_step_result)
                        if last_extracted_obs[stage_id] is not None and hasattr(
                            self.hf_model, "q_head"
                        ):
                            self.buffer_list[stage_id].add_transition(
                                last_extracted_obs[stage_id], real_extracted_obs
                            )
                    last_extracted_obs[stage_id] = extracted_obs
                    last_forward_inputs[stage_id] = result["forward_inputs"]
                    # Store current result for next step's intervention check
                    last_results[stage_id] = result

                    self.send_chunk_actions(output_channel, actions)

            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                last_forward_inputs[stage_id] = self.update_intervene_actions(
                    env_output, last_forward_inputs[stage_id]
                )

                extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                # Get dones and rewards from environment batch (final step of epoch)
                dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                    env_output, extracted_obs
                )
                
                # Check if the step we're saving (t-1) had intervention (like hil-serl: only save intervened steps)
                # Note: We save last_forward_inputs[stage_id], which is the forward_inputs from step t-1
                should_save = True
                if self.only_save_intervened:
                    step_t_minus_1_intervened = False
                    
                    # Priority 1: Check real-world human intervention (from env_output, which is step t-1's output)
                    if "intervene_flags" in env_output and env_output["intervene_flags"] is not None:
                        intervene_flags = env_output["intervene_flags"].bool()
                        step_t_minus_1_intervened = intervene_flags.any().item()
                    
                    # Priority 2: Check simulation: if expert policy was used in step t-1
                    if not step_t_minus_1_intervened and last_results[stage_id] is not None:
                        if "use_expert" in last_results[stage_id]:
                            step_t_minus_1_intervened = bool(last_results[stage_id]["use_expert"])
                    
                    should_save = step_t_minus_1_intervened
                
                if should_save:
                    self.buffer_list[stage_id].dones.append(dones)
                    self.buffer_list[stage_id].truncations.append(env_output["truncations"])
                    self.buffer_list[stage_id].terminations.append(
                        env_output["terminations"]
                    )
                    self.buffer_list[stage_id].rewards.append(rewards)
                    self.buffer_list[stage_id].forward_inputs.append(
                        put_tensor_device(last_forward_inputs[stage_id], "cpu")
                    )

                    with self.worker_timer():
                        actions, result = self.predict(extracted_obs)
                    
                    """hzf
                    if "action" not in result["forward_inputs"]:
                        # Add environment-space action to forward_inputs
                        # Convert actions from numpy to tensor if needed
                        if isinstance(actions, np.ndarray):
                            # Get device from extracted_obs
                            if isinstance(extracted_obs, dict):
                                sample_tensor = None
                                if "main_images" in extracted_obs:
                                    sample_tensor = extracted_obs["main_images"]
                                elif "states" in extracted_obs:
                                    sample_tensor = extracted_obs["states"]
                                elif len(extracted_obs) > 0:
                                    sample_tensor = list(extracted_obs.values())[0]
                                
                                if sample_tensor is not None and torch.is_tensor(sample_tensor):
                                    device = sample_tensor.device
                                else:
                                    device = "cpu"
                            else:
                                device = "cpu"
                            actions_tensor = torch.from_numpy(actions).to(device=device)
                        else:
                            actions_tensor = actions
                        result["forward_inputs"]["action"] = actions_tensor
                    """
                    if "prev_values" in result:
                        self.buffer_list[stage_id].prev_values.append(
                            result["prev_values"].cpu().contiguous()
                        )
                    if "prev_logprobs" in result:
                        self.buffer_list[stage_id].prev_logprobs.append(
                            result["prev_logprobs"].cpu().contiguous()
                        )
                    if hasattr(self.hf_model, "q_head"):
                        self.buffer_list[stage_id].add_transition(
                            last_extracted_obs[stage_id], real_extracted_obs
                        )

        for i in range(self.num_pipeline_stages):
            self.send_rollout_batch(actor_channel, i)

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    actions, _ = self.predict(extracted_obs, mode="eval")
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        self.expert_model = self.expert_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)
        self.expert_model = self.expert_model.to(self.device)

    async def recv_env_output(
        self, input_channel: Channel, mode="train"
    ) -> dict[str, torch.Tensor]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        # Use asyncio so that it can run alongside async weight syncing
        env_output = await input_channel.get(
            key=f"{self._rank}_{mode}", async_op=True
        ).async_wait()
        return env_output

    def send_chunk_actions(self, output_channel: Channel, chunk_actions, mode="train"):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        output_channel.put(
            item=chunk_actions, key=f"{self._rank}_{mode}", async_op=True
        )

    def send_rollout_batch(self, actor_channel: Channel, stage_id: int):
        # send rollout_batch to actor
        split_num = self.get_actor_split_num()
        splitted_rollout_result = self.buffer_list[stage_id].to_splitted_dict(split_num)
        for i in range(split_num):
            actor_channel.put(item=splitted_rollout_result[i], async_op=True)

    def get_actor_split_num(self):
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
