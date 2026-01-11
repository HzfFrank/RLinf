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

import numpy as np
import torch
from tqdm import tqdm

from rlinf.data.io_struct import AsyncEmbodiedRolloutBuffer
from rlinf.scheduler import Channel
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.rollout.hf.dagger_rollout_worker import DaggerRolloutWorker


class AsyncDaggerRolloutWorker(DaggerRolloutWorker):
    def _add_action_to_forward_inputs(self, actions, extracted_obs, result):
        if "action" not in result["forward_inputs"]:
            if isinstance(actions, np.ndarray):
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

    async def generate(
        self, input_channel: Channel, output_channel: Channel, replay_channel: Channel
    ):
        self.buffer_list: list[AsyncEmbodiedRolloutBuffer] = [
            AsyncEmbodiedRolloutBuffer() for _ in range(self.num_pipeline_stages)
        ]

        self.buffer_tasks: list[asyncio.Task] = []
        for buffer in self.buffer_list:
            self.buffer_tasks.append(
                asyncio.create_task(
                    buffer.run(replay_channel, self.get_actor_split_num())
                )
            )

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        # 检查是否只保存干预数据（类似 hil-serl）
        self.only_save_intervened = self.cfg.algorithm.get(
            "only_save_intervened", False
        )

        # 用于调试的统计信息
        self._total_steps = 0
        self._intervened_steps = 0
        self._saved_steps = 0

        # 跟踪 rollout epoch 以进行动态 beta 调整
        rollout_epoch_counter = 0

        progress_bar = tqdm(
            total=None,
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        )

        while not self.should_stop:
            # 根据当前 rollout epoch 更新 beta（专家干预率）
            self.update_beta(rollout_epoch_counter)
            rollout_epoch_counter += 1
            last_extracted_obs = [None for i in range(self.num_pipeline_stages)]
            last_results = [None for i in range(self.num_pipeline_stages)]

            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)

                    if last_results[stage_id] is not None:
                        last_results[stage_id]["forward_inputs"] = (
                            self.update_intervene_actions(
                                env_output, last_results[stage_id]["forward_inputs"]
                            )
                        )

                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                        env_output, extracted_obs
                    )

                    actions, result = self.predict(extracted_obs)

                    # 对于 OpenPI，forward_inputs 包含 model_action 但不包含 action，将 action 添加到 forward_inputs 以便 update_intervene_actions 可以工作
                    self._add_action_to_forward_inputs(actions, extracted_obs, result)

                    # 统计：跟踪当前步骤的专家使用情况
                    if "use_expert" in result:
                        if bool(result["use_expert"]):
                            self._intervened_steps += 1
                    self._total_steps += 1

                    # 检查我们正在保存的步骤（t-1）是否有干预（类似 hil-serl：只保存干预步骤）
                    # 注意：我们保存 last_results[stage_id]，这是步骤 t-1 的预测结果
                    # 这包含 (obs_{t-1}, action_{t-1})，这对于 SFT 训练是正确的
                    # 我们不在这里保存当前步骤的结果，因为 env_output 来自步骤 t-1，
                    # 保存 (obs_{t-1}, action_t) 是不正确的
                    should_save = True
                    if self.only_save_intervened:
                        # 检查步骤 t-1（我们正在保存的步骤）中的干预：
                        # 1. 真实世界：检查 env_output 中的 intervene_flags（人工干预，不需要专家模型）
                        # 2. 仿真：检查步骤 t-1 是否使用了专家策略（last_results[stage_id]["use_expert"]）
                        step_t_minus_1_intervened = False

                        # 优先级 1：检查真实世界的人工干预（来自 env_output，这是步骤 t-1 的输出）
                        # 在真实世界设置中，expert_model 被移除，人工干预由 update_intervene_actions 处理
                        if (
                            "intervene_flags" in env_output
                            and env_output["intervene_flags"] is not None
                        ):
                            intervene_flags = env_output["intervene_flags"].bool()
                            step_t_minus_1_intervened = intervene_flags.any().item()

                        # 优先级 2：检查仿真：步骤 t-1 是否使用了专家策略
                        # 只有在未检测到真实世界干预时才检查此项
                        # last_results[stage_id] 包含步骤 t-1 的预测结果
                        if (
                            not step_t_minus_1_intervened
                            and last_results[stage_id] is not None
                        ):
                            if "use_expert" in last_results[stage_id]:
                                use_expert_value = last_results[stage_id]["use_expert"]
                                step_t_minus_1_intervened = bool(use_expert_value)

                        should_save = step_t_minus_1_intervened

                    # 统计：跟踪已保存的步骤
                    if should_save:
                        self._saved_steps += 1
                        # 每 100 步或前 50 个已保存步骤记录日志以进行调试
                        if (
                            self._saved_steps <= 50 or self._saved_steps % 100 == 0
                        ) and self._rank == 0:
                            print(
                                f"[DEBUG] Saved {self._saved_steps} steps out of {self._total_steps} total steps "
                                f"({self._intervened_steps} intervened steps, "
                                f"save_rate: {self._saved_steps / self._total_steps * 100:.1f}%)"
                            )

                    # 记录前 100 步或 should_save 变化时的日志以进行调试
                    if (
                        self._total_steps <= 100 or (self._total_steps % 1000 == 0)
                    ) and self._rank == 0:
                        use_expert_current = (
                            bool(result.get("use_expert", False)) if result else False
                        )
                        use_expert_last = (
                            bool(last_results[stage_id].get("use_expert", False))
                            if last_results[stage_id]
                            else False
                        )
                        print(
                            f"[DEBUG Step {self._total_steps}] should_save={should_save}, "
                            f"use_expert_current={use_expert_current}, use_expert_last={use_expert_last}, "
                            f"last_results_is_none={last_results[stage_id] is None}, "
                            f"saved_steps={self._saved_steps}, intervened_steps={self._intervened_steps}"
                        )

                    if should_save:
                        await self.buffer_list[stage_id].add(
                            "truncations",
                            env_output["truncations"].bool().cpu().contiguous(),
                        )
                        await self.buffer_list[stage_id].add(
                            "terminations",
                            env_output["terminations"].bool().cpu().contiguous(),
                        )
                        await self.buffer_list[stage_id].add("dones", dones)
                        if rewards is not None:
                            await self.buffer_list[stage_id].add("rewards", rewards)
                        if last_results[stage_id] is not None:
                            await self.buffer_list[stage_id].add_result(
                                last_results[stage_id]
                            )

                        if last_extracted_obs[stage_id] is not None and hasattr(
                            self.hf_model, "q_head"
                        ):
                            await self.buffer_list[stage_id].add_transition(
                                last_extracted_obs[stage_id], real_extracted_obs
                            )

                    last_extracted_obs[stage_id] = extracted_obs
                    last_results[stage_id] = result

                    self.send_chunk_actions(output_channel, actions)

            for i in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)

                # 如果可用，使用 intervene_actions 更新 last_results
                if last_results[i] is not None:
                    last_results[i]["forward_inputs"] = self.update_intervene_actions(
                        env_output, last_results[i]["forward_inputs"]
                    )

                extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                    env_output, extracted_obs
                )

                # 检查我们正在保存的步骤（t-1）是否有干预（类似 hil-serl：只保存干预步骤）
                # 注意：我们保存 last_results[i]，这是步骤 t-1 的预测结果
                # 这包含 (obs_{t-1}, action_{t-1})，这对于 SFT 训练是正确的
                should_save = True
                if self.only_save_intervened:
                    step_t_minus_1_intervened = False

                    # 优先级 1：检查真实世界的人工干预（来自 env_output，这是步骤 t-1 的输出）
                    # 在真实世界设置中，expert_model 被移除，人工干预由 update_intervene_actions 处理
                    if (
                        "intervene_flags" in env_output
                        and env_output["intervene_flags"] is not None
                    ):
                        intervene_flags = env_output["intervene_flags"].bool()
                        step_t_minus_1_intervened = intervene_flags.any().item()

                    # 优先级 2：检查仿真：步骤 t-1 是否使用了专家策略
                    # 只有在未检测到真实世界干预时才检查此项
                    # last_results[i] 包含步骤 t-1 的预测结果
                    if not step_t_minus_1_intervened and last_results[i] is not None:
                        if "use_expert" in last_results[i]:
                            step_t_minus_1_intervened = bool(
                                last_results[i]["use_expert"]
                            )

                    should_save = step_t_minus_1_intervened

                if should_save:
                    await self.buffer_list[i].add(
                        "truncations",
                        env_output["truncations"].bool().cpu().contiguous(),
                    )
                    await self.buffer_list[i].add(
                        "terminations",
                        env_output["terminations"].bool().cpu().contiguous(),
                    )
                    await self.buffer_list[i].add("dones", dones)
                    if rewards is not None:
                        await self.buffer_list[i].add("rewards", rewards)
                    if last_results[i] is not None:
                        await self.buffer_list[i].add_result(
                            put_tensor_device(last_results[i], "cpu")
                        )

                with self.worker_timer():
                    actions, result = self.predict(extracted_obs)

                self._add_action_to_forward_inputs(actions, extracted_obs, result)

                if "prev_values" in result:
                    await self.buffer_list[i].add(
                        "prev_values", result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    await self.buffer_list[i].add_transition(
                        last_extracted_obs[i], real_extracted_obs
                    )

                last_extracted_obs[i] = extracted_obs
                last_results[i] = result

            progress_bar.update(1)

    async def stop(self):
        self.should_stop = True
        for buffer in self.buffer_list:
            await buffer.stop()
        await asyncio.gather(*self.buffer_tasks)
