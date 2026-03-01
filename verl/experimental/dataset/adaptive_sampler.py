# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adaptive difficulty-weighted sampler for OR-Bench GRPO training.

Maintains per-difficulty EMA reward estimates updated every training step,
and samples problems proportionally to the variance proxy w(nv) = r*(1-r),
which peaks at reward=0.5 (the learning frontier) and falls to zero at both
trivially-easy (r→1) and completely-unsolvable (r→0) extremes.
"""
from collections import defaultdict

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


class AdaptiveDifficultyWeightedSampler(AbstractCurriculumSampler):
    """Sample OR-Bench problems weighted by current per-difficulty learning signal.

    Weight function: w(nv) = ema_reward(nv) * (1 - ema_reward(nv)) + floor_weight

    This concentrates compute on the "learning frontier" — difficulty levels where
    the model sometimes succeeds and sometimes fails — rather than on problems that
    are already mastered or completely out of reach.

    Config keys (all under data.sampler.*):
        ema_alpha (float): EMA decay for reward estimates. Default 0.05 (~20-step lag).
        floor_weight (float): Minimum per-difficulty weight to prevent starvation. Default 0.05.
    """

    def __init__(self, data_source, data_config: DictConfig):
        self.data_source = data_source
        sampler_config = data_config.get("sampler", {})
        self.ema_alpha = float(sampler_config.get("ema_alpha", 0.05))
        self.floor_weight = float(sampler_config.get("floor_weight", 0.05))

        # Read num_vars for every problem in the dataset
        n = len(data_source)
        self.sample_num_vars: list[int | None] = []
        for i in range(n):
            row = data_source[i]
            extra_info = row.get("extra_info", {})
            nv = extra_info.get("num_vars") if isinstance(extra_info, dict) else None
            self.sample_num_vars.append(int(nv) if nv is not None else None)

        # Collect all unique difficulty levels
        self.all_nvs: list[int] = sorted({nv for nv in self.sample_num_vars if nv is not None})

        # Initialize EMA reward estimates to 0.5 (neutral → equal initial weights)
        self.ema_rewards: dict[int, float] = {nv: 0.5 for nv in self.all_nvs}

        # Build initial per-sample weight tensor
        self.sample_weights = self._compute_sample_weights()

        self._step = 0
        print(
            f"[AdaptiveDifficultyWeightedSampler] Initialized: {n} problems, "
            f"difficulty levels {self.all_nvs}, ema_alpha={self.ema_alpha}, "
            f"floor_weight={self.floor_weight}"
        )

    def _compute_sample_weights(self) -> torch.Tensor:
        """Recompute per-sample weights from current EMA reward estimates."""
        weights = []
        for nv in self.sample_num_vars:
            if nv is not None and nv in self.ema_rewards:
                r = self.ema_rewards[nv]
                w = r * (1.0 - r) + self.floor_weight
            else:
                # Unknown difficulty: use neutral weight
                w = 0.25 + self.floor_weight
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)

    def update(self, batch: DataProto) -> None:
        """Update EMA reward estimates from the current training batch.

        Called automatically by ray_trainer after every training step.
        batch.non_tensor_batch["reward"] contains per-rollout scores.
        batch.non_tensor_batch["extra_info"] contains per-rollout extra_info dicts.
        """
        rewards = batch.non_tensor_batch.get("reward", None)
        extra_infos = batch.non_tensor_batch.get("extra_info", None)

        if rewards is None or extra_infos is None:
            return

        # Group rewards by num_vars
        rewards_by_nv: dict[int, list[float]] = defaultdict(list)
        for reward, ei in zip(rewards, extra_infos):
            nv = ei.get("num_vars") if isinstance(ei, dict) else None
            if nv is not None:
                rewards_by_nv[int(nv)].append(float(reward))

        # Update EMA and recompute weights
        updated = False
        for nv, nv_rewards in rewards_by_nv.items():
            if nv in self.ema_rewards:
                batch_mean = float(np.mean(nv_rewards))
                self.ema_rewards[nv] = (1.0 - self.ema_alpha) * self.ema_rewards[nv] + self.ema_alpha * batch_mean
                updated = True

        if updated:
            self.sample_weights = self._compute_sample_weights()

        self._step += 1
        if self._step % 50 == 0:
            ema_str = ", ".join(f"nv{nv}={r:.3f}" for nv, r in sorted(self.ema_rewards.items()))
            weight_by_nv = {
                nv: float(self.ema_rewards[nv] * (1.0 - self.ema_rewards[nv]) + self.floor_weight)
                for nv in sorted(self.ema_rewards)
            }
            wt_str = ", ".join(f"nv{nv}={w:.3f}" for nv, w in weight_by_nv.items())
            print(f"[AdaptiveDifficultyWeightedSampler] step={self._step}")
            print(f"  EMA rewards: {ema_str}")
            print(f"  Sample weights: {wt_str}")

    def __iter__(self):
        indices = torch.multinomial(
            self.sample_weights,
            num_samples=len(self.data_source),
            replacement=True,
        )
        yield from indices.tolist()

    def __len__(self) -> int:
        return len(self.data_source)
