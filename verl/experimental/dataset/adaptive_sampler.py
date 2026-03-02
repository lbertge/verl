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

Maintains per-difficulty EMA reward estimates updated once per epoch (at the
start of each __iter__ call), and samples problems proportionally to the
variance proxy w(nv) = r*(1-r), which peaks at reward=0.5 (the learning
frontier) and falls to zero at both trivially-easy (r→1) and completely-
unsolvable (r→0) extremes.

EMA updates are applied once per epoch rather than per step so that:
  1. The weight change and the resampling happen at the same moment.
  2. The reward estimate is based on the full epoch (~40 samples/level)
     rather than a noisy per-step mini-batch (~4 samples/level).
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

    Rewards are accumulated across every training step in the epoch via update(),
    then a single EMA step is applied at the top of __iter__ (epoch boundary) before
    drawing the new epoch's sample indices.

    Config keys (all under data.sampler.*):
        ema_alpha (float): EMA decay for reward estimates per epoch. Default 0.1.
        floor_weight (float): Minimum per-difficulty weight to prevent starvation. Default 0.05.
    """

    def __init__(self, data_source, data_config: DictConfig):
        self.data_source = data_source
        sampler_config = data_config.get("sampler", {})
        self.ema_alpha = float(sampler_config.get("ema_alpha", 0.1))
        self.floor_weight = float(sampler_config.get("floor_weight", 0.05))

        # Read num_vars for every problem in the dataset
        n = len(data_source)
        self.sample_num_vars: list[int | None] = []
        for i in range(n):
            row = data_source[i]
            extra_info = row.get("extra_info", {})
            nv = extra_info.get("num_vars") if isinstance(extra_info, dict) else None
            self.sample_num_vars.append(int(nv) if nv is not None else None)

        # Collect all unique difficulty levels and problem counts per level
        self.all_nvs: list[int] = sorted({nv for nv in self.sample_num_vars if nv is not None})
        self.count_by_nv: dict[int, int] = {nv: self.sample_num_vars.count(nv) for nv in self.all_nvs}

        # Initialize EMA reward estimates to 0.5 (neutral → equal initial weights)
        self.ema_rewards: dict[int, float] = {nv: 0.5 for nv in self.all_nvs}

        # Build initial per-sample weight tensor
        self.sample_weights = self._compute_sample_weights()

        # Reward accumulator: filled by update() each step, consumed by __iter__ each epoch
        self._epoch_rewards: dict[int, list[float]] = defaultdict(list)

        # Set by __iter__ after applying the EMA update; cleared by get_metrics() after logging.
        # Ensures metrics are logged exactly once per epoch (on the first step after the boundary).
        self._metrics_pending: bool = False

        # Accumulates {global_step: {nv: normalized_sampling_prob}} for heatmap
        self.sampling_history: dict[int, dict[int, float]] = {}

        self._epoch = 0
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
        """Accumulate per-rollout rewards for the current training step.

        Called automatically by ray_trainer after every training step.
        Rewards are read from batch.batch["token_level_rewards"] (summed over tokens),
        falling back to batch.non_tensor_batch["reward"] if present.
        The EMA is NOT updated here; it is applied once per epoch in __iter__.
        """
        rewards = batch.non_tensor_batch.get("reward", None)
        if rewards is None:
            # OR-Bench (and most VERL reward fns) write rewards to a token-level tensor,
            # not to non_tensor_batch["reward"].  Sum across the token dimension to get
            # one scalar reward per rollout, matching the extra_info length.
            if "token_level_rewards" in batch.batch:
                rewards = batch.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
            else:
                return

        extra_infos = batch.non_tensor_batch.get("extra_info", None)
        if extra_infos is None:
            return

        for reward, ei in zip(rewards, extra_infos):
            nv = ei.get("num_vars") if isinstance(ei, dict) else None
            if nv is not None:
                self._epoch_rewards[int(nv)].append(float(reward))

    def get_metrics(self, global_step: int) -> dict:
        """Return loggable metrics for the current sampler state.

        Called by ray_trainer after update() every training step, but only returns
        data on the first step after each epoch boundary (when __iter__ applied a new
        EMA update). Returns an empty dict on all other steps to avoid logging stale
        values to wandb between weight updates.

        Returns (on epoch boundary steps) dict with keys:
              train/sampler/ema_reward/nv{NN}  — EMA reward estimate per difficulty
              train/sampler/prob/nv{NN}         — normalized sampling probability (sums to 1)
              train/sampler/weights_heatmap     — wandb.Image of accumulated heatmap
        """
        if not self._metrics_pending:
            return {}
        self._metrics_pending = False

        total_weight = float(self.sample_weights.sum())
        # prob[nv] = (count_at_nv * weight_per_sample_at_nv) / total_weight,
        # i.e. the fraction of draws that come from difficulty level nv. Sums to 1.
        prob_by_nv: dict[int, float] = {}
        for nv in self.all_nvs:
            r = self.ema_rewards[nv]
            w = r * (1.0 - r) + self.floor_weight
            nv_total_w = self.count_by_nv[nv] * w
            prob_by_nv[nv] = nv_total_w / total_weight if total_weight > 0 else 1.0 / len(self.all_nvs)

        # Accumulate history for heatmap
        self.sampling_history[global_step] = prob_by_nv

        metrics: dict = {}
        for nv in sorted(self.all_nvs):
            metrics[f"train/sampler/ema_reward/nv{nv:02d}"] = self.ema_rewards[nv]
            metrics[f"train/sampler/prob/nv{nv:02d}"] = prob_by_nv[nv]

        heatmap = self._generate_sampling_heatmap()
        if heatmap is not None:
            metrics["train/sampler/weights_heatmap"] = heatmap

        return metrics

    def _generate_sampling_heatmap(self):
        """Generate a matplotlib heatmap image of sampling probability vs (step, num_vars)."""
        try:
            import wandb
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            history = self.sampling_history
            steps = sorted(history.keys())
            all_nvs = sorted({nv for probs in history.values() for nv in probs})

            if not steps or not all_nvs:
                return None

            data = np.full((len(all_nvs), len(steps)), np.nan)
            for col, step in enumerate(steps):
                for row, nv in enumerate(all_nvs):
                    if nv in history[step]:
                        data[row, col] = history[step][nv]

            fig, ax = plt.subplots(figsize=(max(6, len(steps) * 0.6), max(4, len(all_nvs) * 0.4)))
            # Use a perceptually distinct colormap from the reward heatmap (plasma vs viridis)
            im = ax.imshow(data, aspect="auto", vmin=0.0, cmap="plasma", origin="lower")
            ax.set_xticks(range(len(steps)))
            ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(all_nvs)))
            ax.set_yticklabels([str(nv) for nv in all_nvs])
            ax.set_xlabel("Training Step")
            ax.set_ylabel("num_vars (difficulty)")
            ax.set_title("Sampling Probability by Difficulty over Training")
            plt.colorbar(im, ax=ax, label="Sampling Probability")
            plt.tight_layout()

            img = wandb.Image(fig)
            plt.close(fig)
            return img
        except Exception:
            return None

    def __iter__(self):
        # Apply one EMA update from the rewards accumulated over the previous epoch,
        # then resample with the updated weights.
        if self._epoch_rewards:
            for nv, nv_rewards in self._epoch_rewards.items():
                if nv in self.ema_rewards:
                    epoch_mean = float(np.mean(nv_rewards))
                    self.ema_rewards[nv] = (1.0 - self.ema_alpha) * self.ema_rewards[nv] + self.ema_alpha * epoch_mean
            self._epoch_rewards.clear()
            self.sample_weights = self._compute_sample_weights()

        self._epoch += 1
        self._metrics_pending = True
        ema_str = ", ".join(f"nv{nv}={r:.3f}" for nv, r in sorted(self.ema_rewards.items()))
        print(f"[AdaptiveDifficultyWeightedSampler] epoch={self._epoch}, EMA rewards: {ema_str}")

        indices = torch.multinomial(
            self.sample_weights,
            num_samples=len(self.data_source),
            replacement=True,
        )
        yield from indices.tolist()

    def __len__(self) -> int:
        return len(self.data_source)
