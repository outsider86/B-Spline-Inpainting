from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PreparedPolicyDataset(Dataset):
    def __init__(self, prepared_path: str | Path, split: str, cache_episodes: int = 3, include_rtc_history: bool = False,
                 parent_prediction_path: str | Path | None = None, *, observation_source: str = "features",
                 observation_horizon: int = 1, rgb_cache_path: str | Path | None = None,
                 vision_cache_path: str | Path | None = None):
        self.root = Path(prepared_path).resolve()
        manifest = json.loads((self.root / "action_manifest.json").read_text())
        configured_vision = manifest.get("config", {}).get("data", {}).get("vision_cache_path")
        self.vision_root = (
            Path(vision_cache_path).resolve()
            if vision_cache_path
            else Path(configured_vision).resolve()
            if configured_vision
            else self.root / "vision"
        )
        self.observation_source = observation_source
        self.observation_horizon = int(observation_horizon)
        self.rgb_root = Path(rgb_cache_path).resolve() if rgb_cache_path else self.root / "rgb"
        if self.observation_source not in {"features", "rgb"}:
            raise ValueError("observation_source must be 'features' or 'rgb'")
        if self.observation_horizon not in {1, 2}:
            raise ValueError("observation_horizon must be 1 or 2")
        splits = json.loads((self.root / "splits.json").read_text())
        if split not in splits:
            raise ValueError(f"unknown split {split!r}")
        self.episode_ids = splits[split]
        self.cache_episodes = cache_episodes
        self.include_rtc_history = include_rtc_history
        self.parent_prediction_root = Path(parent_prediction_path).resolve() if parent_prediction_path else None
        encoder = json.loads((self.root / "encoder.json").read_text())
        self.rtc_history_steps = encoder.get("rtc_history_steps", [2, 4, 6, 8, 10])
        self._cache: OrderedDict[int, tuple[dict, np.ndarray, np.ndarray | None]] = OrderedDict()
        self.index: list[tuple[int, int]] = []
        for eid in self.episode_ids:
            with np.load(self.root / "actions" / f"episode_{eid:06d}.npz") as data:
                self.index.extend((eid, i) for i in range(len(data["state"])))

    def __len__(self) -> int:
        return len(self.index)

    def _episode(self, eid: int) -> tuple[dict, np.ndarray, np.ndarray | None]:
        if eid not in self._cache:
            action_path = self.root / "actions" / f"episode_{eid:06d}.npz"
            observation_path = (
                self.vision_root if self.observation_source == "features" else self.rgb_root
            ) / f"episode_{eid:06d}.npy"
            if not observation_path.exists():
                command = "prepare_observations" if self.observation_source == "features" else "prepare_rgb_cache"
                raise FileNotFoundError(f"missing observation cache {observation_path}; run {command}")
            actions = {k: v for k, v in np.load(action_path).items()}
            observation = np.load(observation_path, mmap_mode="r")
            parent = None
            if self.parent_prediction_root is not None:
                parent_path = self.parent_prediction_root / f"episode_{eid:06d}.npy"
                if not parent_path.exists():
                    raise FileNotFoundError(f"missing cached parent prediction {parent_path}")
                parent = np.load(parent_path, mmap_mode="r")
            self._cache[eid] = actions, observation, parent
            while len(self._cache) > self.cache_episodes:
                self._cache.popitem(last=False)
        self._cache.move_to_end(eid)
        return self._cache[eid]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        eid, frame = self.index[item]
        actions, observation, parent = self._episode(eid)
        history = np.arange(frame - self.observation_horizon + 1, frame + 1).clip(0)
        if self.observation_source == "rgb":
            observation_item = {
                "images": torch.from_numpy(np.array(observation[history], copy=True)),
                "state": torch.from_numpy(actions["normalized_state"][history]),
            }
        else:
            observation_item = {
                "vision_features": torch.from_numpy(
                    np.array(observation[history], copy=True)
                ),
                "state": torch.from_numpy(actions["normalized_state"][history]),
            }
            if self.observation_horizon == 1:
                observation_item = {
                    key: value[0] for key, value in observation_item.items()
                }
        item = {
            **observation_item,
            "continuous_target": torch.from_numpy(actions["continuous_target"][frame]),
            "discrete_target": torch.from_numpy(actions["discrete_target"][frame].astype(np.int64)),
            "control_valid_mask": torch.from_numpy(actions["control_valid_mask"][frame]),
            "action_valid_mask": torch.from_numpy(actions["action_valid_mask"][frame]),
            "raw_action": torch.from_numpy(actions["action"][frame]),
            "target_trajectory": torch.from_numpy(actions["action"][np.minimum(np.arange(frame, frame + 30), len(actions["action"]) - 1)]),
            "normalized_target_trajectory": torch.from_numpy(actions["normalized_action"][np.minimum(np.arange(frame, frame + 30), len(actions["normalized_action"]) - 1)]),
            "encoder_reconstruction": torch.from_numpy(actions["reconstruction"][frame]),
            "quantized_reconstruction": torch.from_numpy(actions["quantized_reconstruction"][frame]),
            "episode_id": torch.tensor(eid), "frame_index": torch.tensor(frame),
            "timestamp": torch.tensor(actions["timestamp"][frame]),
        }
        if self.include_rtc_history:
            previous = [max(0, frame - delay) for delay in self.rtc_history_steps]
            if self.observation_source == "rgb":
                previous_histories = np.stack(
                    [
                        np.arange(point - self.observation_horizon + 1, point + 1).clip(0)
                        for point in previous
                    ]
                )
                item["previous_images"] = torch.from_numpy(
                    np.array(observation[previous_histories], copy=True)
                )
                item["previous_state"] = torch.from_numpy(
                    actions["normalized_state"][previous_histories]
                )
            else:
                previous_histories = np.stack(
                    [
                        np.arange(point - self.observation_horizon + 1, point + 1).clip(0)
                        for point in previous
                    ]
                )
                item["previous_vision_features"] = torch.from_numpy(
                    np.array(observation[previous_histories], copy=True)
                )
                item["previous_state"] = torch.from_numpy(
                    actions["normalized_state"][previous_histories]
                )
                if self.observation_horizon == 1:
                    item["previous_vision_features"] = item["previous_vision_features"][:, 0]
                    item["previous_state"] = item["previous_state"][:, 0]
            item["has_previous"] = torch.tensor([frame >= delay for delay in self.rtc_history_steps])
            if parent is not None:
                # Preserve the cache's native contract: FM parents are float32
                # controls, while discrete parents are exact uint8 tokens.
                item["previous_parent_prediction"] = torch.from_numpy(
                    np.array(parent[previous], copy=True)
                )
        return item


def collate_policy_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def create_policy_dataset(cfg, split: str, **kwargs) -> PreparedPolicyDataset:
    return PreparedPolicyDataset(
        cfg.data.prepared_path,
        split,
        observation_source=cfg.data.observation_source,
        observation_horizon=cfg.data.observation_horizon,
        rgb_cache_path=cfg.data.rgb_cache_path,
        vision_cache_path=cfg.data.vision_cache_path,
        **kwargs,
    )
