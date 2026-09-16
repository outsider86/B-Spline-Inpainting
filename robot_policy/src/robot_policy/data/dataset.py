from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PreparedPolicyDataset(Dataset):
    def __init__(self, prepared_path: str | Path, split: str, cache_episodes: int = 3, include_rtc_history: bool = False,
                 parent_prediction_path: str | Path | None = None):
        self.root = Path(prepared_path).resolve()
        manifest = json.loads((self.root / "action_manifest.json").read_text())
        configured_vision = manifest.get("config", {}).get("data", {}).get("vision_cache_path")
        self.vision_root = Path(configured_vision).resolve() if configured_vision else self.root / "vision"
        splits = json.loads((self.root / "splits.json").read_text())
        if split not in splits:
            raise ValueError(f"unknown split {split!r}")
        self.episode_ids = splits[split]
        self.cache_episodes = cache_episodes
        self.include_rtc_history = include_rtc_history
        self.parent_prediction_root = Path(parent_prediction_path).resolve() if parent_prediction_path else None
        encoder = json.loads((self.root / "encoder.json").read_text())
        self.rtc_history_steps = encoder.get("rtc_history_steps", [2, 4, 6, 8, 10])
        self._cache: OrderedDict[int, tuple[dict, dict, np.ndarray | None]] = OrderedDict()
        self.index: list[tuple[int, int]] = []
        for eid in self.episode_ids:
            with np.load(self.root / "actions" / f"episode_{eid:06d}.npz") as data:
                self.index.extend((eid, i) for i in range(len(data["state"])))

    def __len__(self) -> int:
        return len(self.index)

    def _episode(self, eid: int) -> tuple[dict, dict, np.ndarray | None]:
        if eid not in self._cache:
            action_path = self.root / "actions" / f"episode_{eid:06d}.npz"
            vision_path = self.vision_root / f"episode_{eid:06d}.npy"
            if not vision_path.exists():
                raise FileNotFoundError(f"missing vision cache {vision_path}; run prepare_observations")
            actions = {k: v for k, v in np.load(action_path).items()}
            vision = {"features": np.load(vision_path, mmap_mode="r")}
            parent = None
            if self.parent_prediction_root is not None:
                parent_path = self.parent_prediction_root / f"episode_{eid:06d}.npy"
                if not parent_path.exists():
                    raise FileNotFoundError(f"missing cached parent prediction {parent_path}")
                parent = np.load(parent_path, mmap_mode="r")
            self._cache[eid] = actions, vision, parent
            while len(self._cache) > self.cache_episodes:
                self._cache.popitem(last=False)
        self._cache.move_to_end(eid)
        return self._cache[eid]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        eid, frame = self.index[item]
        actions, vision, parent = self._episode(eid)
        item = {
            "vision_features": torch.from_numpy(np.array(vision["features"][frame], copy=True)),
            "state": torch.from_numpy(actions["normalized_state"][frame]),
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
            item["previous_vision_features"] = torch.from_numpy(np.array(vision["features"][previous], copy=True))
            item["previous_state"] = torch.from_numpy(actions["normalized_state"][previous])
            item["has_previous"] = torch.tensor([frame >= delay for delay in self.rtc_history_steps])
            if parent is not None:
                item["previous_parent_prediction"] = torch.from_numpy(np.array(parent[previous], copy=True)).long()
        return item


def collate_policy_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}
