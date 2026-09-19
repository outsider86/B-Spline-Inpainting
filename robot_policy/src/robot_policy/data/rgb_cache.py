from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F


def _decode_resize(path: Path, size: int, batch_size: int = 128) -> np.ndarray:
    frames: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == batch_size:
                chunks.append(_resize(frames, size))
                frames.clear()
    if frames:
        chunks.append(_resize(frames, size))
    return np.concatenate(chunks) if chunks else np.empty((0, 3, size, size), np.uint8)


def _resize(frames: list[np.ndarray], size: int) -> np.ndarray:
    tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    resized = F.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )
    return resized.round().clamp(0, 255).to(torch.uint8).numpy()


def prepare_rgb_cache(
    cfg: Any, overwrite: bool = False, rank: int = 0, world_size: int = 1
) -> dict[str, Any]:
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("rank/world_size must satisfy 0 <= rank < world_size")
    dataset = Path(cfg.data.dataset_path).resolve()
    prepared = Path(cfg.data.prepared_path).resolve()
    output = (
        Path(cfg.data.rgb_cache_path).resolve()
        if cfg.data.rgb_cache_path
        else prepared / "rgb"
    )
    output.mkdir(parents=True, exist_ok=True)
    splits = json.loads((prepared / "splits.json").read_text())
    all_episode_ids = sorted({int(x) for values in splits.values() for x in values})
    episode_ids = all_episode_ids[rank::world_size]
    total = 0
    shapes: dict[str, list[int]] = {}
    for episode in episode_ids:
        target = output / f"episode_{episode:06d}.npy"
        if target.exists() and not overwrite:
            cached = np.load(target, mmap_mode="r")
            total += len(cached)
            shapes[str(episode)] = list(cached.shape)
            continue
        cameras = []
        for camera in cfg.data.camera_keys:
            path = dataset / "videos" / "chunk-000" / camera / f"episode_{episode:06d}.mp4"
            if not path.is_file():
                raise FileNotFoundError(path)
            cameras.append(_decode_resize(path, cfg.vision.image_size))
        lengths = {len(value) for value in cameras}
        if len(lengths) != 1:
            raise RuntimeError(f"camera length mismatch in episode {episode}: {lengths}")
        with np.load(prepared / "actions" / f"episode_{episode:06d}.npz") as action_data:
            expected_frames = len(action_data["state"])
        if next(iter(lengths)) != expected_frames:
            raise RuntimeError(
                f"RGB/action length mismatch in episode {episode}: "
                f"RGB={next(iter(lengths))}, actions={expected_frames}"
            )
        stacked = np.stack(cameras, axis=1)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, stacked)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        total += len(stacked)
        shapes[str(episode)] = list(stacked.shape)
    manifest = {
        "format": "uint8_rgb_chw",
        "image_size": cfg.vision.image_size,
        "camera_keys": list(cfg.data.camera_keys),
        "episodes": len(episode_ids),
        "frames": total,
        "episode_shapes": shapes,
        "source": str(dataset),
        "rank": rank,
        "world_size": world_size,
        "complete": world_size == 1 and len(episode_ids) == len(all_episode_ids),
    }
    manifest_name = "manifest.json" if world_size == 1 else f"worker_{rank}_manifest.json"
    (output / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
