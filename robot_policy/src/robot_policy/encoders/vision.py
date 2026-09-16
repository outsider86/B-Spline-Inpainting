from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VisionBatch:
    fused_patches: torch.Tensor
    dino_pixels: torch.Tensor
    siglip_pixels: torch.Tensor


class FrozenDinoSigLIP(nn.Module):
    """Reference-aligned frozen DINOv2+SigLIP dense feature fusion.

    It follows DiscreteDiffusionVLA's second-to-last-layer concatenation and
    separate checkpoint normalization. A fixed spatial average pool reduces
    cache size before the trainable policy projector.
    """

    def __init__(self, cfg: Any):
        super().__init__()
        import timm
        self.cfg = cfg
        self.dino = timm.create_model(cfg.vision.dino_model, pretrained=cfg.vision.pretrained, num_classes=0, img_size=cfg.vision.image_size)
        self.siglip = timm.create_model(cfg.vision.siglip_model, pretrained=cfg.vision.pretrained, num_classes=0, img_size=cfg.vision.image_size)
        self.dino_cfg = timm.data.resolve_model_data_config(self.dino)
        self.siglip_cfg = timm.data.resolve_model_data_config(self.siglip)
        self.output_dim = self.dino.embed_dim + self.siglip.embed_dim
        self.native_grid = cfg.vision.image_size // 14
        for module in (self.dino, self.siglip):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval(); self.siglip.eval()
        return self

    def preprocess(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError("RGB input must have shape [B,H,W,3]")
        x = rgb.permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(self.cfg.vision.image_size, self.cfg.vision.image_size), mode="bicubic", align_corners=False, antialias=True)
        def normalize(value: torch.Tensor, data_cfg: dict[str, Any]) -> torch.Tensor:
            mean = value.new_tensor(data_cfg["mean"])[None, :, None, None]
            std = value.new_tensor(data_cfg["std"])[None, :, None, None]
            return (value - mean) / std
        return normalize(x, self.dino_cfg), normalize(x, self.siglip_cfg)

    def _patches(self, model: nn.Module, pixels: torch.Tensor) -> torch.Tensor:
        index = len(model.blocks) + int(self.cfg.vision.extraction_layer_offset)
        result = model.get_intermediate_layers(pixels, n={index})
        return result[0]

    @torch.inference_mode()
    def forward(self, rgb: torch.Tensor) -> VisionBatch:
        dino_pixels, siglip_pixels = self.preprocess(rgb)
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16, enabled=rgb.device.type == "cuda"):
            dino = self._patches(self.dino, dino_pixels)
            siglip = self._patches(self.siglip, siglip_pixels)
        if dino.shape[1] != siglip.shape[1]:
            raise RuntimeError(f"unaligned patch counts: DINO={dino.shape}, SigLIP={siglip.shape}")
        fused = torch.cat([dino, siglip], dim=-1)
        b, n, d = fused.shape
        side = int(round(n ** 0.5))
        if side * side != n:
            raise RuntimeError(f"expected dense square patch grid, got {n} tokens")
        fused = fused.transpose(1, 2).reshape(b, d, side, side)
        fused = F.adaptive_avg_pool2d(fused, self.cfg.vision.pooled_grid)
        fused = fused.flatten(2).transpose(1, 2).contiguous()
        return VisionBatch(fused, dino_pixels, siglip_pixels)


def decode_video(path: Path):
    import av
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def prepare_vision_features(cfg: Any, rank: int = 0, world_size: int = 1, batch_size: int = 16) -> dict[str, Any]:
    import json
    dataset = Path(cfg.data.dataset_path).resolve()
    prepared = Path(cfg.data.prepared_path).resolve()
    output = (
        Path(cfg.data.vision_cache_path).resolve()
        if cfg.data.vision_cache_path
        else prepared / "vision"
    )
    output.mkdir(parents=True, exist_ok=True)
    splits = json.loads((prepared / "splits.json").read_text())
    episode_ids = sorted(i for values in splits.values() for i in values)
    model = FrozenDinoSigLIP(cfg).cuda().eval()
    completed = []
    for eid in episode_ids:
        if eid % world_size != rank:
            continue
        target = output / f"episode_{eid:06d}.npy"
        if target.exists():
            completed.append(eid); continue
        camera_features = []
        frame_count = None
        for camera in cfg.data.camera_keys:
            video = dataset / "videos" / "chunk-000" / camera / f"episode_{eid:06d}.mp4"
            if not video.exists():
                raise FileNotFoundError(video)
            chunks, pending = [], []
            for frame in decode_video(video):
                pending.append(frame)
                if len(pending) == batch_size:
                    rgb = torch.from_numpy(np.stack(pending)).cuda(non_blocking=True)
                    chunks.append(model(rgb).fused_patches.cpu().to(torch.float16).numpy())
                    pending.clear()
            if pending:
                rgb = torch.from_numpy(np.stack(pending)).cuda(non_blocking=True)
                chunks.append(model(rgb).fused_patches.cpu().to(torch.float16).numpy())
            features = np.concatenate(chunks)
            frame_count = len(features) if frame_count is None else frame_count
            if len(features) != frame_count:
                raise RuntimeError(f"camera frame mismatch in episode {eid}")
            camera_features.append(features)
        action_file = prepared / "actions" / f"episode_{eid:06d}.npz"
        expected = len(np.load(action_file)["state"])
        if frame_count != expected:
            raise RuntimeError(f"video/parquet frame mismatch in episode {eid}: {frame_count} != {expected}")
        np.save(target, np.stack(camera_features, axis=1), allow_pickle=False)
        completed.append(eid)
    revision = {
        "dino": cfg.vision.dino_model, "siglip": cfg.vision.siglip_model,
        "pretrained": cfg.vision.pretrained, "image_size": cfg.vision.image_size,
        "geometry": "shared naive bicubic resize", "normalization": "per-checkpoint timm mean/std",
        "extraction": "second-to-last block dense patches; prefix/register tokens excluded by get_intermediate_layers",
        "fusion": "channel concatenation then fixed adaptive average pooling",
        "pooled_grid": cfg.vision.pooled_grid, "cache_dtype": cfg.vision.cache_dtype,
        "rank": rank, "world_size": world_size, "completed_episodes": completed,
    }
    (output / f"worker_{rank:02d}_manifest.json").write_text(json.dumps(revision, indent=2) + "\n")
    return revision
