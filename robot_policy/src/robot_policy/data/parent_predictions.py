from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from robot_policy.config import load_config
from robot_policy.data.dataset import PreparedPolicyDataset, collate_policy_batch
from robot_policy.policies import load_policy_checkpoint


@torch.inference_mode()
def build(cfg, checkpoint: str, output: str | Path, batch_size: int) -> dict:
    device = torch.device("cuda")
    torch.manual_seed(cfg.train.seed)
    model, payload = load_policy_checkpoint(checkpoint, cfg, device)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    episode_count = frame_count = 0
    for split in ("train", "val", "test"):
        dataset = PreparedPolicyDataset(cfg.data.prepared_path, split)
        by_episode: dict[int, list[int]] = {}
        for index, (episode, _) in enumerate(dataset.index):
            by_episode.setdefault(episode, []).append(index)
        for episode, indices in by_episode.items():
            chunks = []
            for start in range(0, len(indices), batch_size):
                batch = collate_policy_batch([dataset[i] for i in indices[start:start + batch_size]])
                model_batch = {"vision_features": batch["vision_features"].to(device), "state": batch["state"].to(device)}
                chunks.append(model.sample(model_batch).cpu())
            prediction = torch.cat(chunks).numpy()
            if cfg.policy.architecture != "fm":
                prediction = prediction.astype(np.uint8)
            else:
                prediction = prediction.astype(np.float32)
            np.save(root / f"episode_{episode:06d}.npy", prediction)
            episode_count += 1
            frame_count += len(indices)
    checkpoint_hash = sha256(Path(checkpoint).read_bytes()).hexdigest()
    manifest = {
        "architecture": cfg.policy.architecture,
        "action_representation": cfg.data.action_representation,
        "parent_checkpoint": str(Path(checkpoint).resolve()),
        "parent_checkpoint_sha256": checkpoint_hash,
        "episodes": episode_count,
        "frames": frame_count,
        "batch_size": batch_size,
        "seed": cfg.train.seed,
        "exact_integer_tokens": cfg.policy.architecture != "fm",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def hash_keyed_output(cfg, checkpoint: str) -> Path:
    checkpoint_hash = sha256(Path(checkpoint).read_bytes()).hexdigest()
    return Path(cfg.data.prepared_path) / "parent_predictions" / cfg.policy.architecture / checkpoint_hash


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/raw_actions.yaml")
    parser.add_argument("--architecture", required=True, choices=["fm", "discrete_layerwise", "discrete_joint"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, [f"policy.architecture={args.architecture}"])
    output = Path(args.output) if args.output else hash_keyed_output(cfg, args.checkpoint)
    print(json.dumps(build(cfg, args.checkpoint, output, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
