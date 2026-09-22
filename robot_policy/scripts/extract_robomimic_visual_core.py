#!/usr/bin/env python3
"""Extract two RGB VisualCore modules from a trusted RoboMimic v0.1 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from robot_policy.encoders.robomimic_pretrained import extract_visual_core_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifact = extract_visual_core_artifact(args.checkpoint)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(
        f"wrote {output} from {artifact['source_sha256']} "
        f"({', '.join(artifact['source_camera_keys'])})"
    )


if __name__ == "__main__":
    main()
