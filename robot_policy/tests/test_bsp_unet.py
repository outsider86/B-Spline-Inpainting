from __future__ import annotations

import json
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

from robot_policy.config import Config
from robot_policy.data.dataset import create_policy_dataset
from robot_policy.policies import create_policy
from robot_policy.policies.common import mixed_block_corruption


def _prepared(tmp_path: Path, representation: str) -> Path:
    root = tmp_path / representation
    (root / "actions").mkdir(parents=True)
    (root / "rgb").mkdir()
    config = (
        {"action_horizon": 30, "action_dim": 7, "vocab_size": 256}
        if representation == "raw"
        else {
            "chunk_size": 30,
            "degree": 3,
            "num_basis": 18,
            "span_length_steps": 2,
            "action_dim": 7,
            "vocab_size": 256,
        }
    )
    low = [-1.0] * 7 if representation == "raw" else np.full((18, 7), -1.0).tolist()
    high = [1.0] * 7 if representation == "raw" else np.full((18, 7), 1.0).tolist()
    (root / "encoder.json").write_text(
        json.dumps({"config": config, "calibration": {"low": low, "high": high}})
    )
    (root / "action_manifest.json").write_text(json.dumps({"config": {}}))
    (root / "splits.json").write_text(json.dumps({"train": [0], "val": [], "test": []}))
    frames = 3
    steps = 30 if representation == "raw" else 18
    np.savez(
        root / "actions" / "episode_000000.npz",
        state=np.zeros((frames, 7), np.float32),
        normalized_state=np.arange(frames * 7, dtype=np.float32).reshape(frames, 7),
        continuous_target=np.zeros((frames, steps, 7), np.float32),
        discrete_target=np.zeros((frames, steps, 7), np.uint8),
        control_valid_mask=np.ones((frames, steps, 7), bool),
        action_valid_mask=np.ones((frames, 30), bool),
        action=np.zeros((frames, 7), np.float32),
        normalized_action=np.zeros((frames, 7), np.float32),
        reconstruction=np.zeros((frames, 30, 7), np.float32),
        quantized_reconstruction=np.zeros((frames, 30, 7), np.float32),
        timestamp=np.arange(frames, dtype=np.float64),
    )
    rgb = np.arange(frames * 2 * 3 * 8 * 8, dtype=np.uint8).reshape(frames, 2, 3, 8, 8)
    np.save(root / "rgb" / "episode_000000.npy", rgb)
    return root


def _cfg(tmp_path: Path, architecture: str, representation: str, horizon: int) -> Config:
    cfg = Config()
    cfg.data.prepared_path = str(_prepared(tmp_path, representation))
    cfg.data.rgb_cache_path = str(Path(cfg.data.prepared_path) / "rgb")
    cfg.data.observation_source = "rgb"
    cfg.data.observation_horizon = horizon
    cfg.data.action_representation = representation
    cfg.vision.image_size = 32
    cfg.vision.crop_size = 28
    cfg.vision.spatial_keypoints = 4
    cfg.policy.architecture = architecture
    cfg.policy.model_size = "BSP-UNet"
    # Tiny U-Net for unit tests; full BSP dimensions are covered by the GPU smoke audit.
    cfg.policy.unet_down_dims = (16, 32, 64)
    cfg.policy.unet_time_dim = 16
    cfg.policy.unet_kernel_size = 3
    cfg.policy.unet_groups = 4
    cfg.policy.discrete_embed_dim = 4
    cfg.policy.fm_steps = 2
    cfg.policy.discrete_rounds = 2
    return cfg


def test_rgb_dataset_returns_consecutive_history_and_clamps_first_frame(tmp_path):
    cfg = _cfg(tmp_path, "bsp_unet_fm", "raw", 2)
    dataset = create_policy_dataset(cfg, "train")
    first = dataset[0]
    last = dataset[2]
    assert first["images"].shape == (2, 2, 3, 8, 8)
    assert torch.equal(first["images"][0], first["images"][1])
    assert torch.equal(first["state"][0], first["state"][1])
    assert not torch.equal(last["images"][0], last["images"][1])
    assert last["state"][0, 0].item() == 7
    assert last["state"][1, 0].item() == 14


def test_parent_cache_preserves_continuous_control_dtype(tmp_path):
    cfg = _cfg(tmp_path, "bsp_unet_fm", "raw", 1)
    parent = tmp_path / "parent"
    parent.mkdir()
    values = np.linspace(-0.75, 0.75, 3 * 30 * 7, dtype=np.float32).reshape(3, 30, 7)
    np.save(parent / "episode_000000.npy", values)
    dataset = create_policy_dataset(
        cfg, "train", include_rtc_history=True, parent_prediction_path=parent
    )
    cached = dataset[2]["previous_parent_prediction"]
    assert cached.dtype == torch.float32
    assert torch.any(cached != cached.round())


@pytest.mark.parametrize("architecture", ["bsp_unet_fm", "bsp_unet_discrete"])
@pytest.mark.parametrize("representation", ["raw", "bspline"])
@pytest.mark.parametrize("horizon", [1, 2])
def test_bsp_unet_trains_generates_and_preserves_hard_mask(
    tmp_path, architecture, representation, horizon
):
    cfg = _cfg(tmp_path, architecture, representation, horizon)
    model = create_policy(cfg)
    steps = 30 if representation == "raw" else 18
    batch = {
        "images": torch.randint(0, 256, (1, horizon, 2, 3, 32, 32), dtype=torch.uint8),
        "state": torch.randn(1, horizon, 7),
        "continuous_target": torch.rand(1, steps, 7) * 2 - 1,
        "discrete_target": torch.randint(0, 256, (1, steps, 7)),
        "control_valid_mask": torch.ones(1, steps, 7, dtype=torch.bool),
        "action_valid_mask": torch.ones(1, 30, dtype=torch.bool),
        "normalized_target_trajectory": torch.rand(1, 30, 7) * 2 - 1,
    }
    result = model(batch)
    result["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.observation.parameters())

    fixed = torch.zeros(1, steps, 7, dtype=torch.bool)
    fixed[:, :3] = True
    if architecture == "bsp_unet_fm":
        prefix = torch.rand(1, steps, 7) * 2 - 1
        model.zero_grad(set_to_none=True)
        rtc_result = model(
            batch,
            {"fixed_mask": fixed, "prefix_values": prefix},
        )
        rtc_result["loss"].backward()
        assert torch.isfinite(rtc_result["loss"])
        assert any(
            parameter.grad is not None
            for parameter in model.unet.parameters()
        )
        generated = model.sample(batch, steps=2, prefix_values=prefix, fixed_mask=fixed)
    else:
        prefix = torch.randint(0, 256, (1, steps, 7))
        generated, trace = model.sample(
            batch, rounds=2, prefix_values=prefix, fixed_mask=fixed, return_trace=True
        )
        assert len(trace) == 2
        assert not (generated == model.mask_id).any()
    assert generated.shape == (1, steps, 7)
    assert torch.equal(generated[fixed], prefix[fixed])


def test_two_cameras_have_independent_scratch_resnets(tmp_path):
    model = create_policy(_cfg(tmp_path, "bsp_unet_fm", "raw", 1))
    cameras = model.observation.camera_encoders
    assert len(cameras) == 2
    assert cameras[0].backbone[0].weight.data_ptr() != cameras[1].backbone[0].weight.data_ptr()
    assert all(parameter.requires_grad for parameter in model.observation.parameters())


def test_full_mask_corruption_masks_every_valid_token():
    tokens = torch.arange(18).reshape(2, 9)
    valid = torch.ones_like(tokens, dtype=torch.bool)
    valid[:, -1] = False
    corrupted, supervised = mixed_block_corruption(
        tokens, valid, block_size=3, full_mask_probability=1.0
    )
    assert torch.equal(supervised, valid)
    assert torch.equal(corrupted[valid], torch.full_like(corrupted[valid], 256))
    assert torch.equal(corrupted[~valid], tokens[~valid])


def test_full_mask_probability_is_validated():
    cfg = Config()
    cfg.policy.discrete_full_mask_probability = 1.01
    with pytest.raises(ValueError, match="discrete_full_mask_probability"):
        cfg.validate()


def test_raw_fm_ttrtc_sets_fixed_rows_to_endpoint_time_and_masks_their_loss(tmp_path):
    cfg = _cfg(tmp_path, "bsp_unet_fm", "raw", 1)
    model = create_policy(cfg)
    target = torch.randn(2, 30, 7)
    fixed = torch.zeros_like(target, dtype=torch.bool)
    fixed[0, :3] = True
    # The second item is the reference delay-zero training case.
    captured = {}

    def capture_velocity(self, x, batch, time):
        captured["x"] = x.detach().clone()
        captured["time"] = time.detach().clone()
        return torch.zeros_like(x)

    model.velocity = MethodType(capture_velocity, model)
    batch = {
        "continuous_target": target,
        "control_valid_mask": torch.ones_like(fixed),
        "action_valid_mask": torch.ones(2, 30, dtype=torch.bool),
        "normalized_target_trajectory": target.clone(),
    }
    result = model.loss(
        batch,
        {"fixed_mask": fixed, "prefix_values": target.clone()},
    )

    assert torch.equal(captured["time"][0, :3], torch.ones(3))
    assert torch.all(captured["time"][0, 3:] < 1)
    assert torch.all(captured["time"][1] < 1)
    torch.testing.assert_close(captured["x"][0, :3], target[0, :3])
    assert torch.isfinite(result["loss"])
