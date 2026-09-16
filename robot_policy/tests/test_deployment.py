from __future__ import annotations

import asyncio
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robot_policy.config import Config, config_dict
from robot_policy.deployment import msgpack_numpy
from robot_policy.deployment.checkpoint import inspect_checkpoint
from robot_policy.deployment.export_stats import build_piper_statistics, build_start_statistics
from robot_policy.deployment.policy_wrapper import PolicyServerWrapper
from robot_policy.deployment.websocket_server import WebsocketPolicyServer
from robot_policy.policies import create_policy


def _prepared(tmp_path: Path, representation: str) -> Path:
    path = tmp_path / representation
    path.mkdir()
    if representation == "raw":
        encoder = {
            "type": "RawActionSequenceConfig",
            "config": {"action_horizon": 30, "action_dim": 7, "vocab_size": 256},
            "calibration": {"low": [-1.0] * 7, "high": [1.0] * 7},
        }
    else:
        low = np.full((18, 7), -1.5).tolist()
        high = np.full((18, 7), 1.5).tolist()
        encoder = {
            "type": "UniformLeftBSplineConfig",
            "config": {
                "action_dim": 7,
                "chunk_size": 30,
                "frequency_hz": 30.0,
                "degree": 3,
                "num_basis": 18,
                "span_length_steps": 2,
                "vocab_size": 256,
                "regularization": 1e-4,
                "end_padding": True,
                "implementation_version": "uniform_left_direct_fit_v1",
            },
            "calibration": {"low": low, "high": high},
        }
    canonical = json.dumps(encoder, sort_keys=True).encode()
    normalization = {
        "state_mean": [10.0] * 7,
        "state_std": [2.0] * 7,
        "state_min": [8.0] * 7,
        "state_max": [12.0] * 7,
        "action_q01": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 0.0],
        "action_q99": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 1.0],
        "encoder": encoder,
        "encoder_sha256": sha256(canonical).hexdigest(),
    }
    (path / "encoder.json").write_text(json.dumps(encoder) + "\n")
    (path / "normalization.json").write_text(json.dumps(normalization) + "\n")
    return path


def _checkpoint(
    tmp_path: Path, architecture: str, representation: str, training_type: str
) -> tuple[Path, Config]:
    prepared = _prepared(tmp_path, representation)
    cfg = Config()
    cfg.data.action_representation = representation
    cfg.data.prepared_path = str(prepared)
    cfg.policy.architecture = architecture
    cfg.policy.hidden_dim = 24
    cfg.policy.depth = 1
    cfg.policy.heads = 4
    cfg.policy.block_size = 21
    cfg.policy.fm_steps = 2
    cfg.policy.discrete_rounds = 2
    model = create_policy(cfg)
    path = tmp_path / f"{architecture}_{representation}_{training_type}.pt"
    torch.save(
        {
            "architecture": architecture,
            "training_type": training_type,
            "model": model.state_dict(),
            "config": config_dict(cfg),
            "update": 12,
            "parent_checkpoint": "parent.pt" if training_type != "base" else None,
            "code_snapshot_sha256": "abc",
        },
        path,
    )
    return path, cfg


def _example(cfg: Config, *, physical_state: bool = False) -> dict:
    state = np.full(7, 12.0 if physical_state else 1.0, dtype=np.float32)
    return {
        "vision_features": np.zeros(
            (2, cfg.vision.pooled_grid**2, 2176), dtype=np.float32
        ),
        "state": state[None],
        "lang": "Stack the cups.",
    }


@pytest.mark.parametrize("architecture", ["fm", "discrete_layerwise", "discrete_joint"])
@pytest.mark.parametrize("representation", ["raw", "bspline"])
def test_all_policy_families_load_and_serve_base_chunks(
    tmp_path, architecture, representation
):
    checkpoint, cfg = _checkpoint(tmp_path, architecture, representation, "base")
    wrapper = PolicyServerWrapper(
        checkpoint,
        device="cpu",
        precision="fp32",
        binary_gripper=False,
        vision_encoder=object(),
    )
    result = wrapper.predict_action([_example(cfg)], seed=4)
    assert result["actions"].shape == (1, 30, 7)
    assert np.isfinite(result["actions"]).all()
    assert wrapper.metadata["architecture"] == architecture
    assert wrapper.metadata["action_representation"] == representation
    assert wrapper.metadata["action_chunk_size"] == 30
    assert not wrapper.metadata["supports_inference_time_rtc"]
    if representation == "bspline":
        assert result["normalized_control_rows"].shape == (1, 18, 7)
    else:
        assert "normalized_control_rows" not in result


def test_state_coordinate_modes_are_equivalent(tmp_path):
    checkpoint, cfg = _checkpoint(tmp_path, "discrete_layerwise", "raw", "base")
    wrapper = PolicyServerWrapper(
        checkpoint,
        device="cpu",
        precision="fp32",
        binary_gripper=False,
        vision_encoder=object(),
    )
    normalized = wrapper.predict_action([_example(cfg)], seed=2)["actions"]
    physical = wrapper.predict_action(
        [_example(cfg, physical_state=True)], state_coordinates="physical", seed=2
    )["actions"]
    np.testing.assert_allclose(normalized, physical)


def test_exported_statistics_match_legacy_piper_client_transform(tmp_path):
    checkpoint, _ = _checkpoint(tmp_path, "fm", "raw", "base")
    stats = build_piper_statistics(checkpoint)["new_embodiment"]
    assert stats["state"]["min"] == [8.0] * 7
    assert stats["state"]["max"] == [12.0] * 7
    assert stats["action"]["min"] == [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 0.0]


def test_start_statistics_use_first_physical_state_of_each_episode(tmp_path):
    checkpoint, cfg = _checkpoint(tmp_path, "fm", "raw", "base")
    action_dir = Path(cfg.data.prepared_path) / "actions"
    action_dir.mkdir()
    np.savez(action_dir / "episode_000000.npz", state=np.array([[1] * 7, [99] * 7]))
    np.savez(action_dir / "episode_000001.npz", state=np.array([[3] * 7, [-99] * 7]))
    stats = build_start_statistics(checkpoint)
    assert stats["min"] == [1.0] * 7
    assert stats["max"] == [3.0] * 7
    assert stats["median"] == [2.0] * 7


def test_raw_rtc_preserves_shifted_physical_prefix(tmp_path):
    checkpoint, cfg = _checkpoint(tmp_path, "fm", "raw", "ttrtc")
    wrapper = PolicyServerWrapper(
        checkpoint,
        device="cpu",
        precision="fp32",
        binary_gripper=False,
        vision_encoder=object(),
    )
    previous = np.linspace(-0.5, 0.5, 30 * 7, dtype=np.float32).reshape(1, 30, 7)
    # Convert normalized test values to physical action coordinates.
    low = wrapper.action_low.cpu().numpy()
    high = wrapper.action_high.cpu().numpy()
    previous_physical = (previous + 1) * 0.5 * (high - low) + low
    result = wrapper.predict_action_realtime(
        [_example(cfg)], prev_action_chunk=previous_physical, inference_delay=3, seed=9
    )
    np.testing.assert_allclose(
        result["actions"][:, :3], previous_physical[:, 3:6], atol=1e-5, rtol=1e-5
    )
    assert result["inference_metadata"]["fixed_control_rows"] == 3
    assert wrapper.metadata["supports_inference_time_rtc"]
    assert wrapper.metadata["rtc_mode"] == "training_time_hard_prefix"


def test_bspline_rtc_preserves_parameter_prefix_and_rejects_decoded_chunk(tmp_path):
    checkpoint, cfg = _checkpoint(tmp_path, "fm", "bspline", "rtc")
    wrapper = PolicyServerWrapper(
        checkpoint,
        device="cpu",
        precision="fp32",
        binary_gripper=False,
        vision_encoder=object(),
    )
    previous = torch.linspace(-0.8, 0.8, 18 * 7).reshape(1, 18, 7)
    expected = wrapper.codec.shift_and_refit(previous, torch.tensor([3]))
    result = wrapper.predict_action_realtime(
        [_example(cfg)],
        prev_control_rows=previous.numpy(),
        inference_delay=3,
        seed=7,
    )
    assert result["normalized_control_rows"].shape == (1, 18, 7)
    np.testing.assert_allclose(
        result["normalized_control_rows"][:, :5],
        expected.numpy()[:, :5],
        atol=1e-5,
        rtol=1e-5,
    )
    assert result["inference_metadata"]["affected_spans"] == 2
    assert result["inference_metadata"]["fixed_control_rows"] == 5
    assert wrapper.metadata["supports_parameter_row_rtc"]
    with pytest.raises(ValueError, match="decoded prev_action_chunk is forbidden"):
        wrapper.predict_action_realtime(
            [_example(cfg)],
            prev_action_chunk=np.zeros((1, 30, 7), dtype=np.float32),
            inference_delay=3,
        )


def test_contract_validation_rejects_wrong_camera_language_state_and_delay(tmp_path):
    checkpoint, cfg = _checkpoint(tmp_path, "fm", "raw", "ttrtc")
    wrapper = PolicyServerWrapper(
        checkpoint,
        device="cpu",
        precision="fp32",
        vision_encoder=object(),
    )
    bad_state = _example(cfg)
    bad_state["state"] = np.zeros(6)
    with pytest.raises(ValueError, match="finite 7-vector"):
        wrapper.predict_action([bad_state])
    bad_task = _example(cfg)
    bad_task["lang"] = "Do something else."
    with pytest.raises(ValueError, match="does not match fixed task"):
        wrapper.predict_action([bad_task])
    images = {"state": np.zeros(7), "lang": "Stack the cups.", "image": [np.zeros((8, 8, 3), np.uint8)]}
    with pytest.raises(ValueError, match="2 images"):
        wrapper.predict_action([images])
    with pytest.raises(ValueError, match="1..10"):
        wrapper.predict_action_realtime(
            [_example(cfg)], prev_action_chunk=np.zeros((30, 7)), inference_delay=11
        )


def test_equal_resolution_camera_views_use_one_batched_vision_call(tmp_path):
    checkpoint, _ = _checkpoint(tmp_path, "fm", "raw", "base")

    class FakeVision:
        def __init__(self):
            self.shapes = []

        def __call__(self, rgb):
            self.shapes.append(tuple(rgb.shape))
            return SimpleNamespace(
                fused_patches=torch.zeros(len(rgb), 16, 2176, device=rgb.device)
            )

    vision = FakeVision()
    wrapper = PolicyServerWrapper(
        checkpoint,
        device="cpu",
        precision="fp32",
        vision_encoder=vision,
        binary_gripper=False,
    )
    example = {
        "image": [np.zeros((48, 64, 3), np.uint8), np.zeros((48, 64, 3), np.uint8)],
        "state": np.zeros(7, np.float32),
        "lang": "Stack the cups.",
    }
    assert wrapper.predict_action([example])["actions"].shape == (1, 30, 7)
    assert vision.shapes == [(2, 48, 64, 3)]


def test_msgpack_numpy_and_router_are_reference_compatible():
    class FakePolicy:
        metadata = {"action_chunk_size": 30}

        def predict_action(self, examples):
            return {"actions": np.zeros((len(examples), 30, 7), np.float32)}

        def predict_action_realtime(self, examples, **kwargs):
            return self.predict_action(examples)

        def reset(self, **kwargs):
            return {"reset": True}

    server = WebsocketPolicyServer(FakePolicy(), metadata=FakePolicy.metadata)
    request = {
        "type": "infer",
        "request_id": "abc",
        "payload": {"examples": [{"state": np.zeros(7, np.float32)}]},
    }
    decoded = msgpack_numpy.unpackb(msgpack_numpy.packb(request))
    response = server.route_message(decoded)
    roundtrip = msgpack_numpy.unpackb(msgpack_numpy.packb(response))
    assert roundtrip["ok"] and roundtrip["request_id"] == "abc"
    assert roundtrip["data"]["actions"].shape == (1, 30, 7)
    assert server.route_message({"type": "ping", "request_id": 2})["ok"]
    assert server.route_message({"type": "reset", "payload": {}})["data"]["reset"]
    assert not server.route_message({"type": "invalid"})["ok"]


def test_real_websocket_handshake_and_inference_roundtrip():
    pytest.importorskip("websockets")
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    class FakePolicy:
        metadata = {"action_chunk_size": 30, "camera_order": ["global", "hand"]}

        def predict_action(self, examples):
            return {"actions": np.ones((len(examples), 30, 7), np.float32)}

        def reset(self, **kwargs):
            return {"reset": True}

    async def exercise():
        policy = FakePolicy()
        server = WebsocketPolicyServer(policy, host="127.0.0.1", port=0)
        async with serve(
            server._handler, "127.0.0.1", 0, compression=None, max_size=None
        ) as listener:
            port = listener.sockets[0].getsockname()[1]
            async with connect(
                f"ws://127.0.0.1:{port}", compression=None, max_size=None
            ) as client:
                metadata = msgpack_numpy.unpackb(await client.recv())
                assert metadata["action_chunk_size"] == 30
                await client.send(
                    msgpack_numpy.packb(
                        {
                            "type": "infer",
                            "request_id": "wire",
                            "payload": {"examples": [{"state": np.zeros(7)}]},
                        }
                    )
                )
                response = msgpack_numpy.unpackb(await client.recv())
                assert response["ok"] and response["request_id"] == "wire"
                assert response["data"]["actions"].shape == (1, 30, 7)

    asyncio.run(exercise())


def test_all_36_organized_checkpoints_have_deployable_embedded_contracts():
    sweep = Path(__file__).resolve().parents[1] / "outputs" / "SWEEP"
    checkpoints = sorted(sweep.glob("dit_*/*/checkpoints/*.pt"))
    assert len(checkpoints) == 36
    contracts = []
    for checkpoint in checkpoints:
        metadata = inspect_checkpoint(checkpoint)
        contracts.append(
            (
                metadata.config.policy.model_size,
                metadata.config.data.action_representation,
                metadata.architecture,
                metadata.is_rtc,
            )
        )
        assert metadata.prepared_path.is_dir()
        assert metadata.config.data.action_horizon == 30
        assert metadata.config.data.camera_keys == (
            "observation.images.global",
            "observation.images.hand",
        )
    assert len(set(contracts)) == 36
