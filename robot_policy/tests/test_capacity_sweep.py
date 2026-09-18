from hashlib import sha256
import json
from pathlib import Path

import pytest

from robot_policy.config import (
    ACTIVE_ARCHITECTURES,
    ACTIVE_MODEL_SIZES,
    LEGACY_ARCHITECTURES,
    LEGACY_MODEL_SIZES,
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_MODEL_SIZES,
    Config,
    load_config,
    require_active_architecture,
    require_active_model_size,
)
from robot_policy.data.parent_predictions import hash_keyed_output
from robot_policy.training import (
    _atomic_torch_save,
    _matching_parent_cache,
    _skip_optimizer_step,
    validate,
)
from scripts.write_model_size_sweep_reports import capacity_rows, result_rows

import torch


def test_active_architecture_scope_excludes_legacy_layerwise():
    assert ACTIVE_ARCHITECTURES == ("fm", "discrete_joint")
    assert LEGACY_ARCHITECTURES == ("discrete_layerwise",)
    assert SUPPORTED_ARCHITECTURES == ACTIVE_ARCHITECTURES + LEGACY_ARCHITECTURES

    # Historical checkpoints remain constructible and loadable.
    cfg = Config()
    cfg.policy.architecture = "discrete_layerwise"
    cfg.validate()
    with pytest.raises(ValueError, match="legacy/load-only"):
        require_active_architecture(cfg.policy.architecture, "training")


def test_active_model_size_scope_excludes_legacy_dit_l():
    assert ACTIVE_MODEL_SIZES == ("custom", "DiT-S", "DiT-B")
    assert LEGACY_MODEL_SIZES == ("DiT-L",)
    assert SUPPORTED_MODEL_SIZES == ACTIVE_MODEL_SIZES + LEGACY_MODEL_SIZES

    # Historical DiT-L configs remain valid and loadable.
    cfg = Config()
    cfg.policy.model_size = "DiT-L"
    cfg.policy.hidden_dim = 1024
    cfg.policy.depth = 24
    cfg.policy.heads = 16
    cfg.validate()
    with pytest.raises(ValueError, match="legacy/load-only"):
        require_active_model_size(cfg.policy.model_size, "training")


def test_validation_is_from_scratch_generation_and_never_passes_targets():
    class Model(torch.nn.Module):
        def sample(self, batch):
            assert set(batch) == {"vision_features", "state"}
            return torch.zeros(len(batch["state"]), 30, 7)

        def decoded_action_mse(self, controls, batch):
            valid = batch["action_valid_mask"].bool().unsqueeze(-1).expand_as(controls)
            return ((controls - batch["normalized_target_trajectory"]) ** 2)[valid].mean()

    class Codec:
        def decode_tokens(self, tokens):
            raise AssertionError("FM validation must not decode tokens")

    cfg = Config()
    cfg.data.action_representation = "raw"
    cfg.policy.architecture = "fm"
    batch = {
        "vision_features": torch.zeros(2, 2, 16, 2176),
        "state": torch.zeros(2, 7),
        "continuous_target": torch.ones(2, 30, 7),
        "discrete_target": torch.ones(2, 30, 7, dtype=torch.long),
        "control_valid_mask": torch.ones(2, 30, 7, dtype=torch.bool),
        "action_valid_mask": torch.ones(2, 30, dtype=torch.bool),
        "normalized_target_trajectory": torch.ones(2, 30, 7),
    }
    metrics = validate(Model(), [batch], torch.device("cpu"), cfg, rtc_parent=object(), codec=Codec())
    assert metrics["loss"] == metrics["action_mse"] == metrics["generation_action_mse"] == 1.0
    assert metrics["generation_control_mse"] == 1.0


def test_model_size_configs_inherit_longrun_contract():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "dit_s": ("DiT-S", 384, 6, 4),
        "dit_b": ("DiT-B", 768, 12, 12),
        "dit_l": ("DiT-L", 1024, 24, 16),
    }
    for slug, shape in expected.items():
        for representation in ("raw", "bspline"):
            cfg = load_config(root / "configs" / "model_size_sweep" / f"{slug}_{representation}.yaml")
            assert (cfg.policy.model_size, cfg.policy.hidden_dim, cfg.policy.depth, cfg.policy.heads) == shape
            assert cfg.data.action_representation == representation
            assert cfg.train.updates == 50_000
            assert cfg.train.rtc_updates == 5_000
            assert cfg.train.batch_size == cfg.train.effective_batch_size == 32
            assert cfg.wandb.base_project == "robot-policy-50k-bs32-basic"
            assert cfg.wandb.rtc_project == "robot-policy-5k-bs32-ttRTC"


def test_full_vision_configs_keep_capacity_and_batch_contract():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "dit_s": ("DiT-S", 384, 6, 4),
        "dit_b": ("DiT-B", 768, 12, 12),
        "dit_l": ("DiT-L", 1024, 24, 16),
    }
    for slug, shape in expected.items():
        for representation in ("raw", "bspline"):
            cfg = load_config(root / "configs" / "full_vision_512" / f"{slug}_{representation}.yaml")
            assert (cfg.policy.model_size, cfg.policy.hidden_dim, cfg.policy.depth, cfg.policy.heads) == shape
            assert cfg.data.action_representation == representation
            assert cfg.vision.pooled_grid == 16
            assert len(cfg.data.camera_keys) * cfg.vision.pooled_grid**2 == 512
            assert cfg.train.updates == 50_000 and cfg.train.rtc_updates == 5_000
            assert cfg.train.batch_size == cfg.train.effective_batch_size == 32
            assert cfg.train.validation_max_batches == 4
            assert cfg.train.rtc_learning_rate == 3e-5
            assert cfg.train.fm_grad_skip_threshold == 3.0
            assert cfg.train.fm_grad_skip_after_updates == 1000
            assert cfg.wandb.base_project == "robot-policy-full-vision-512-basic"
            assert cfg.wandb.rtc_project == "robot-policy-full-vision-512-ttRTC"


def test_named_model_size_rejects_shape_drift():
    cfg = Config()
    cfg.policy.model_size = "DiT-S"
    with pytest.raises(ValueError, match="requires hidden_dim/depth/heads"):
        cfg.validate()


def test_parent_prediction_cache_requires_exact_checkpoint_hash(tmp_path):
    cfg = Config()
    cfg.data.action_representation = "raw"
    cfg.data.prepared_path = str(tmp_path / "prepared")
    cfg.policy.architecture = "discrete_joint"
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"capacity-specific-parent")
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    legacy = Path(cfg.data.prepared_path) / "parent_predictions" / cfg.policy.architecture
    legacy.mkdir(parents=True)
    (legacy / "manifest.json").write_text(json.dumps({
        "architecture": cfg.policy.architecture,
        "action_representation": "raw",
        "parent_checkpoint_sha256": "wrong-size-parent",
    }))
    assert _matching_parent_cache(cfg, str(checkpoint)) is None

    keyed = legacy / digest
    keyed.mkdir()
    (keyed / "manifest.json").write_text(json.dumps({
        "architecture": cfg.policy.architecture,
        "action_representation": "raw",
        "parent_checkpoint_sha256": digest,
    }))
    assert _matching_parent_cache(cfg, str(checkpoint)) == keyed
    assert hash_keyed_output(cfg, str(checkpoint)) == keyed


def test_atomic_checkpoint_save_publishes_complete_payload(tmp_path):
    checkpoint = tmp_path / "model.pt.resume"
    _atomic_torch_save({"update": 1000, "tensor": torch.arange(8)}, checkpoint)
    first = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert first["update"] == 1000

    _atomic_torch_save({"update": 2000, "tensor": torch.arange(16)}, checkpoint)
    second = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert second["update"] == 2000
    assert second["tensor"].tolist() == list(range(16))
    assert not list(tmp_path.glob(".*.tmp"))


def test_optimizer_guard_rejects_fm_spikes_and_all_nonfinite_gradients():
    assert not _skip_optimizer_step("fm", 22.0, 3.0, 1000, 1000)
    assert not _skip_optimizer_step("fm", 2.99, 3.0, 1001, 1000)
    assert _skip_optimizer_step("fm", 3.01, 3.0, 1001, 1000)
    assert _skip_optimizer_step("fm", 3.01, 3.0, 1, 0)
    assert not _skip_optimizer_step("fm", 2.99, 3.0, 1, 0)
    assert not _skip_optimizer_step("fm", 10_000.0, float("inf"), 1001, 1000)
    assert not _skip_optimizer_step("discrete_joint", 3_538.0, 3.0, 10_000, 1000)
    assert _skip_optimizer_step("fm", float("inf"), 3.0, 1, 1000)
    assert _skip_optimizer_step("discrete_joint", float("nan"), 3.0, 1, 1000)


def test_bilingual_capacity_report_rows_include_audit_evidence(tmp_path):
    checkpoint = tmp_path / "fm_base.pt"
    record = {
        "model_size": "DiT-S",
        "representation": "raw",
        "architecture": "fm",
        "stage": "base",
        "parameters": 123,
        "final_train_action_mse": 0.1,
        "final_validation_action_mse": 0.2,
        "path": str(checkpoint),
        "sha256": "a" * 64,
        "wandb": {"run_id": "run123", "url": "https://wandb.example/run123"},
    }
    assert "FM" in capacity_rows([record])[0]
    assert "流匹配" in capacity_rows([record], chinese=True)[0]
    english = result_rows([record])[0]
    chinese = result_rows([record], chinese=True)[0]
    assert "0.2" in english and "run123" in english and "a" * 64 in english
    assert "流匹配" in chinese and "基础训练" in chinese
