from hashlib import sha256
import json
from pathlib import Path

import pytest

from robot_policy.config import Config, load_config
from robot_policy.data.parent_predictions import hash_keyed_output
from robot_policy.training import _atomic_torch_save, _matching_parent_cache
from scripts.write_model_size_sweep_reports import capacity_rows, result_rows

import torch


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
