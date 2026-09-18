import torch
import numpy as np

from robot_policy.config import Config
from robot_policy.evaluation.inference_rtc import dataset_action_minmax, ground_truth_condition, select_full_horizon_indices
from robot_policy.policies import create_policy


def _batch(representation: str, batch_size: int = 1):
    rows = 30 if representation == "raw" else 18
    return {
        "vision_features": torch.randn(batch_size, 2, 16, 2176),
        "state": torch.randn(batch_size, 7),
        "continuous_target": torch.randn(batch_size, rows, 7),
        "discrete_target": torch.randint(0, 256, (batch_size, rows, 7)),
        "control_valid_mask": torch.ones(batch_size, rows, 7, dtype=torch.bool),
    }


def _config(representation: str, architecture: str) -> Config:
    cfg = Config()
    cfg.data.action_representation = representation
    cfg.policy.architecture = architecture
    cfg.policy.hidden_dim = 24
    cfg.policy.depth = 1
    cfg.policy.heads = 4
    cfg.policy.discrete_rounds = 2
    cfg.policy.fm_steps = 2
    return cfg


def test_ground_truth_condition_uses_raw_prefix_rows():
    cfg = _config("raw", "fm")
    batch = _batch("raw", 2)
    values, fixed, mapping = ground_truth_condition(batch, cfg, "fm", 6)
    assert values.data_ptr() == batch["continuous_target"].data_ptr()
    assert fixed.shape == (2, 30, 7)
    assert fixed[:, :6].all() and not fixed[:, 6:].any()
    assert mapping["fixed_control_rows"] == 6


def test_ground_truth_condition_uses_complete_bspline_support():
    cfg = _config("bspline", "discrete_joint")
    batch = _batch("bspline", 2)
    values, fixed, mapping = ground_truth_condition(batch, cfg, "discrete_joint", 6)
    assert values.data_ptr() == batch["discrete_target"].data_ptr()
    assert fixed.shape == (2, 18, 7)
    # Six raw steps cover three spans; cubic support requires D + degree = 6 controls.
    assert fixed[:, :6].all() and not fixed[:, 6:].any()
    assert mapping["affected_spans"] == 3
    assert mapping["fixed_control_rows"] == 6


def test_every_policy_preserves_oracle_prefix_for_both_representations():
    for representation in ("raw", "bspline"):
        for architecture in ("fm", "discrete_layerwise", "discrete_joint"):
            cfg = _config(representation, architecture)
            batch = _batch(representation)
            prefix, fixed, _ = ground_truth_condition(batch, cfg, architecture, 4)
            model = create_policy(cfg).eval()
            prediction = model.sample(
                batch, steps=2, rounds=2, prefix_values=prefix, fixed_mask=fixed, use_cache=True
            )
            if architecture == "fm":
                torch.testing.assert_close(prediction[fixed], prefix[fixed])
            else:
                assert torch.equal(prediction[fixed], prefix[fixed])


def test_episode_balanced_complete_chunk_selection():
    class Dataset:
        index = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0)]

        def __getitem__(self, index):
            # Exclude one item to prove incomplete chunks are never selected.
            valid = torch.ones(30, dtype=torch.bool)
            if index == 1:
                valid[-1] = False
            return {"action_valid_mask": valid}

    chosen = select_full_horizon_indices(Dataset(), 5, 7)
    assert 1 not in chosen
    assert len(chosen) == 5
    assert {Dataset.index[index][0] for index in chosen[:3]} == {1, 2, 3}


def test_dataset_action_minmax_uses_all_prepared_episodes(tmp_path):
    actions = tmp_path / "actions"
    actions.mkdir()
    first = np.stack([np.arange(7), np.arange(7) + 10]).astype(np.float32)
    second = np.stack([np.arange(7) - 5, np.arange(7) + 20]).astype(np.float32)
    np.savez(actions / "episode_000001.npz", action=first)
    np.savez(actions / "episode_000002.npz", action=second)
    minimum, maximum = dataset_action_minmax(tmp_path)
    np.testing.assert_array_equal(minimum, np.arange(7) - 5)
    np.testing.assert_array_equal(maximum, np.arange(7) + 20)
