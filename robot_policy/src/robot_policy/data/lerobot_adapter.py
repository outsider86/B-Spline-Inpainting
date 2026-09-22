from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from robot_policy.config import Config, config_dict
from robot_policy.encoders.bspline_adapter import BSplineAdapter


def _paths(cfg: Config) -> tuple[Path, Path]:
    return Path(cfg.data.dataset_path).resolve(), Path(cfg.data.prepared_path).resolve()


def load_episode(path: Path, cfg: Config) -> dict[str, np.ndarray]:
    columns = [cfg.data.state_key, cfg.data.action_key, "timestamp", "frame_index", "episode_index"]
    data = pq.read_table(path, columns=columns).to_pydict()
    return {
        "state": np.asarray(data[cfg.data.state_key], dtype=np.float32),
        "action": np.asarray(data[cfg.data.action_key], dtype=np.float32),
        "timestamp": np.asarray(data["timestamp"], dtype=np.float64),
        "frame_index": np.asarray(data["frame_index"], dtype=np.int64),
        "episode_index": np.asarray(data["episode_index"], dtype=np.int64),
    }


def make_splits(cfg: Config, episode_ids: list[int]) -> dict[str, list[int]]:
    expected = cfg.data.train_episodes + cfg.data.val_episodes + cfg.data.test_episodes
    if expected != len(episode_ids):
        raise ValueError(f"split sizes total {expected}, but dataset has {len(episode_ids)} episodes")
    ids = np.asarray(sorted(episode_ids), dtype=np.int64)
    np.random.default_rng(cfg.data.split_seed).shuffle(ids)
    a = cfg.data.train_episodes
    b = a + cfg.data.val_episodes
    return {"train": sorted(ids[:a].tolist()), "val": sorted(ids[a:b].tolist()), "test": sorted(ids[b:].tolist())}


def audit_dataset(cfg: Config) -> dict[str, Any]:
    root, _ = _paths(cfg)
    info = json.loads((root / "meta/info.json").read_text())
    modality = json.loads((root / "meta/modality.json").read_text())
    missing = [key for key in (*cfg.data.camera_keys, cfg.data.state_key, cfg.data.action_key) if key not in info["features"]]
    if missing:
        raise ValueError(f"dataset is missing configured fields: {missing}")
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet episodes found under {root}")
    episode_ids, lengths, all_dt, bad_frames = [], [], [], []
    for path in files:
        ep = load_episode(path, cfg)
        eid = int(ep["episode_index"][0])
        if ep["state"].ndim != 2 or ep["state"].shape[1] != cfg.data.state_dim:
            raise ValueError(
                f"episode {eid} state shape {ep['state'].shape} does not match "
                f"configured state_dim={cfg.data.state_dim}"
            )
        if ep["action"].ndim != 2 or ep["action"].shape[1] != 7:
            raise ValueError(
                f"episode {eid} action shape {ep['action'].shape} must be [frames, 7]"
            )
        episode_ids.append(eid)
        lengths.append(len(ep["action"]))
        all_dt.extend(np.diff(ep["timestamp"]).tolist())
        if not np.array_equal(ep["frame_index"], np.arange(len(ep["frame_index"]))):
            bad_frames.append(eid)
        if not np.isfinite(ep["action"]).all() or not np.isfinite(ep["state"]).all():
            raise ValueError(f"non-finite state/action in episode {eid}")
    dt = np.asarray(all_dt)
    expected_dt = 1.0 / cfg.data.frequency_hz
    report = {
        "dataset": str(root), "codebase_version": info["codebase_version"],
        "robot_type": info["robot_type"], "episodes": len(files), "frames": int(sum(lengths)),
        "episode_length_min": min(lengths), "episode_length_max": max(lengths),
        "fps": info["fps"], "dt_min": float(dt.min()), "dt_median": float(np.median(dt)),
        "dt_max": float(dt.max()), "dt_anomalies_over_1ms": int(np.sum(np.abs(dt - expected_dt) > 1e-3)),
        "nonconsecutive_frame_episodes": bad_frames,
        "camera_keys": list(cfg.data.camera_keys), "state_key": cfg.data.state_key,
        "state_dim": cfg.data.state_dim,
        "action_key": cfg.data.action_key, "state_modality": modality["state"],
        "action_modality": modality["action"], "action_alignment": "action[t] paired with observation[t] per LeRobot row; collection semantics not independently encoded in metadata",
    }
    return report


def _windows(actions: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    n, dim = actions.shape
    index = np.minimum(np.arange(n)[:, None] + np.arange(horizon)[None, :], n - 1)
    windows = actions[index]
    valid = np.minimum(horizon, n - np.arange(n))
    return windows.reshape(n, horizon, dim), valid


def prepare_action_targets(cfg: Config) -> dict[str, Any]:
    root, output = _paths(cfg)
    action_dir = output / "actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    report = audit_dataset(cfg)
    files = sorted((root / "data").rglob("*.parquet"))
    episodes = {int(path.stem.split("_")[-1]): load_episode(path, cfg) for path in files}
    splits = make_splits(cfg, sorted(episodes))
    train_state = np.concatenate([episodes[i]["state"] for i in splits["train"]])
    train_action = np.concatenate([episodes[i]["action"] for i in splits["train"]])
    state_mean, state_std = train_state.mean(0), train_state.std(0)
    state_std = np.maximum(state_std, 1e-6)
    action_low, action_high = np.quantile(train_action, [0.01, 0.99], axis=0)
    action_span = np.maximum(action_high - action_low, 1e-8)

    adapter = BSplineAdapter(cfg) if cfg.data.action_representation == "bspline" else None
    train_controls: list[np.ndarray] = []
    normalized_windows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for eid, ep in episodes.items():
        normalized = np.clip(2 * (ep["action"] - action_low) / action_span - 1, -1, 1)
        windows, valid = _windows(normalized.astype(np.float64), cfg.data.action_horizon)
        normalized_windows[eid] = windows, valid
        if adapter is not None and eid in splits["train"]:
            train_controls.extend(adapter.fit(window) for window in windows)
    if adapter is not None:
        adapter.calibrate_controls(train_controls)

    fit_errors, quant_errors = [], []
    manifest_eps = []
    for eid, ep in episodes.items():
        windows, valid_steps = normalized_windows[eid]
        target = windows.astype(np.float32)
        if adapter is not None:
            encoded = [adapter.encode_window(w, int(v)) for w, v in zip(windows, valid_steps)]
            controls = np.stack([x.continuous for x in encoded])
            tokens = np.stack([x.discrete for x in encoded]).astype(np.uint8)
            reconstruction = np.stack([x.reconstruction for x in encoded])
            quant_reconstruction = np.stack([x.quantized_reconstruction for x in encoded])
            raw_valid = np.stack([x.raw_valid_mask for x in encoded])
            control_valid = np.stack([x.control_valid_mask for x in encoded])
        else:
            controls = target
            tokens = np.rint((controls + 1.0) * 0.5 * (cfg.spline.vocab_size - 1)).clip(0, cfg.spline.vocab_size - 1).astype(np.uint8)
            reconstruction = target.copy()
            quant_reconstruction = (tokens.astype(np.float32) / (cfg.spline.vocab_size - 1) * 2.0 - 1.0).astype(np.float32)
            raw_valid = np.arange(cfg.data.action_horizon)[None, :] < valid_steps[:, None]
            control_valid = np.broadcast_to(raw_valid[..., None], controls.shape).copy()
        valid3 = raw_valid[..., None]
        fit_errors.append(np.abs(reconstruction - target)[valid3.repeat(7, axis=2)])
        quant_errors.append(np.abs(quant_reconstruction - reconstruction)[valid3.repeat(7, axis=2)])
        np.savez_compressed(
            action_dir / f"episode_{eid:06d}.npz", state=ep["state"],
            normalized_state=((ep["state"] - state_mean) / state_std).astype(np.float32),
            action=ep["action"], normalized_action=((ep["action"] - action_low) * 2 / action_span - 1).clip(-1, 1).astype(np.float32),
            continuous_target=controls, discrete_target=tokens, action_valid_mask=raw_valid,
            control_valid_mask=control_valid, reconstruction=reconstruction,
            quantized_reconstruction=quant_reconstruction, timestamp=ep["timestamp"], frame_index=ep["frame_index"],
        )
        manifest_eps.append({"episode_id": eid, "frames": len(ep["action"]), "split": next(k for k,v in splits.items() if eid in v)})

    if adapter is not None:
        calibration = adapter.calibration_state()
        encoder_record = {
            "type": "UniformLeftBSplineConfig", "config": vars(adapter.config),
            "tokenizer_id": adapter.tokenizer_id, "calibration": calibration,
            "predicted_fields": "18x7 control points or ordered bins", "fixed_fields": "knots, span length, degree, sample period, phase",
            "rtc_history_steps": [2, 4, 6, 8, 10],
        }
    else:
        encoder_record = {
            "type": "RawActionSequenceConfig",
            "config": {"action_horizon": cfg.data.action_horizon, "action_dim": int(train_action.shape[1]), "vocab_size": cfg.spline.vocab_size},
            "tokenizer_id": "raw_action_q01_q99_uniform_256_v1",
            "calibration": {"low": [-1.0] * int(train_action.shape[1]), "high": [1.0] * int(train_action.shape[1])},
            "predicted_fields": f"{cfg.data.action_horizon}x{train_action.shape[1]} normalized raw actions or ordered bins",
            "fixed_fields": "action horizon, sample period, normalization quantiles",
            "rtc_history_steps": list(range(cfg.rtc.raw_delay_min, cfg.rtc.raw_delay_max + 1)),
        }
    encoder_bytes = json.dumps(encoder_record, sort_keys=True).encode()
    stats = {
        "state_mean": state_mean.tolist(), "state_std": state_std.tolist(),
        "action_q01": action_low.tolist(), "action_q99": action_high.tolist(),
        "encoder": encoder_record, "encoder_sha256": sha256(encoder_bytes).hexdigest(),
    }
    (output / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    (output / "normalization.json").write_text(json.dumps(stats, indent=2) + "\n")
    (output / "encoder.json").write_bytes(encoder_bytes + b"\n")
    fit = np.concatenate(fit_errors); quant = np.concatenate(quant_errors)
    metrics = {
        "fit_mae": float(fit.mean()), "fit_rmse": float(np.sqrt(np.mean(fit**2))),
        "fit_max": float(fit.max()), "fit_quantiles": {str(q): float(np.quantile(fit,q)) for q in (0.5,0.9,0.95,0.99)},
        "additional_quantization_mae": float(quant.mean()), "additional_quantization_rmse": float(np.sqrt(np.mean(quant**2))),
        "additional_quantization_max": float(quant.max()), "additional_quantization_quantiles": {str(q): float(np.quantile(quant,q)) for q in (0.5,0.9,0.95,0.99)},
    }
    manifest = {"config": config_dict(cfg), "dataset_audit": report, "splits": splits, "episodes": manifest_eps, "metrics": metrics}
    (output / "action_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
