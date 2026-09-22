from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_policy.config import Config, config_dict


class WandbTracker:
    """Small strict W&B adapter: enabled online runs fail instead of silently degrading."""

    def __init__(self, cfg: Config, architecture: str, rtc: bool, output_path: str | Path,
                 resume_info: dict[str, Any] | None = None):
        self.run = None
        self.info: dict[str, Any] | None = None
        if not cfg.wandb.enabled or cfg.wandb.mode == "disabled":
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B logging is enabled but wandb is not installed") from exc
        stage = "ttrtc" if rtc else "base"
        suffix = f"-{cfg.wandb.run_suffix}" if cfg.wandb.run_suffix else ""
        run_name = f"{architecture}-{stage}{suffix}"
        output_dir = Path(cfg.wandb.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        project = cfg.wandb.rtc_project if rtc else cfg.wandb.base_project
        if resume_info and resume_info.get("project") != project:
            raise ValueError(
                f"resume W&B project {resume_info.get('project')!r} does not match configured project {project!r}"
            )
        run_id = resume_info.get("run_id") if resume_info else None
        self.run = wandb.init(
            project=project,
            entity=cfg.wandb.entity,
            name=run_name,
            group=f"{cfg.data.action_representation}-{stage}",
            job_type=architecture,
            config=config_dict(cfg),
            mode=cfg.wandb.mode,
            dir=str(output_dir.resolve()),
            tags=[cfg.data.action_representation, architecture, stage, cfg.policy.model_size,
                  "deterministic" if cfg.train.deterministic else "nondeterministic"],
            id=run_id,
            resume="must" if run_id else None,
        )
        if self.run is None:
            raise RuntimeError("wandb.init returned no run while logging is enabled")
        self.run.define_metric("update")
        self.run.define_metric("train/*", step_metric="update")
        self.run.define_metric("validation/*", step_metric="update")
        self.info = {
            "project": project,
            "entity": self.run.entity,
            "run_id": self.run.id,
            "run_name": self.run.name,
            "url": self.run.url,
            "mode": cfg.wandb.mode,
            "output_checkpoint": str(Path(output_path).resolve()),
        }

    def log_train(self, update: int, values: dict[str, float]) -> None:
        if self.run is not None:
            self.run.log({"update": update, **{f"train/{k}": v for k, v in values.items()}}, step=update)

    def log_validation(self, update: int, values: dict[str, float]) -> None:
        if self.run is not None:
            self.run.log({"update": update, **{f"validation/{k}": v for k, v in values.items()}}, step=update)

    def finish(self, checkpoint: Path, summary: dict[str, Any]) -> None:
        if self.run is None:
            return
        for key, value in summary.items():
            self.run.summary[key] = value
        # Checkpoints remain local under the experiment output tree. W&B is
        # metrics/summary-only; never stage or upload model artifacts.
        self.run.finish()
