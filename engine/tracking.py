import json
import os
import subprocess
import urllib.request
from pathlib import Path

import mlflow
from mlflow.entities import Run
from omegaconf import DictConfig, OmegaConf


def start_run(
    cfg: DictConfig,
    *,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> Run:
    """Start (or resume, with `run_id`) an MLflow Run. A child Run (bench/infer)
    lives in its parent's Experiment; otherwise the Experiment is
    `cfg.experiment_name`, created with its artifacts under `tracking.artifact_dir`."""
    mlflow.set_tracking_uri(cfg.tracking.uri)
    experiment_id = None
    if parent_run_id:
        experiment_id = mlflow.get_run(parent_run_id).info.experiment_id
    elif run_id is None:
        experiment_id = _experiment_id(cfg.experiment_name, cfg.tracking.artifact_dir)
    return mlflow.start_run(
        run_id=run_id,
        experiment_id=experiment_id,
        parent_run_id=parent_run_id,
        run_name=cfg.get("run_name"),
        tags=tags,
        log_system_metrics=cfg.tracking.system_metrics,
    )


def _experiment_id(name: str, artifact_dir: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    location = (Path(artifact_dir) / name).resolve().as_uri()
    return mlflow.create_experiment(name, artifact_location=location)


def log_config(cfg: DictConfig) -> None:
    """Every config value as a param, plus the resolved config file."""
    flat = _flatten(OmegaConf.to_container(cfg, resolve=True))
    mlflow.log_params({k: v for k, v in flat.items() if k not in ("resume", "ckpt")})
    mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def log_code_state() -> None:
    """MLflow tags the git commit itself; uncommitted changes go in a patch."""
    # ponytail: `git diff HEAD` misses untracked files; add them if that bites
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
    except FileNotFoundError:
        return
    if diff:
        mlflow.log_text(diff, "uncommitted.patch")


def resolve_checkpoint(uri: str, tracking_uri: str) -> Path:
    """`runs:/<run_id>/checkpoints/best.pt` or a plain local path."""
    if uri.startswith("runs:/"):
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=uri, tracking_uri=tracking_uri
        )
        return Path(local)
    return Path(uri)


def notify(message: str) -> None:
    """Post to the Discord webhook in $DISCORD_WEBHOOK_URL, if set. Never raises."""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    request = urllib.request.Request(
        url,
        data=json.dumps({"content": message[:2000]}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "hcs-notify/1.0"},
    )
    try:
        urllib.request.urlopen(request, timeout=10)
    except OSError as e:
        print(f"[WARN] Notification failed: {e}")
