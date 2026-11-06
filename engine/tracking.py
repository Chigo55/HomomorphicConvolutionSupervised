import json
import os
import subprocess
import sys
import tempfile
import traceback
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.artifacts
import torch
from omegaconf import DictConfig, OmegaConf

REPO = Path(__file__).resolve().parents[1]


@contextmanager
def start_run(
    cfg: DictConfig,
    *,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Start (or resume, with `run_id`) an MLflow Run. A child Run (bench/infer)
    lives in its parent's Experiment; otherwise the Experiment is
    `cfg.experiment_name`, created with its artifacts under `tracking.artifact_dir`.
    What the block prints is saved as `logs/<UTC time>.log`."""
    mlflow.set_tracking_uri(cfg.tracking.uri)
    experiment_id = None
    if parent_run_id:
        experiment_id = mlflow.get_run(parent_run_id).info.experiment_id
    elif run_id is None:
        experiment_id = _experiment_id(cfg.experiment_name, cfg.tracking.artifact_dir)
    with (
        mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment_id,
            parent_run_id=parent_run_id,
            run_name=cfg.get("run_name"),
            tags=tags,
            log_system_metrics=cfg.tracking.system_metrics,
        ) as run,
        tempfile.TemporaryDirectory() as tmp,
    ):
        log_path = Path(tmp) / f"{_timestamp()}.log"
        try:
            with open(log_path, "w", encoding="utf-8") as log, _tee(log):
                try:
                    yield run
                except BaseException:
                    log.write(traceback.format_exc())
                    raise
        finally:
            mlflow.log_artifact(str(log_path), "logs")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


class _Tee:
    def __init__(self, stream, file) -> None:
        self.stream, self.file = stream, file

    def write(self, text: str) -> int:
        self.file.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.file.flush()
        self.stream.flush()

    def __getattr__(self, name: str):
        return getattr(self.stream, name)


@contextmanager
def _tee(file) -> Iterator[None]:
    """Copy stdout/stderr into `file`."""
    # ponytail: logging handlers bound to the old streams (pyiqa, hydra) bypass
    # this; print() and warnings are captured
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(out, file), _Tee(err, file)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = out, err


def _experiment_id(name: str, artifact_dir: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    location = (Path(artifact_dir) / name).resolve().as_uri()
    return mlflow.create_experiment(name, artifact_location=location)


def flatten(cfg: DictConfig) -> dict:
    """`{"train.lr": 1e-4, ...}` with interpolations resolved."""

    def walk(d: dict, prefix: str) -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(walk(v, f"{prefix}{k}."))
            else:
                out[f"{prefix}{k}"] = v
        return out

    return walk(OmegaConf.to_container(cfg, resolve=True), "")


def log_config(cfg: DictConfig) -> None:
    """Every config value as a param, plus the resolved config file."""
    params = {k: v for k, v in flatten(cfg).items() if k not in ("resume", "ckpt")}
    mlflow.log_params(params)
    mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")


def log_code_state() -> None:
    """The commit and any uncommitted diff, as `code/<UTC time>.patch`. Called
    for every fit segment, so a resumed Run records the code it resumed with."""

    # ponytail: `git diff HEAD` misses untracked files; add them if that bites
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout

    try:
        commit, diff = git("rev-parse", "HEAD").strip(), git("diff", "HEAD")
    except FileNotFoundError:
        return
    mlflow.log_text(f"# commit {commit}\n{diff}", f"code/{_timestamp()}.patch")


def fetch_checkpoint(uri: str, tracking_uri: str) -> dict:
    """Load `runs:/<run_id>/checkpoints/<name>.pt` on the CPU."""
    path = mlflow.artifacts.download_artifacts(
        artifact_uri=uri, tracking_uri=tracking_uri
    )
    return torch.load(path, map_location="cpu")


def notify(message: str) -> None:
    """Post to the Discord webhook in $DISCORD_WEBHOOK_URL, if set. Never raises."""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps({"content": message[:2000]}).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "hcs-notify/1.0",
            },
        )
        urllib.request.urlopen(request, timeout=10)
    except Exception as e:  # a broken notification must never mask the real error
        print(f"[WARN] Notification failed: {e}")
