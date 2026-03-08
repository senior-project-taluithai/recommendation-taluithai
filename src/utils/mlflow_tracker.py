"""
MLflow tracking utilities for the recommendation pipeline.

Provides a thin wrapper around MLflow so training scripts can log
hyperparameters, metrics per epoch, and model artifacts with minimal boilerplate.
Falls back gracefully if MLflow is not installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow not installed — tracking disabled. Install with: pip install mlflow")


class MLflowTracker:
    """
    Lightweight MLflow wrapper.

    Usage:
        tracker = MLflowTracker("taluithai-gnn", params={...})
        for epoch in range(n):
            tracker.log_metrics({"loss": loss, "acc": acc}, step=epoch)
        tracker.log_model(model, "gnn_best")
        tracker.end()
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ):
        self.enabled = MLFLOW_AVAILABLE
        self._run = None

        if not self.enabled:
            return

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        self._run = mlflow.start_run(run_name=run_name)
        logger.info(
            f"MLflow run started: {self._run.info.run_id} "
            f"(experiment={experiment_name})"
        )

        if params:
            # MLflow params must be strings — flatten dicts/lists
            flat = {}
            for k, v in params.items():
                if isinstance(v, (dict, list)):
                    flat[k] = json.dumps(v)
                else:
                    flat[k] = str(v)
            mlflow.log_params(flat)

        if tags:
            mlflow.set_tags(tags)

    # ── Helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _sanitize_key(key: str) -> str:
        """Replace characters not allowed by MLflow (e.g. @) in metric/param names."""
        return key.replace("@", "_at_")

    # ── Metrics ───────────────────────────────────────────────────────────────
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled:
            return
        sanitized = {self._sanitize_key(k): v for k, v in metrics.items()}
        mlflow.log_metrics(sanitized, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if not self.enabled:
            return
        mlflow.log_metric(self._sanitize_key(key), value, step=step)

    # ── Artifacts ─────────────────────────────────────────────────────────────
    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log a file or directory as an artifact."""
        if not self.enabled:
            return
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_artifacts(self, local_dir: str | Path, artifact_path: str | None = None) -> None:
        """Log all files in a directory as artifacts."""
        if not self.enabled:
            return
        mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)

    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log a PyTorch model using MLflow's PyTorch flavor."""
        if not self.enabled:
            return
        try:
            mlflow.pytorch.log_model(model, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log model via mlflow.pytorch: {e}")
            # Fallback: save state_dict as artifact
            import tempfile, torch
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                torch.save(model.state_dict(), f.name)
                mlflow.log_artifact(f.name, artifact_path)

    def log_dict(self, data: dict, filename: str) -> None:
        """Log a dictionary as a JSON artifact."""
        if not self.enabled:
            return
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            mlflow.log_artifact(f.name)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def end(self) -> None:
        if not self.enabled:
            return
        if self._run:
            mlflow.end_run()
            logger.info(f"MLflow run ended: {self._run.info.run_id}")
            self._run = None

    @property
    def run_id(self) -> str | None:
        if self._run:
            return self._run.info.run_id
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end()
