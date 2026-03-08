#!/usr/bin/env python3
"""
Register trained models into MLflow Model Registry with proper versioning.

Creates two registered models:
  - taluithai-gnn          (GNN graph encoder)
  - taluithai-two-tower    (Two-Tower recommendation)

Each training iteration becomes a model version:
  v1 = baseline (labeled data only)
  v2 = noisy-ER (unfiltered entity resolution)
  v3 = clean-ER (filtered entity resolution)  ← champion

Usage:
    python scripts/_register_models.py
"""

import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlflow
import mlflow.pytorch
import torch
from pathlib import Path
from config import (
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    EMBED_DIR,
    ROOT_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Model Registry Names ─────────────────────────────────────────────────
GNN_MODEL_NAME = "taluithai-gnn"
TWO_TOWER_MODEL_NAME = "taluithai-two-tower"


def get_run_id_by_name(run_name: str) -> str | None:
    """Find run_uuid by run name."""
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f'run_name = "{run_name}"',
    )
    if len(runs) > 0:
        return runs.iloc[0]["run_id"]
    return None


def ensure_registered_model(client, name: str):
    """Create registered model if it doesn't exist."""
    try:
        client.get_registered_model(name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(name)
        logger.info(f"  Created registered model: {name}")


def register_model_from_run(
    run_name: str,
    registered_model_name: str,
    model_path_in_artifacts: str = "model",
    description: str = "",
    alias: str | None = None,
    tags: dict[str, str] | None = None,
    local_model_file: str | None = None,
    is_historical: bool = False,
):
    """
    Register a model version from an existing run.
    
    For real runs with model files: logs via mlflow.pytorch and registers.
    For historical/manual runs: creates version via client API pointing to run.
    """
    run_id = get_run_id_by_name(run_name)
    if not run_id:
        logger.warning(f"Run '{run_name}' not found, skipping")
        return None

    logger.info(f"Registering from run: {run_name} ({run_id[:8]}...)")

    client = mlflow.tracking.MlflowClient()
    ensure_registered_model(client, registered_model_name)

    if is_historical:
        # Historical run — no real model file, just create a version entry
        logger.info(f"  Historical run — creating version entry")
        run_info = client.get_run(run_id)
        source = run_info.info.artifact_uri + f"/{model_path_in_artifacts}"
        
        mv = client.create_model_version(
            name=registered_model_name,
            source=source,
            run_id=run_id,
            description=description,
            tags=tags,
        )
        version = mv.version
    else:
        # Real run — log model files as artifacts, then register
        if registered_model_name == GNN_MODEL_NAME:
            files_to_log = [MODEL_DIR / "gnn_best.pt"]
        else:
            files_to_log = [
                MODEL_DIR / "query_tower_traced.pt",
                MODEL_DIR / "query_tower.pt",
                MODEL_DIR / "item_tower.pt",
                MODEL_DIR / "two_tower_best.pt",
            ]

        missing = [f for f in files_to_log if not f.exists()]
        if missing:
            logger.warning(f"  Missing files: {missing}")
            return None

        logger.info(f"  Logging {len(files_to_log)} model file(s)...")

        with mlflow.start_run(run_id=run_id):
            for f in files_to_log:
                mlflow.log_artifact(str(f), artifact_path=model_path_in_artifacts)

        # Create model version via client API (bypasses logged_model requirement)
        run_info = client.get_run(run_id)
        source = run_info.info.artifact_uri + f"/{model_path_in_artifacts}"
        
        mv = client.create_model_version(
            name=registered_model_name,
            source=source,
            run_id=run_id,
            description=description,
            tags=tags,
        )
        version = mv.version

    logger.info(f"  → Registered {registered_model_name} v{version}")

    # Set alias (e.g., "champion", "production")
    if alias:
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=alias,
            version=str(version),
        )
        logger.info(f"  → Alias @{alias} → v{version}")

    return version


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # ══════════════════════════════════════════════════════════════════════
    # 1. Register GNN models
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Registering GNN models...")
    logger.info("=" * 60)

    # v1: noisy-ER GNN (historical — no model file)
    register_model_from_run(
        run_name="gnn-L3-h256-e50-noisy-er",
        registered_model_name=GNN_MODEL_NAME,
        description="GNN v1: Noisy ER data (7,237 unfiltered matches). Accuracy=92.5%",
        tags={"data_version": "v2-noisy-er", "accuracy": "0.925"},
        is_historical=True,
    )

    # v2: clean-ER GNN (current best) — from vast.ai training
    register_model_from_run(
        run_name="gnn-L3-h256-e50",
        registered_model_name=GNN_MODEL_NAME,
        description="GNN v2: Clean ER data (5,839 filtered matches). Accuracy=91.3%. Currently deployed.",
        alias="champion",
        tags={"data_version": "v3-clean-er", "accuracy": "0.9126"},
    )

    # ══════════════════════════════════════════════════════════════════════
    # 2. Register Two-Tower models
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Registering Two-Tower models...")
    logger.info("=" * 60)

    # v1: baseline Two-Tower (historical)
    register_model_from_run(
        run_name="two-tower-ips-e15-baseline",
        registered_model_name=TWO_TOWER_MODEL_NAME,
        description="Two-Tower v1: Baseline (labeled data only, 201,335 pairs). Recall@50=0.5969, NDCG@10=0.2529",
        tags={"data_version": "v1-baseline", "recall_50": "0.5969", "ndcg_10": "0.2529"},
        is_historical=True,
    )

    # v2: noisy-ER Two-Tower (historical)
    register_model_from_run(
        run_name="two-tower-ips-e15-noisy-er",
        registered_model_name=TWO_TOWER_MODEL_NAME,
        description="Two-Tower v2: Noisy ER (7,237 unfiltered matches, 210,572 pairs). Recall@50=0.5907, NDCG@10=0.2458 (regression)",
        tags={"data_version": "v2-noisy-er", "recall_50": "0.5907", "ndcg_10": "0.2458"},
        is_historical=True,
    )

    # v3: clean-ER Two-Tower (current best) — from vast.ai training
    register_model_from_run(
        run_name="two-tower-ips-e15-lr0.0001",
        registered_model_name=TWO_TOWER_MODEL_NAME,
        description="Two-Tower v3: Clean ER (5,839 filtered matches, 207,174 pairs). Recall@50=0.5947, NDCG@10=0.2557. Currently deployed to Qdrant.",
        alias="champion",
        tags={"data_version": "v3-clean-er", "recall_50": "0.5947", "ndcg_10": "0.2557"},
    )

    # ══════════════════════════════════════════════════════════════════════
    # 3. Summary
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL REGISTRY SUMMARY")
    print("=" * 70)

    for model_name in [GNN_MODEL_NAME, TWO_TOWER_MODEL_NAME]:
        rm = client.get_registered_model(model_name)
        print(f"\n📦 {model_name}")
        if rm.description:
            print(f"   {rm.description}")

        # List versions
        from mlflow.entities.model_registry import ModelVersionTag
        versions = client.search_model_versions(f"name='{model_name}'")
        for mv in sorted(versions, key=lambda x: int(x.version)):
            aliases_str = ""
            if mv.aliases:
                aliases_str = " ← " + ", ".join(f"@{a}" for a in mv.aliases)
            print(f"   v{mv.version}: {mv.description or 'no description'}{aliases_str}")
            if mv.tags:
                tag_str = ", ".join(f"{k}={v}" for k, v in mv.tags.items())
                print(f"         tags: {tag_str}")

    print("\n" + "=" * 70)
    print("Done! Use `mlflow.pyfunc.load_model('models:/taluithai-two-tower@champion')` to load the best model.")
    print("=" * 70)


if __name__ == "__main__":
    main()
