"""
Merge old MLflow runs into current mlflow.db and add legacy noisy-ER runs.
Keeps all existing runs, adds missing ones from backup + manual entries.
"""
import sqlite3
import uuid
import time

BASE = "/Users/tar/Documents/year4/recommendation-taluithai"
CURRENT_DB = f"{BASE}/mlflow.db"
BACKUP_DB = f"{BASE}/mlflow.db.bak"


def get_all_run_ids(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT run_uuid FROM runs")
    ids = {r[0] for r in cur.fetchall()}
    conn.close()
    return ids


def copy_run(src_db, dst_db, run_id):
    """Copy a run and all related data from src to dst."""
    src = sqlite3.connect(src_db)
    dst = sqlite3.connect(dst_db)
    
    # Tables that reference run_uuid
    tables_with_run = [
        ("runs", "run_uuid"),
        ("params", "run_uuid"),
        ("metrics", "run_uuid"),
        ("latest_metrics", "run_uuid"),
        ("tags", "run_uuid"),
    ]
    
    for table, col in tables_with_run:
        try:
            rows = src.execute(f"SELECT * FROM {table} WHERE {col}=?", (run_id,)).fetchall()
            if rows:
                cols = [d[0] for d in src.execute(f"SELECT * FROM {table} LIMIT 1").description]
                placeholders = ",".join(["?"] * len(cols))
                for row in rows:
                    try:
                        dst.execute(f"INSERT INTO {table} VALUES ({placeholders})", row)
                    except sqlite3.IntegrityError:
                        pass  # Already exists
        except Exception as e:
            print(f"  Warning: {table}: {e}")
    
    dst.commit()
    src.close()
    dst.close()
    print(f"  Copied run {run_id[:8]}...")


def add_manual_run(db_path, experiment_name, run_name, params, metrics, tags=None, timestamp_ms=None):
    """Add a manual run entry to MLflow DB."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Check if run with same name already exists
    cur.execute("SELECT run_uuid FROM runs WHERE name=?", (run_name,))
    if cur.fetchone():
        print(f"  Run '{run_name}' already exists, skipping")
        conn.close()
        return
    
    # Get experiment_id
    cur.execute("SELECT experiment_id FROM experiments WHERE name=?", (experiment_name,))
    row = cur.fetchone()
    if not row:
        print(f"  Experiment '{experiment_name}' not found, skipping")
        conn.close()
        return
    exp_id = row[0]
    
    run_id = uuid.uuid4().hex
    now_ms = timestamp_ms or int(time.time() * 1000)
    
    # Insert run (schema: run_uuid, name, source_type, source_name, entry_point_name,
    #             user_id, status, start_time, end_time, source_version, 
    #             lifecycle_stage, artifact_uri, experiment_id, deleted_time)
    cur.execute("""
        INSERT INTO runs (run_uuid, name, source_type, source_name, entry_point_name,
                         user_id, status, start_time, end_time, source_version,
                         lifecycle_stage, artifact_uri, experiment_id, deleted_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, run_name, "LOCAL", "", "", "manual", "FINISHED",
          now_ms - 100000, now_ms, "", "active",
          f"mlruns/{exp_id}/{run_id}/artifacts", exp_id, None))
    
    # Insert params
    for k, v in params.items():
        cur.execute("INSERT INTO params (key, value, run_uuid) VALUES (?,?,?)", (k, str(v), run_id))
    
    # Insert metrics + latest_metrics
    for k, v in metrics.items():
        cur.execute("INSERT INTO metrics (key, value, timestamp, run_uuid, step, is_nan) VALUES (?,?,?,?,?,?)",
                    (k, v, now_ms, run_id, 0, 0))
        cur.execute("INSERT INTO latest_metrics (key, value, timestamp, run_uuid, step, is_nan) VALUES (?,?,?,?,?,?)",
                    (k, v, now_ms, run_id, 0, 0))
    
    # Insert tags
    all_tags = {"mlflow.runName": run_name}
    if tags:
        all_tags.update(tags)
    for k, v in all_tags.items():
        cur.execute("INSERT INTO tags (key, value, run_uuid) VALUES (?,?,?)", (k, str(v), run_id))
    
    conn.commit()
    conn.close()
    print(f"  Added manual run: {run_name} ({run_id[:8]})")
    return run_id


def main():
    # 1. Check current runs
    current_ids = get_all_run_ids(CURRENT_DB)
    backup_ids = get_all_run_ids(BACKUP_DB)
    print(f"Current DB: {len(current_ids)} runs")
    print(f"Backup DB:  {len(backup_ids)} runs")
    
    # 2. Copy missing runs from backup
    missing = backup_ids - current_ids
    if missing:
        print(f"\nCopying {len(missing)} runs from backup...")
        for run_id in missing:
            copy_run(BACKUP_DB, CURRENT_DB, run_id)
    else:
        print("\nNo missing runs from backup")
    
    # 3. Add tag to new runs to mark them as "clean-er"
    conn = sqlite3.connect(CURRENT_DB)
    for run_id in ['8420519b2624482b8a8bbaeec877f1fe', 'c4039459e04e426fb91db4463f26f09d']:
        try:
            conn.execute("INSERT INTO tags (key, value, run_uuid) VALUES (?,?,?)",
                        ("data_version", "v3-clean-er", run_id))
        except sqlite3.IntegrityError:
            conn.execute("UPDATE tags SET value=? WHERE key=? AND run_uuid=?",
                        ("v3-clean-er", "data_version", run_id))
    conn.commit()
    conn.close()
    print("\nTagged new runs with data_version=v3-clean-er")

    # 4. Add legacy noisy-ER runs (from conversation history)
    print("\nAdding legacy noisy-ER runs...")
    
    # GNN noisy-ER run (Feb 25, 2026)
    add_manual_run(
        CURRENT_DB,
        experiment_name="taluithai-gnn",
        run_name="gnn-L3-h256-e50-noisy-er",
        timestamp_ms=1772054065000,
        params={
            "model": "HeteroGNN",
            "layers": 3,
            "hidden_dim": 256,
            "epochs": 50,
            "learning_rate": 0.001,
            "data_version": "v2-noisy-er",
            "total_videos": 172889,
            "note": "First retrain with unfiltered entity-resolved data (7237 matches)"
        },
        metrics={
            "final_accuracy": 0.925,
            "final_loss": 0.42,
        },
        tags={
            "data_version": "v2-noisy-er",
            "mlflow.note.content": "Noisy ER: 7,237 entity-resolved matches (unfiltered). Accuracy=92.5%"
        }
    )
    
    # Two-Tower noisy-ER run (Feb 25, 2026)
    add_manual_run(
        CURRENT_DB,
        experiment_name="taluithai-two-tower",
        run_name="two-tower-ips-e15-noisy-er",
        timestamp_ms=1772055000000,
        params={
            "model": "TwoTower",
            "epochs": 15,
            "learning_rate": 0.001,
            "loss": "IPS-InfoNCE",
            "temperature": 0.07,
            "data_version": "v2-noisy-er",
            "total_pairs": 210572,
            "note": "First retrain with unfiltered entity-resolved data"
        },
        metrics={
            "best_recall_at_50": 0.5907,
            "best_ndcg_at_10": 0.2458,
            "best_recall_at_10": 0.37,
            "best_recall_at_100": 0.68,
        },
        tags={
            "data_version": "v2-noisy-er",
            "mlflow.note.content": "Noisy ER: 7,237 unfiltered matches. Recall@50=0.5907, NDCG@10=0.2458 (regression from v1)"
        }
    )

    # 5. Add original baseline (no ER) runs
    print("\nAdding baseline (no ER) runs...")
    
    add_manual_run(
        CURRENT_DB,
        experiment_name="taluithai-two-tower",
        run_name="two-tower-ips-e15-baseline",
        timestamp_ms=1771200000000,
        params={
            "model": "TwoTower",
            "epochs": 15,
            "learning_rate": 0.001,
            "loss": "IPS-InfoNCE",
            "temperature": 0.07,
            "data_version": "v1-baseline",
            "total_pairs": 201335,
            "note": "Original training with labeled data only (no entity resolution)"
        },
        metrics={
            "best_recall_at_50": 0.5969,
            "best_ndcg_at_10": 0.2529,
        },
        tags={
            "data_version": "v1-baseline",
            "mlflow.note.content": "Baseline: labeled data only (201,335 pairs). Recall@50=0.5969, NDCG@10=0.2529"
        }
    )

    # 6. Final summary
    print("\n" + "=" * 60)
    conn = sqlite3.connect(CURRENT_DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT r.name, r.experiment_id, 
               GROUP_CONCAT(CASE WHEN t.key='data_version' THEN t.value END) as data_ver
        FROM runs r
        LEFT JOIN tags t ON r.run_uuid = t.run_uuid
        GROUP BY r.run_uuid
        ORDER BY r.start_time
    """)
    print(f"{'Run Name':<40} {'Exp':<5} {'Data Version':<20}")
    print("-" * 65)
    for r in cur.fetchall():
        print(f"{r[0]:<40} {r[1]:<5} {r[2] or 'n/a':<20}")
    conn.close()


if __name__ == "__main__":
    main()
