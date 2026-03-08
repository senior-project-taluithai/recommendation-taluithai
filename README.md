# Taluithai Recommendation Pipeline

Hybrid Two-Tower Retrieval + GNN Graph-Enhanced Re-Ranker for Thai tourism place recommendations.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Two-Tower Retrieval (Static, ~10ms)           │
│  Query Tower (TikTok text+engagement) → 128d            │
│  Item Tower  (Place text+meta+GNN emb) → 128d           │
│  ANN search via Qdrant (Cosine, top-100)                │
├─────────────────────────────────────────────────────────┤
│  Stage 2: Graph-Enhanced Re-Ranker (Dynamic, ~15ms)     │
│  8 signals: retrieval, graph_sim, co_author,            │
│  hashtag, category, province, popularity, seen_penalty  │
│  → Weighted sum → top-K final results                   │
└─────────────────────────────────────────────────────────┘
```

## Data Sources

| Source | DB | Key Tables/Collections | Count |
|--------|----|-----------------------|-------|
| Places | PostgreSQL | `tat.places` | 20,135 |
| Events | PostgreSQL | `tat.events` | 21,229 |
| TikTok Videos | MongoDB | `tiktok_scraper.content_metadata` | 180,319 |
| Video-Place Pairs | MongoDB | `tiktok_scraper.place_videos` | 201,335 pairs |
| Authors | MongoDB | (extracted from videos) | 113,746 |
| Hashtags | MongoDB | (extracted from videos) | 252,000 |

## Pipeline Steps

### Step 0: Setup (Local or vast.ai)

```bash
# Clone repo
git clone <repo-url> && cd recommendation-taluithai

# Install dependencies
pip install -r requirements.txt

# For PyG with CUDA (adjust CUDA version):
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

### Step 1: Extract Data

Pulls data from PostgreSQL and MongoDB, saves to `data/` as Parquet and JSON.

```bash
python scripts/01_extract_data.py
```

**Output:**
```
data/
├── places.parquet           # 20,135 places with all features
├── categories.parquet       # 5 categories
├── provinces.parquet        # 77 provinces
├── training_pairs.parquet   # ~201K video-place training pairs
├── video_features.json      # Engagement stats per video
├── graph_data.json          # Edges for heterogeneous graph
└── propensity_scores.json   # IPS weights per place
```

> **Note:** Run this step on a machine with network access to the databases.
> On vast.ai, you can extract locally first and upload `data/` via `scp`.

### Step 2: Train GNN

Builds a heterogeneous graph (places, authors, videos, hashtags) from TikTok data and trains a SimplifiedGNN with BPR loss. Exports 128d place/author embeddings.

```bash
python scripts/02_train_gnn.py --epochs 50 --lr 0.001
```

**Output:**
```
models/gnn_best.pt
embeddings/
├── gnn_place_embeddings.npz     # (20K, 128)
├── gnn_author_embeddings.npz    # (114K, 128)
├── gnn_hashtag_embeddings.npz   # (n_hashtags, 128)
├── place_to_authors.json        # For re-ranker
└── place_to_hashtags.json       # For re-ranker
```

**GPU Memory:** ~2-4 GB for ~386K nodes, ~900K edges.

### Step 3: Train Two-Tower

Trains the Query-Item Two-Tower with IPS-weighted InfoNCE loss. Optionally uses GNN embeddings as input features.

```bash
# With GNN embeddings (recommended)
python scripts/03_train_two_tower.py --graph_embeds embeddings/gnn_place_embeddings.npz

# Without GNN
python scripts/03_train_two_tower.py --no_ips

# Full options
python scripts/03_train_two_tower.py \
  --epochs 15 \
  --lr 1e-4 \
  --batch_size 256 \
  --temperature 0.07 \
  --use_ips \
  --wandb
```

**First run** encodes ~221K texts with bge-m3 (~30 min on GPU), then caches to disk for subsequent runs.

**Output:**
```
models/
├── two_tower_best.pt    # Best by Recall@50
└── two_tower_final.pt   # Last epoch
```

**Key Metrics:** Recall@10, Recall@50, Recall@100, NDCG@10

### Step 4: Export to Qdrant

Pre-computes item embeddings for all places and uploads to Qdrant for real-time retrieval.

```bash
# Full export (compute + upload)
python scripts/04_export_qdrant.py

# Dry run (compute only, e.g., on vast.ai without Qdrant access)
python scripts/04_export_qdrant.py --dry_run

# Then upload from local machine
python scripts/04_export_qdrant.py --model models/two_tower_best.pt
```

**Output:**
```
embeddings/two_tower_place_embeddings.npz   # (20K, 128)
models/query_tower.pt                       # For inference API
models/item_tower.pt                        # For re-indexing
models/query_tower_traced.pt                # TorchScript (optional)
Qdrant: place_recommendations collection   # 20K vectors, 128d, Cosine
```

## vast.ai Workflow

```bash
# 1. On local machine: extract data
python scripts/01_extract_data.py

# 2. Upload to vast.ai
scp -r data/ root@<vast-ip>:/workspace/recommendation-taluithai/
scp requirements.txt config.py root@<vast-ip>:/workspace/recommendation-taluithai/

# 3. On vast.ai GPU:
pip install -r requirements.txt
python scripts/02_train_gnn.py
python scripts/03_train_two_tower.py
python scripts/04_export_qdrant.py --dry_run

# 4. Download results
scp -r root@<vast-ip>:/workspace/recommendation-taluithai/models/ ./
scp -r root@<vast-ip>:/workspace/recommendation-taluithai/embeddings/ ./

# 5. On local: upload to Qdrant
python scripts/04_export_qdrant.py
```

## Project Structure

```
recommendation-taluithai/
├── config.py                      # All configs, DB connections, hyperparams
├── requirements.txt               # Python dependencies
├── README.md
├── src/
│   ├── data/
│   │   ├── extract.py             # PostgreSQL + MongoDB extraction
│   │   └── dataset.py             # PyTorch Dataset, bge-m3 encoding
│   ├── models/
│   │   ├── query_tower.py         # Query Tower MLP (1028→128)
│   │   ├── item_tower.py          # Item Tower MLP (1205→128)
│   │   ├── two_tower.py           # Combined model
│   │   └── gnn.py                 # SimplifiedGNN + HeteroGraphSAGE
│   └── utils/
│       ├── loss.py                # IPS-weighted InfoNCE
│       └── metrics.py             # Recall@K, NDCG@K
├── scripts/
│   ├── 01_extract_data.py         # Step 1: Data extraction
│   ├── 02_train_gnn.py            # Step 2: GNN training
│   ├── 03_train_two_tower.py      # Step 3: Two-Tower training
│   └── 04_export_qdrant.py        # Step 4: Qdrant export
├── data/                          # Extracted data (git-ignored)
├── models/                        # Model checkpoints (git-ignored)
└── embeddings/                    # Exported embeddings (git-ignored)
```

## Key Design Decisions

- **Query = TikTok video** (not user_id) — no user interaction data, use video content as proxy
- **IPS Debiasing** — Power-Law propensity: `P_i = (pop_i/max)^0.5` to upweight unpopular places
- **GNN Graph Embeddings** — capture co-occurrence (same-author, same-hashtag) structure from TikTok
- **bge-m3** — multilingual text encoder (Thai + English), 1024d output
- **Graph Gate** in Item Tower — learned sigmoid gate on GNN embeddings (handles missing gracefully)

## Environment Variables

Override defaults via env vars:

```bash
export PG_HOST=34.87.52.21
export PG_PASSWORD="uRv0|RVoo!<1y1}X<%G9W&-NcLw(H15y"
export MONGO_URI="mongodb://..."
export QDRANT_HOST=db-taluithai.oswinfalk.xyz
export QDRANT_PORT=6333
```

## Requirements

- Python 3.10+
- CUDA GPU recommended (vast.ai: RTX 3090 / A100)
- ~8 GB VRAM for Two-Tower training (bge-m3 encoding)
- ~2-4 GB VRAM for GNN training
