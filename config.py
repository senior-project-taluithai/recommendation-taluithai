"""
Configuration for the recommendation training pipeline.
DB connections, hyperparameters, and paths.
"""

import os
from pathlib import Path

# Load .env file if present (for local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
EMBED_DIR = ROOT_DIR / "embeddings"

for d in [DATA_DIR, MODEL_DIR, EMBED_DIR]:
    d.mkdir(exist_ok=True)

# ─── PostgreSQL ───────────────────────────────────────────────────────────────
PG_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", 5432)),
    "database": os.getenv("PG_DATABASE", "taluithai"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}

# ─── MongoDB ──────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_TIKTOK_DB = "tiktok_scraper"

# ─── Qdrant ───────────────────────────────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = "place_recommendations"  # new collection for two-tower

# ─── Text Encoder ─────────────────────────────────────────────────────────────
TEXT_ENCODER_MODEL = "BAAI/bge-m3"
TEXT_EMBED_DIM = 1024

# ─── Two-Tower Hyperparameters ────────────────────────────────────────────────
EMBEDDING_DIM = 128          # final embedding dimension
HIDDEN_DIM = 512             # intermediate MLP dimension
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
EPOCHS = 15
TEMPERATURE = 0.07           # InfoNCE temperature
NUM_WORKERS = 4

# Number of categories & provinces (for embedding layers)
NUM_CATEGORIES = 6           # 5 + 1 padding
NUM_PROVINCES = 78           # 77 + 1 padding
NUM_REGIONS = 7              # 6 regions + 1 padding

# ─── IPS (Inverse Propensity Scoring) ─────────────────────────────────────────
IPS_W_TIKTOK = 0.7           # weight for tiktok video count
IPS_W_REVIEW = 0.3           # weight for google review count
IPS_ETA = 0.5                # power-law smoothing exponent
IPS_CLIP_MIN = 0.01          # minimum propensity score

# ─── GNN Hyperparameters ─────────────────────────────────────────────────────
GNN_HIDDEN_DIM = 256
GNN_EMBED_DIM = 128          # output embedding for place/author
GNN_HASHTAG_EMBED_DIM = 64   # output embedding for hashtags
GNN_NUM_LAYERS = 3
GNN_LEARNING_RATE = 1e-3
GNN_EPOCHS = 50
GNN_BATCH_SIZE = 1024
HASHTAG_MIN_FREQ = 3         # filter hashtags with < 3 occurrences

# ─── Validation ───────────────────────────────────────────────────────────────
VAL_RATIO = 0.1              # 10% for validation
RECALL_K = [10, 50, 100]     # Recall@K metrics

# ─── Province → Region mapping ────────────────────────────────────────────────
# Will be loaded from PG at runtime, but define region enum here
REGION_NAMES = [
    "ภาคเหนือ",       # 1
    "ภาคกลาง",        # 2
    "ภาคตะวันออกเฉียงเหนือ",  # 3
    "ภาคตะวันตก",     # 4
    "ภาคตะวันออก",    # 5
    "ภาคใต้",          # 6
]

# ─── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
MLFLOW_EXPERIMENT_GNN = "taluithai-gnn"
MLFLOW_EXPERIMENT_TWO_TOWER = "taluithai-two-tower"
