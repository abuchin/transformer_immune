"""
Configuration settings for the Transformer Immune Cells project.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data processing parameters
MAX_SEQUENCE_LENGTH = 512
N_TOP_GENES = 2000
TARGET_SUM = 1e6
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42

# Model parameters
NUM_LABELS = 3
LEARNING_RATE = 3e-5
BATCH_SIZE = 128
NUM_EPOCHS = 10
LOGGING_STEPS = 50

# Training parameters
FREEZE_BASE_LAYERS = True
UNFREEZE_LAST_N_LAYERS = 1

# File paths
GENE_TOKEN_DICT_FILENAME = "gene_token_dict.json"
GENES_FILENAME = "genes.npy"
GENE_TOKENS_FILENAME = "gene_tokens.npy"

# Google Drive paths (for Colab)
GOOGLE_DRIVE_BASE = "/content/drive/Othercomputers/My MacBook Pro (2)/SCIENCE/Projects/immunology_transformer"
RAW_DATA_PATH = f"{GOOGLE_DRIVE_BASE}/data/human_immune_health_atlas_nk-ilc.h5ad"
PREPROCESSED_DATA_PATH = f"{GOOGLE_DRIVE_BASE}/data/preprocessed/full_data"
MODEL_SAVE_PATH = f"{GOOGLE_DRIVE_BASE}/model/trained_model"

# Geneformer model
GENEFORMER_MODEL_NAME = "ctheodoris/Geneformer"

# Species for gene mapping
SPECIES = "human"

# Parallel processing
N_JOBS = 4 