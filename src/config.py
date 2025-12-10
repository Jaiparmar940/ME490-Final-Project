"""Central configuration for the elastic-modulus project.

This file was created by prompting Cursor with:
"Create a central configuration file for an elastic modulus ML project with paths, random seeds, and hyperparameter grids for Random Forest and MLP models"

Design choices (hyperparameter grids, split ratios, random seed) were made by Jay Parmar.
Hyperparameter tuning was performed by Jay Parmar.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
FEATURE_PATH = PROCESSED_DATA_DIR / "elasticity_features.parquet"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "elasticity_cleaned.csv"
RAW_DATA_PATH = RAW_DATA_DIR / "elasticity_raw.csv"

# Model storage
MODEL_DIR = PROJECT_ROOT / "models"
RF_MODEL_PATH = MODEL_DIR / "random_forest.joblib"
MLP_MODEL_PATH = MODEL_DIR / "mlp.joblib"

# Cached artifacts
SPLIT_METADATA_PATH = PROCESSED_DATA_DIR / "train_test_split_indices.json"
TEST_SET_PATH = PROCESSED_DATA_DIR / "test_set.joblib"

# Target and split settings
TARGET_COLUMN = "youngs_modulus_vrh"
TEST_SIZE = 0.2
RANDOM_SEED = 42
N_FOLDS = 5

# Hyperparameter grids
RF_PARAM_GRID = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [None, 15, 25],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", 0.6],
}

MLP_PARAM_GRID = {
    "model__hidden_layer_sizes": [(128, 128), (256, 128), (256, 128, 64)],
    "model__alpha": [1e-4, 1e-3, 1e-2],
    "model__learning_rate_init": [1e-3, 5e-3],
    "model__activation": ["relu", "tanh"],
}

