"""Script to clean data, build features, and train models."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.features import save_features
from src.preprocessing import save_clean_dataset
from src.train import train_and_tune_models
from src.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


def main() -> None:
    ensure_dir(config.DATA_DIR)
    ensure_dir(config.RAW_DATA_DIR)
    ensure_dir(config.PROCESSED_DATA_DIR)
    ensure_dir(config.MODEL_DIR)

    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {config.RAW_DATA_PATH}. "
            "Run scripts/run_download.py before training."
        )

    if not config.CLEANED_DATA_PATH.exists():
        logger.info("Cleaning raw dataset at %s", config.RAW_DATA_PATH)
        save_clean_dataset(config.RAW_DATA_PATH, config.CLEANED_DATA_PATH)
    else:
        logger.info("Using existing cleaned dataset at %s", config.CLEANED_DATA_PATH)

    if not config.FEATURE_PATH.exists():
        logger.info("Building feature matrix at %s", config.FEATURE_PATH)
        save_features(config.CLEANED_DATA_PATH, config.FEATURE_PATH)
    else:
        logger.info("Using cached features at %s", config.FEATURE_PATH)

    logger.info("Training and tuning models...")
    train_and_tune_models(config.FEATURE_PATH)


if __name__ == "__main__":
    main()


