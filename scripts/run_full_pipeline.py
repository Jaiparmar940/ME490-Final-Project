"""Entry script to execute the complete elasticity ML pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_download import download_elasticity_data
from src.evaluate import evaluate_models
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
        logger.info("Raw data not found. Downloading...")
        download_elasticity_data(config.RAW_DATA_PATH)
    else:
        logger.info("Raw data already exists at %s. Skipping download.", config.RAW_DATA_PATH)

    if not config.CLEANED_DATA_PATH.exists():
        logger.info("Cleaning dataset...")
        save_clean_dataset(config.RAW_DATA_PATH, config.CLEANED_DATA_PATH)
    else:
        logger.info("Cleaned dataset already available.")

    if not config.FEATURE_PATH.exists():
        logger.info("Generating features...")
        save_features(config.CLEANED_DATA_PATH, config.FEATURE_PATH)
    else:
        logger.info("Feature matrix already exists.")

    logger.info("Training models...")
    train_and_tune_models(config.FEATURE_PATH)

    logger.info("Evaluating models...")
    evaluate_models(
        features_path=config.FEATURE_PATH,
        models_dir=config.MODEL_DIR,
        fig_dir=config.FIGURES_DIR,
    )
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()

