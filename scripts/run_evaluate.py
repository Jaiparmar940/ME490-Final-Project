"""Script to evaluate trained models and generate figures."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.evaluate import evaluate_models
from src.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


def main() -> None:
    ensure_dir(config.FIGURES_DIR)
    ensure_dir(config.MODEL_DIR)
    logger.info("Evaluating models in %s", config.MODEL_DIR)
    metrics = evaluate_models(
        features_path=config.FEATURE_PATH,
        models_dir=config.MODEL_DIR,
        fig_dir=config.FIGURES_DIR,
    )
    for model_name, model_metrics in metrics.items():
        logger.info("%s metrics: %s", model_name, model_metrics)


if __name__ == "__main__":
    main()


