"""Model evaluation script for elastic modulus predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config
from .train import load_feature_dataset
from .visualization import plot_prediction_scatter, plot_residual_histogram
from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


def _load_test_set(features_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load cached test set or recreate it deterministically."""
    if config.TEST_SET_PATH.exists():
        cached = joblib.load(config.TEST_SET_PATH)
        return cached["X_test"], cached["y_test"]

    X, y = load_feature_dataset(features_path)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )
    joblib.dump({"X_test": X_test, "y_test": y_test}, config.TEST_SET_PATH)
    logger.warning("Test split cache missing. Re-created using deterministic split.")
    return X_test, y_test


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def evaluate_models(
    features_path: Path,
    models_dir: Path,
    fig_dir: Path = config.FIGURES_DIR,
) -> Dict[str, Dict[str, float]]:
    """Evaluate saved models and persist plots."""
    ensure_dir(fig_dir)
    X_test, y_test = _load_test_set(features_path)
    model_paths = {
        "random_forest": models_dir / "random_forest.joblib",
        "mlp": models_dir / "mlp.joblib",
    }

    metrics: Dict[str, Dict[str, float]] = {}
    for name, model_path in model_paths.items():
        if not model_path.exists():
            logger.warning("Model %s not found at %s. Skipping.", name, model_path)
            continue

        model = joblib.load(model_path)
        preds = model.predict(X_test)
        model_metrics = _compute_metrics(y_test, preds)
        metrics[name] = model_metrics
        logger.info(
            "%s Metrics -> MAE: %.3f | RMSE: %.3f | R2: %.3f",
            name,
            model_metrics["MAE"],
            model_metrics["RMSE"],
            model_metrics["R2"],
        )

        plot_prediction_scatter(y_test.values, preds, fig_dir / f"{name}_pred_vs_true.png")
        plot_residual_histogram(y_test.values - preds, fig_dir / f"{name}_residuals.png")

        if name == "random_forest" and hasattr(model.named_steps.get("model"), "feature_importances_"):
            importances = model.named_steps["model"].feature_importances_
            feature_names = X_test.columns
            from .visualization import plot_feature_importances

            plot_feature_importances(importances, feature_names, fig_dir / "rf_feature_importances.png")

        crystal_cols = [c for c in X_test.columns if c.startswith("crys_")]
        if crystal_cols:
            per_system = {}
            for col in crystal_cols:
                mask = X_test[col] == 1
                if mask.sum() < 5:
                    continue
                per_system[col.replace("crys_", "")] = _compute_metrics(
                    y_test[mask],
                    preds[mask],
                )
            if per_system:
                metrics[name]["per_crystal"] = per_system  # type: ignore[index]

    ensure_dir(models_dir)
    joblib.dump(metrics, models_dir / "evaluation_metrics.joblib")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--features", type=Path, default=config.FEATURE_PATH)
    parser.add_argument("--models_dir", type=Path, default=config.MODEL_DIR)
    parser.add_argument("--fig_dir", type=Path, default=config.FIGURES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_models(args.features, args.models_dir, args.fig_dir)
    for name, vals in metrics.items():
        logger.info("%s metrics: %s", name, vals)


if __name__ == "__main__":
    main()

