"""Training and hyperparameter tuning for elastic modulus models.

This file was created by prompting Cursor with:
"Create training functions for Random Forest and MLP models using RandomizedSearchCV with 5-fold cross-validation, including train/test split persistence"

Hyperparameter tuning was performed by Jay Parmar.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple
import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from . import config
from .models import create_mlp_pipeline, create_random_forest_pipeline, get_param_grids
from .utils import ensure_dir, serialize_json, set_seed, setup_logger

logger = setup_logger(__name__)


def load_feature_dataset(features_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and target vector from disk."""
    if features_path.suffix == ".parquet":
        df = pd.read_parquet(features_path)
    else:
        df = pd.read_csv(features_path)
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    return X, y


def _count_param_options(param_grid: Dict) -> int:
    """Compute the total number of parameter combinations in a grid."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def _run_search(
    estimator,
    param_grid: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 15,
) -> RandomizedSearchCV:
    """Execute RandomizedSearchCV for a given estimator."""
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=config.N_FOLDS,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=config.RANDOM_SEED,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def train_and_tune_models(features_path: Path = config.FEATURE_PATH) -> Dict[str, Dict]:
    """Train RF and MLP models with CV tuning, persisting best estimators."""
    set_seed(config.RANDOM_SEED)
    X, y = load_feature_dataset(features_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )
    split_metadata = {
        "train_indices": X_train.index.tolist(),
        "test_indices": X_test.index.tolist(),
    }
    serialize_json(split_metadata, config.SPLIT_METADATA_PATH)

    models = {
        "random_forest": create_random_forest_pipeline(),
        "mlp": create_mlp_pipeline(),
    }
    param_grids = get_param_grids()

    results: Dict[str, Dict] = {}
    ensure_dir(config.MODEL_DIR)

    for name, estimator in models.items():
        logger.info("Tuning %s model...", name)
        grid_size = _count_param_options(param_grids[name])
        search = _run_search(
            estimator=estimator,
            param_grid=param_grids[name],
            X_train=X_train,
            y_train=y_train,
            n_iter=min(20, grid_size),
        )
        best_estimator = search.best_estimator_
        model_path = config.RF_MODEL_PATH if name == "random_forest" else config.MLP_MODEL_PATH
        joblib.dump(best_estimator, model_path)
        logger.info("Saved %s best estimator to %s", name, model_path)

        cv_results_df = pd.DataFrame(search.cv_results_)
        cv_results_path = config.MODEL_DIR / f"{name}_cv_results.csv"
        cv_results_df.to_csv(cv_results_path, index=False)

        meta = {
            "best_params": search.best_params_,
            "best_score_MAE": -search.best_score_,
            "model_path": str(model_path),
            "cv_results_path": str(cv_results_path),
        }
        results[name] = meta

    # Persist the held-out split for evaluation
    joblib.dump({"X_test": X_test, "y_test": y_test}, config.TEST_SET_PATH)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RandomForest and MLP models.")
    parser.add_argument("--features", type=Path, default=config.FEATURE_PATH, help="Path to feature matrix file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = train_and_tune_models(args.features)
    for name, meta in results.items():
        logger.info("%s | Best MAE: %.3f | Params: %s", name, meta["best_score_MAE"], meta["best_params"])


if __name__ == "__main__":
    main()

