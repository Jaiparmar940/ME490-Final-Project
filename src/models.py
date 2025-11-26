"""Model definitions and helper utilities."""

from __future__ import annotations
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from . import config


def create_random_forest_pipeline() -> Pipeline:
    """Build a RandomForest regression pipeline."""
    return Pipeline(
        steps=[
            ("model", RandomForestRegressor(random_state=config.RANDOM_SEED, n_estimators=400)),
        ]
    )


def create_mlp_pipeline() -> Pipeline:
    """Build an MLP regression pipeline with feature scaling."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    max_iter=1000,
                    learning_rate_init=1e-3,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def get_param_grids() -> dict:
    """Retrieve hyperparameter search grids for the supported models."""
    return {
        "random_forest": config.RF_PARAM_GRID,
        "mlp": config.MLP_PARAM_GRID,
    }

