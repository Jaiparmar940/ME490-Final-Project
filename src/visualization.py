"""Reusable plotting helpers."""

from __future__ import annotations
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve
from .utils import ensure_dir


def plot_target_distribution(target: pd.Series, save_path: Path | None = None) -> None:
    """Plot a histogram + KDE for the target."""
    plt.figure(figsize=(6, 4))
    sns.histplot(target, bins=40, kde=True, color="royalblue")
    plt.xlabel("Young's Modulus (GPa)")
    plt.ylabel("Count")
    plt.title("Target Distribution")
    if save_path:
        ensure_dir(save_path.parent)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path | None = None) -> None:
    """Scatter plot of predictions vs truth with y=x reference."""
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Predictions vs True")
    plt.xlim(lims)
    plt.ylim(lims)
    if save_path:
        ensure_dir(save_path.parent)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residual_histogram(residuals: np.ndarray, save_path: Path | None = None) -> None:
    """Plot residual distribution."""
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=40, kde=True, color="tomato")
    plt.xlabel("Residual (True - Pred)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    if save_path:
        ensure_dir(save_path.parent)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_importances(importances: Sequence[float], feature_names: Sequence[str], save_path: Path | None = None) -> None:
    """Horizontal bar plot for feature importances."""
    order = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=np.array(importances)[order], y=np.array(feature_names)[order], orient="h")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances")
    if save_path:
        ensure_dir(save_path.parent)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


def analyze_feature_importance_by_group(
    importances: Sequence[float], feature_names: Sequence[str]
) -> dict:
    """Analyze feature importances grouped by feature type.
    
    Returns a dictionary with grouped analysis including:
    - Total importance per group
    - Top 5 features per group
    - Percentage contribution
    """
    import pandas as pd
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Group features
    magpie_features = importance_df[importance_df['feature'].str.startswith('MagpieData')]
    element_fraction_features = importance_df[importance_df['feature'].str.startswith('ElementFraction')]
    engineered_features = importance_df[importance_df['feature'].isin([
        'mean_atomic_number', 'max_atomic_number', 'avg_valence_electrons'
    ])]
    symmetry_features = importance_df[
        importance_df['feature'].str.startswith('crys_') | 
        importance_df['feature'].str.startswith('sg_')
    ]
    
    # Calculate totals
    total_all = importance_df['importance'].sum()
    
    groups = {
        'Magpie Statistical Features': {
            'df': magpie_features,
            'total': magpie_features['importance'].sum(),
            'percentage': (magpie_features['importance'].sum() / total_all) * 100,
            'top_5': magpie_features.head(5).to_dict('records')
        },
        'ElementFraction Features': {
            'df': element_fraction_features,
            'total': element_fraction_features['importance'].sum(),
            'percentage': (element_fraction_features['importance'].sum() / total_all) * 100,
            'top_5': element_fraction_features.head(5).to_dict('records')
        },
        'Engineered Features': {
            'df': engineered_features,
            'total': engineered_features['importance'].sum(),
            'percentage': (engineered_features['importance'].sum() / total_all) * 100,
            'top_5': engineered_features.head(5).to_dict('records')
        },
        'Symmetry Features': {
            'df': symmetry_features,
            'total': symmetry_features['importance'].sum(),
            'percentage': (symmetry_features['importance'].sum() / total_all) * 100,
            'top_5': symmetry_features.head(5).to_dict('records')
        }
    }
    
    return groups


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """Plot a correlation heatmap of the provided dataframe columns."""
    corr = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="viridis", annot=False)
    plt.title("Correlation Heatmap")
    if save_path:
        ensure_dir(save_path.parent)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_learning_curves(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    save_path: Path | None = None,
    cv: int = 5,
    train_sizes: Sequence[float] | None = None,
    scoring: str = "neg_mean_absolute_error",
    n_jobs: int = -1,
) -> None:
    """Plot learning curves showing train/validation scores vs training set size.
    
    Args:
        estimator: Fitted or unfitted scikit-learn estimator
        X: Feature matrix
        y: Target vector
        save_path: Path to save the figure
        cv: Number of cross-validation folds
        train_sizes: Relative or absolute numbers of training examples to use.
            If None, uses np.linspace(0.2, 1.0, 5) (20%, 40%, 60%, 80%, 100% of training data)
        scoring: Scoring metric (default: neg_mean_absolute_error)
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.2, 1.0, 5)  # 5 points: 20%, 40%, 60%, 80%, 100%
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=42,
    )
    
    # Convert negative MAE to positive MAE for plotting
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.fill_between(
        train_sizes_abs,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="steelblue",
        label="Train ± 1 std",
    )
    plt.fill_between(
        train_sizes_abs,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.1,
        color="coral",
        label="Validation ± 1 std",
    )
    plt.plot(train_sizes_abs, train_scores_mean, "o-", color="steelblue", label="Train MAE")
    plt.plot(train_sizes_abs, val_scores_mean, "o-", color="coral", label="Validation MAE")
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Absolute Error (GPa)")
    plt.title("Learning Curves")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    if save_path:
        ensure_dir(save_path.parent)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()

