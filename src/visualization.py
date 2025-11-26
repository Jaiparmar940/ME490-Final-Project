"""Reusable plotting helpers."""

from __future__ import annotations
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

