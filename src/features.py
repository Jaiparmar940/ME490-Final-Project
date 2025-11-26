"""Feature engineering using matminer for elastic modulus prediction."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementFraction, ElementProperty
from pymatgen.core import Composition
from . import config
from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


def _build_composition_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Add a pymatgen Composition object column."""

    def to_composition(row: pd.Series) -> Composition | None:
        formula = row.get("formula_pretty")
        if pd.isna(formula):
            return None
        try:
            return Composition(formula)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to parse composition %s (material_id=%s): %s", formula, row.get("material_id"), exc)
            return None

    df["composition_obj"] = df.apply(to_composition, axis=1)
    return df


def _add_engineered_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple heuristic descriptors from Composition objects."""
    mean_atomic_numbers = []
    max_atomic_numbers = []
    valence_e_counts = []

    for comp in df["composition_obj"]:
        elements = list(comp.elements)
        fractions = np.array([comp.get_atomic_fraction(el) for el in elements])
        numbers = np.array([el.Z for el in elements])
        mean_atomic_numbers.append(float(np.dot(numbers, fractions)))
        max_atomic_numbers.append(float(numbers.max()))
        valence_e_counts.append(float(np.dot([el.common_oxidation_states[0] if el.common_oxidation_states else 0 for el in elements], fractions)))

    df["mean_atomic_number"] = mean_atomic_numbers
    df["max_atomic_number"] = max_atomic_numbers
    df["avg_valence_electrons"] = valence_e_counts
    return df


def engineer_features(cleaned_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate composition + symmetry features and return X, y."""
    df = pd.read_csv(cleaned_path)
    df = (
        df.dropna(subset=[config.TARGET_COLUMN, "formula_pretty"])
        .reset_index(drop=True)
    )
    df = _build_composition_objects(df)
    df = df[df["composition_obj"].notnull()].reset_index(drop=True)

    featurizer = MultipleFeaturizer(
        [
            ElementFraction(),
            ElementProperty.from_preset("magpie"),
        ]
    )
    df = featurizer.featurize_dataframe(df, "composition_obj", ignore_errors=True)
    df = _add_engineered_stats(df)

    df["crystal_system"] = df.get(
        "symmetry.crystal_system",
        pd.Series(["unknown"] * len(df), index=df.index),
    ).fillna("unknown")
    crystal_dummies = pd.get_dummies(df["crystal_system"], prefix="crys")
    df = pd.concat([df, crystal_dummies], axis=1)

    space_groups = df.get("spacegroup.number")
    if space_groups is None:
        space_groups = df.get("symmetry.number")
    if space_groups is None:
        space_groups = pd.Series([-1] * len(df), index=df.index)
    space_groups = space_groups.fillna(-1).astype(int)
    bins = [-1, 50, 100, 150, 200, 230]
    labels = ["sg_1_50", "sg_51_100", "sg_101_150", "sg_151_200", "sg_201_230"]
    df["sg_bin"] = pd.cut(space_groups, bins=bins, labels=labels, include_lowest=True)
    sg_dummies = pd.get_dummies(df["sg_bin"], prefix="", prefix_sep="")
    df = pd.concat([df, sg_dummies], axis=1)

    feature_columns = [
        col
        for col in df.columns
        if col.startswith("MagpieData")
        or col.startswith("ElementFraction")
        or col.startswith("crys_")
        or (col.startswith("sg_") and col != "sg_bin")
        or col in ["mean_atomic_number", "max_atomic_number", "avg_valence_electrons"]
    ]

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[config.TARGET_COLUMN].astype(float)
    return X, y


def save_features(cleaned_path: Path, output_path: Path) -> Path:
    """Create the feature matrix and persist it."""
    ensure_dir(output_path.parent)
    X, y = engineer_features(cleaned_path)
    dataset = pd.concat([X, y.rename(config.TARGET_COLUMN)], axis=1)
    if output_path.suffix == ".parquet":
        dataset.to_parquet(output_path, index=False)
    else:
        dataset.to_csv(output_path, index=False)
    logger.info("Saved feature matrix with %d rows and %d columns to %s", len(dataset), dataset.shape[1], output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate features from cleaned elasticity data.")
    parser.add_argument("--input", type=Path, default=config.CLEANED_DATA_PATH, help="Cleaned CSV input.")
    parser.add_argument("--output", type=Path, default=config.FEATURE_PATH, help="Path to save feature matrix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_features(args.input, args.output)


if __name__ == "__main__":
    main()

