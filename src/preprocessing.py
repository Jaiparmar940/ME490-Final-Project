"""Data cleaning utilities for elasticity dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


def _derive_youngs_modulus(df: pd.DataFrame) -> pd.Series:
    """Derive VRH Young's modulus from available flattened columns."""

    candidates = [
        "youngs_modulus.vrh",
        "youngs_modulus",
        "young_modulus.vrh",
        "young_modulus",
        "elasticity.youngs_modulus_vrh",
        "elasticity.youngs_modulus",
        "elasticity.young_modulus_vrh",
        "elasticity.young_modulus",
    ]
    youngs = pd.Series(np.nan, index=df.index, dtype=float)
    for col in candidates:
        if col in df.columns:
            youngs = pd.to_numeric(df[col], errors="coerce")
            break

    if youngs.isna().all():
        logger.warning("Young's modulus columns missing; attempting VRH reconstruction from bulk/shear moduli.")

    bulk_candidates = ["bulk_modulus.vrh", "bulk_modulus", "elasticity.bulk_modulus_vrh"]
    shear_candidates = ["shear_modulus.vrh", "shear_modulus", "elasticity.shear_modulus_vrh"]

    bulk = None
    shear = None
    for col in bulk_candidates:
        if col in df.columns:
            bulk = pd.to_numeric(df[col], errors="coerce")
            break
    for col in shear_candidates:
        if col in df.columns:
            shear = pd.to_numeric(df[col], errors="coerce")
            break

    needs_fallback = youngs.isna()
    if needs_fallback.any() and bulk is not None and shear is not None:
        denom = 3 * bulk + shear
        with np.errstate(divide="ignore", invalid="ignore"):
            approx = (9 * bulk * shear) / denom
        youngs = youngs.where(~needs_fallback, approx)

    return youngs


def clean_dataset(raw_path: Path) -> pd.DataFrame:
    """Load raw CSV and perform cleaning/feature derivations."""
    logger.info("Loading raw dataset from %s", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Loaded %d rows with %d columns", len(df), len(df.columns))

    df["youngs_modulus_vrh"] = _derive_youngs_modulus(df)
    df = df[df["youngs_modulus_vrh"].notnull()].copy()
    logger.info("Rows with valid Young's modulus: %d", len(df))

    # Remove duplicates by material_id (keep most recent entry)
    if "material_id" in df.columns:
        df = df.drop_duplicates(subset="material_id", keep="first")
        logger.info("Rows after deduplication: %d", len(df))

    # Remove unrealistic values using percentile clipping
    lower, upper = df["youngs_modulus_vrh"].quantile([0.01, 0.99])
    df = df[(df["youngs_modulus_vrh"] >= lower) & (df["youngs_modulus_vrh"] <= upper)].copy()
    logger.info("Rows after outlier filtering: %d", len(df))

    # Keep relevant columns
    keep_columns = [
        "material_id",
        "formula_pretty",
        "composition_reduced",
        "symmetry.crystal_system",
        "symmetry.space_group_symbol",
        "spacegroup.number",
        "symmetry.number",
        "bulk_modulus.vrh",
        "shear_modulus.vrh",
        "youngs_modulus.vrh",
        "youngs_modulus",
        "youngs_modulus_vrh",
    ]
    available_columns = [col for col in keep_columns if col in df.columns]
    df = df[available_columns]

    return df


def save_clean_dataset(raw_path: Path, output_path: Path) -> Path:
    """Clean raw data and persist the processed CSV."""
    ensure_dir(output_path.parent)
    cleaned_df = clean_dataset(raw_path)
    cleaned_df.to_csv(output_path, index=False)
    logger.info("Saved cleaned dataset with %d rows to %s", len(cleaned_df), output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw elasticity dataset.")
    parser.add_argument("--input", type=Path, default=config.RAW_DATA_PATH, help="Path to raw CSV file.")
    parser.add_argument("--output", type=Path, default=config.CLEANED_DATA_PATH, help="Path for cleaned CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_clean_dataset(args.input, args.output)


if __name__ == "__main__":
    main()

