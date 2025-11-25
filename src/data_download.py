"""Download elasticity data from the Materials Project API."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from mp_api.client import MPRester

from . import config
from .utils import ensure_dir, get_env_var, setup_logger

logger = setup_logger(__name__)


SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "composition_reduced",
    "symmetry",
    "elements",
    "nelements",
    "nsites",
    "density",
    "density_atomic",
    "volume",
    "chemsys",
    "is_stable",
]

ELASTICITY_FIELDS = [
    "material_id",
    "elastic_tensor",
    "compliance_tensor",
    "bulk_modulus",
    "shear_modulus",
    "youngs_modulus",
    "universal_anisotropy",
    "homogeneous_poisson",
]


def _collect_summary_docs(api_key: str) -> pd.DataFrame:
    logger.info("Fetching structural/compositional metadata from Materials Project summary endpoint...")
    docs = []
    with MPRester(api_key) as mpr:
        for doc in mpr.summary.search(
            fields=SUMMARY_FIELDS,
            chunk_size=1000,
            has_props=["elasticity"],
        ):
            docs.append(doc.dict())
    if not docs:
        raise RuntimeError("Summary endpoint returned zero records. Check API key or query parameters.")
    return pd.json_normalize(docs)


def _collect_elasticity_docs(api_key: str) -> pd.DataFrame:
    logger.info("Fetching elasticity tensors from Materials Project elasticity endpoint...")
    docs = []
    with MPRester(api_key) as mpr:
        for doc in mpr.elasticity.search(fields=ELASTICITY_FIELDS, chunk_size=1000):
            docs.append(doc.dict())
    if not docs:
        raise RuntimeError("Elasticity endpoint returned zero records. Check API key or permissions.")
    return pd.json_normalize(docs)


def download_elasticity_data(output_path: str | Path) -> Path:
    """Download elasticity entries and save to disk as CSV."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    api_key = get_env_var("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY is not set. Please export your Materials Project API key.")

    summary_df = _collect_summary_docs(api_key)
    elasticity_df = _collect_elasticity_docs(api_key)
    merged = summary_df.merge(
        elasticity_df,
        on="material_id",
        how="inner",
        suffixes=("", "_elasticity"),
    )
    drop_cols = [col for col in merged.columns if col.endswith("_elasticity")]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    logger.info("Fetched %d merged records with elasticity tensors.", len(merged))
    if merged.empty:
        raise RuntimeError("Failed to merge summary and elasticity datasets. No overlapping material_ids.")

    merged.to_csv(output_path, index=False)
    logger.info("Saved raw elasticity dataset to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Materials Project elasticity data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RAW_DATA_PATH,
        help="Destination CSV path for the raw dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_elasticity_data(args.output)


if __name__ == "__main__":
    main()

