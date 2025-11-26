"""Utility script to delete generated data/model artifacts."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.utils import ensure_dir, setup_logger  # noqa: E402

logger = setup_logger(__name__)

TARGET_PATHS = [
    config.RAW_DATA_DIR,
    config.PROCESSED_DATA_DIR,
    config.MODEL_DIR,
]


def _wipe_path(path: Path, dry_run: bool) -> None:
    """Remove path entirely, honoring dry-run mode."""
    if not path.exists():
        logger.info("Skipping %s (not found)", path)
        return

    if dry_run:
        logger.info("[dry-run] Would delete %s", path)
        return

    if path.is_file() or path.is_symlink():
        path.unlink()
    else:
        shutil.rmtree(path)
    logger.info("Deleted %s", path)


def cleanup_artifacts(dry_run: bool = False, recreate_dirs: bool = False) -> None:
    """Delete generated datasets, processed artifacts, and trained models."""
    for target in TARGET_PATHS:
        _wipe_path(target, dry_run=dry_run)
        if recreate_dirs and not dry_run:
            ensure_dir(target)
            logger.info("Recreated empty directory %s", target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove downloaded/processed datasets and model artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Only log paths that would be deleted.")
    parser.add_argument(
        "--recreate-dirs",
        action="store_true",
        help="Recreate empty target directories after deletion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleanup_artifacts(dry_run=args.dry_run, recreate_dirs=args.recreate_dirs)


if __name__ == "__main__":
    main()


