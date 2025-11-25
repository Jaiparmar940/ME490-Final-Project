"""Convenience script to download Materials Project elasticity data."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_download import download_elasticity_data
from src.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


def main() -> None:
    ensure_dir(config.RAW_DATA_DIR)
    logger.info("Downloading elasticity data to %s", config.RAW_DATA_PATH)
    download_elasticity_data(config.RAW_DATA_PATH)


if __name__ == "__main__":
    main()


