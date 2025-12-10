"""Common utilities for logging, reproducibility, and path handling.

This file was created by prompting Cursor with:
"Create utility functions for logging setup, random seed management, directory creation, environment variable loading, and JSON serialization"
"""

from __future__ import annotations
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict
import numpy as np
from dotenv import load_dotenv


def setup_logger(name: str = "elastic") -> logging.Logger:
    """Configure (or retrieve) a project logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and hash randomness for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def get_env_var(key: str, default: str | None = None, load_dotenv_file: bool = True) -> str | None:
    """Fetch environment variables, optionally loading from a .env file first."""
    if load_dotenv_file:
        load_dotenv()
    value = os.getenv(key, default)
    return value


def serialize_json(data: Dict[str, Any], path: Path) -> None:
    """Persist a JSON serializable mapping to disk."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

