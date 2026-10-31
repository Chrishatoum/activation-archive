"""Shared dataset-loading helpers for activation benchmarks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.load_activations import read_row_group_bits


def load_row_group(game_id: str, row_group: int, root: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Load the exact BF16 bit data for one row group in one game.

    The returned `metadata` dictionary includes the row metadata columns, and the
    array contains the exact activation bit patterns as uint16 values, preserving
    the original BF16 bit layout.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    record = next(item for item in manifest["games"] if item["game_id"] == game_id)
    parquet_path = root / record["activation_file"]
    return read_row_group_bits(parquet_path, row_group)


def load_row_group_or_stream(game_id: str, row_group: int | None, root: Path, stream_all: bool = False):
    """Placeholder for the eventual streaming-all mode.

    For now this is intentionally simple: a single row group or a future loop over
    all row groups in the same game.
    """
    if stream_all:
        raise NotImplementedError("Full-stream mode is not implemented yet; use --row-group for the first benchmark.")
    return load_row_group(game_id, row_group, root)
