"""Shared paths for benchmark output files."""

from __future__ import annotations

from pathlib import Path


def row_group_result_path(
    game_id: str,
    row_group: int,
    method: str,
    variant: str | None = None,
) -> Path:
    """Return the default path for one method's row-group result."""
    filename = method
    if variant:
        filename += f"_{variant}"
    return Path("benchmarks/results") / game_id / "row_groups" / f"{row_group:04d}" / f"{filename}.json"