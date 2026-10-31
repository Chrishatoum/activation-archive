#!/usr/bin/env python3
"""Build an uncompressed reference archive from an activation Parquet file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.activation_archive import build_archive_from_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Source activation Parquet file")
    parser.add_argument("archive", type=Path, help="New archive directory to create")
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--row-group", type=int, help="Convert only this row group")
    parser.add_argument("--batch-rows", type=int, default=1024)
    args = parser.parse_args()
    if args.batch_rows < 1:
        parser.error("--batch-rows must be positive")
    result = build_archive_from_parquet(
        args.parquet,
        args.archive,
        args.game_id,
        batch_rows=args.batch_rows,
        row_group=args.row_group,
    )
    print(json.dumps(result, indent=2))
    print(f"archive_path: {args.archive}")


if __name__ == "__main__":
    main()