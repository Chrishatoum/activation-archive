#!/usr/bin/env python3
"""Create a Zstandard-compressed copy of a reference activation archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.activation_archive import compress_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Uncompressed reference archive")
    parser.add_argument("destination", type=Path, help="New compressed archive directory")
    parser.add_argument("--compression-level", type=int, default=3)
    args = parser.parse_args()
    result = compress_archive(
        args.source,
        args.destination,
        compression_level=args.compression_level,
    )
    print(json.dumps(result, indent=2))
    print(f"archive_path: {args.destination}")


if __name__ == "__main__":
    main()