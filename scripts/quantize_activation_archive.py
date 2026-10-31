#!/usr/bin/env python3
"""Create an affine Quant8 archive from an uncompressed archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.quantized_archive import compress_archive_quant8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Uncompressed reference archive")
    parser.add_argument("destination", type=Path, help="New Quant8 archive directory")
    parser.add_argument(
        "--granularity",
        choices=["global", "layer_site", "layer_site_dim"],
        default="global",
        help="One scale for the archive, per (layer, activation_site), or per dimension within each group",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        nargs="+",
        help="Reference archives from other games to fit scales on; default fits on the source (in-sample)",
    )
    parser.add_argument(
        "--store-clipped-exceptions",
        action="store_true",
        help="Store exact BF16 bits for every clipped value in a sparse per-chunk exception table",
    )
    args = parser.parse_args()
    result = compress_archive_quant8(
        args.source,
        args.destination,
        granularity=args.granularity,
        calibration_paths=args.calibration,
        store_clipped_exceptions=args.store_clipped_exceptions,
    )
    print(json.dumps(result, indent=2))
    print(f"archive_path: {args.destination}")


if __name__ == "__main__":
    main()