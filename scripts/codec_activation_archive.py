#!/usr/bin/env python3
"""Create a lossy codec archive (BF16 truncation, FP8, zfp, or SZ3) from an uncompressed archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.codec_archive import CODECS, compress_archive_codec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Uncompressed reference archive")
    parser.add_argument("destination", type=Path, help="New codec archive directory")
    parser.add_argument("--codec", choices=sorted(CODECS), required=True)
    parser.add_argument("--mantissa-bits", type=int, help="bf16_trunc_zstd: mantissa bits kept (0-7)")
    parser.add_argument("--zstd-level", type=int, help="bf16_trunc_zstd: Zstandard level (default 3)")
    parser.add_argument("--rate", type=float, help="zfp_fixed_rate: bits per value")
    parser.add_argument("--tolerance", type=float, help="sz3_abs_rms: absolute error bound as a fraction of block RMS")
    args = parser.parse_args()
    params = {
        name: value
        for name, value in {
            "mantissa_bits": args.mantissa_bits,
            "zstd_level": args.zstd_level,
            "rate": args.rate,
            "tolerance": args.tolerance,
        }.items()
        if value is not None
    }
    result = compress_archive_codec(args.source, args.destination, args.codec, **params)
    print(json.dumps({key: value for key, value in result.items() if key != "chunks"}, indent=2))
    print(f"archive_path: {args.destination}")


if __name__ == "__main__":
    main()
