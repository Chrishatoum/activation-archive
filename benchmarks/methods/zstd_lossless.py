#!/usr/bin/env python3
"""Lossless zstandard benchmark for one Parquet row group."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.data import load_row_group
from benchmarks.common.paths import row_group_result_path
from benchmarks.common.schema import make_result, validate_result

DEFAULT_ROOT = Path("/srv/gibbon-activations/gpt_oss_pilot_3games")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", type=str, default="low_0056")
    parser.add_argument("--row-group", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("GIBBON_DATA_ROOT", DEFAULT_ROOT)))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compression-level", type=int, default=3)
    args = parser.parse_args()

    try:
        import zstandard
    except ImportError as exc:
        parser.error("zstandard is required; install the project dependencies from environment.yml")

    read_started = time.perf_counter()
    metadata, bits = load_row_group(args.game_id, args.row_group, args.root)
    read_time = time.perf_counter() - read_started

    raw_bytes = bits.tobytes()
    compressor = zstandard.ZstdCompressor(level=args.compression_level)
    encode_started = time.perf_counter()
    encoded_bytes = compressor.compress(raw_bytes)
    encode_time = time.perf_counter() - encode_started

    decoder = zstandard.ZstdDecompressor()
    decode_started = time.perf_counter()
    decoded_bytes = decoder.decompress(encoded_bytes)
    decode_time = time.perf_counter() - decode_started

    roundtrip_exact = decoded_bytes == raw_bytes
    if not roundtrip_exact:
        raise RuntimeError("zstd round-trip did not reproduce the original BF16 bytes")

    result = make_result(
        method="zstd_lossless",
        game_id=args.game_id,
        row_group=args.row_group,
        rows=int(bits.shape[0]),
        dimensions=int(bits.shape[1]),
        raw_size_bytes=len(raw_bytes),
        encoded_size_bytes=len(encoded_bytes),
        read_time_s=read_time,
        encode_time_s=encode_time,
        decode_time_s=decode_time,
        relative_frobenius_error=0.0,
        max_abs_error=0.0,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        roundtrip_exact=roundtrip_exact,
        metadata_fields=sorted(metadata.keys()),
        method_params={"compression_level": args.compression_level},
        data_root=str(args.root),
    )
    validate_result(result)

    output_path = args.output or row_group_result_path(
        args.game_id,
        args.row_group,
        "zstd_lossless",
        variant=f"level_{args.compression_level}",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
