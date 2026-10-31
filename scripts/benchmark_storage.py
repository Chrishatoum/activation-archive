#!/usr/bin/env python3
"""Minimal benchmark for exact BF16 row-group storage and lossless compression.

This script is intentionally small and focused on the first project task:
measure the baseline cost of reading a row group and prototype a lossless
representation that preserves the exact BF16 bits and metadata alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ROOT = Path("/srv/gibbon-activations/gpt_oss_pilot_3games")


def _load_loader_module() -> Any:
    loader_path = Path(__file__).resolve().parents[1] / "python" / "load_activations.py"
    spec = importlib.util.spec_from_file_location("load_activations", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load loader from {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row_group_bytes_and_hash(bits: np.ndarray) -> tuple[bytes, str, float]:
    t0 = time.perf_counter()
    raw = bits.tobytes()
    elapsed = time.perf_counter() - t0
    return raw, hashlib.sha256(raw).hexdigest(), elapsed


def _benchmark_method(name: str, payload: bytes) -> dict[str, Any]:
    t0 = time.perf_counter()
    if name == "raw":
        encoded = payload
        decoded = payload
        encode_time = time.perf_counter() - t0
        decode_time = 0.0
    elif name == "zlib":
        t1 = time.perf_counter()
        encoded = zlib.compress(payload, level=9)
        encode_time = time.perf_counter() - t1
        t2 = time.perf_counter()
        decoded = zlib.decompress(encoded)
        decode_time = time.perf_counter() - t2
    else:
        raise ValueError(f"Unknown benchmark method: {name}")

    return {
        "method": name,
        "encoded_size": len(encoded),
        "encode_time_s": encode_time,
        "decode_time_s": decode_time,
        "roundtrip_ok": decoded == payload,
        "sha256": hashlib.sha256(decoded).hexdigest(),
    }


def benchmark_game(game_id: str, row_group: int, root: Path, out_dir: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    record = next(item for item in manifest["games"] if item["game_id"] == game_id)
    parquet_path = root / record["activation_file"]

    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    loader = _load_loader_module()
    t0 = time.perf_counter()
    metadata, bits = loader.read_row_group_bits(parquet_path, row_group)
    read_time = time.perf_counter() - t0

    raw_bytes, raw_hash, raw_convert_time = _row_group_bytes_and_hash(bits)
    raw_size = len(raw_bytes)

    baseline = {
        "game_id": game_id,
        "parquet_path": str(parquet_path),
        "row_group": row_group,
        "rows": int(bits.shape[0]),
        "dimensions": int(bits.shape[1]),
        "read_time_s": read_time,
        "read_rows_per_s": bits.shape[0] / read_time if read_time > 0 else float("inf"),
        "raw_size_bytes": raw_size,
        "raw_sha256": raw_hash,
        "raw_bytes_to_numpy_time_s": raw_convert_time,
        "metadata_fields": sorted(metadata.keys()),
    }

    results = [baseline]
    for name in ("raw", "zlib"):
        result = _benchmark_method(name, raw_bytes)
        result["rows"] = int(bits.shape[0])
        result["dimensions"] = int(bits.shape[1])
        results.append(result)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{game_id}_rowgroup_{row_group}_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_id", choices=["high_0056", "low_0056", "high_0066"])
    parser.add_argument("--row-group", type=int, default=0)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("GIBBON_DATA_ROOT", DEFAULT_ROOT)),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "gibbon-work" / "benchmarks",
    )
    args = parser.parse_args()

    results = benchmark_game(args.game_id, args.row_group, args.root, args.out_dir)

    print(f"Benchmark for {args.game_id}, row group {args.row_group}")
    for entry in results:
        if entry.get("method") is not None:
            print(
                f"  {entry['method']:>4}: bytes={entry['encoded_size']:,}, "
                f"encode={entry['encode_time_s']:.6f}s, "
                f"decode={entry['decode_time_s']:.6f}s, "
                f"roundtrip={entry['roundtrip_ok']}"
            )
        else:
            print(
                f"  baseline: rows={entry['rows']}, dims={entry['dimensions']}, "
                f"raw_size={entry['raw_size_bytes']:,} bytes, "
                f"read_time={entry['read_time_s']:.6f}s, "
                f"rows_per_s={entry['read_rows_per_s']:.1f}"
            )

    print(f"Saved benchmark output to {args.out_dir / f'{args.game_id}_rowgroup_{args.row_group}_benchmark.json'}")


if __name__ == "__main__":
    main()
