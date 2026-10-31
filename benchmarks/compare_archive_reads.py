#!/usr/bin/env python3
"""Compare Parquet and reference-archive reads on identical row-group queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.data import load_row_group
from benchmarks.common.environment import environment_snapshot, evict_from_page_cache
from python.activation_archive import ActivationArchive
from python.codec_archive import CODECS, CodecActivationArchive
from python.quantized_archive import QuantizedActivationArchive

DEFAULT_ROOT = Path("/srv/gibbon-activations/gpt_oss_pilot_3games")
DEFAULT_ARCHIVE = Path.home() / "gibbon-work" / "archives" / "low_0056-rg0-reference-v1-validation"


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _select_rows(
    metadata: dict[str, np.ndarray],
    bits: np.ndarray,
    query: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    mask = np.ones(len(metadata["round"]), dtype=bool)
    if query["round_index"] is not None:
        mask &= metadata["round"] == query["round_index"]
    if query["token_start"] is not None:
        mask &= metadata["token_idx"] >= query["token_start"]
        mask &= metadata["token_idx"] < query["token_stop"]
    if query["layer"] is not None:
        mask &= metadata["layer"] == query["layer"]
    if query["activation_site"] is not None:
        mask &= metadata["activation_site"] == query["activation_site"]
    return {name: values[mask] for name, values in metadata.items()}, bits[mask].copy()


def _timings(reader, iterations: int) -> list[float]:
    reader()
    values = []
    for _ in range(iterations):
        started = time.perf_counter()
        reader()
        values.append(time.perf_counter() - started)
    return values


def _cold_timings(reader, cache_paths: list[Path], trials: int) -> list[float]:
    """Evict `cache_paths` from the page cache before each of `trials` single reads.

    Best-effort: on a filesystem or platform where posix_fadvise has no effect
    (e.g. some network filesystems), these numbers will look warm; that is a
    limitation of the eviction mechanism, not of the measurement loop.
    """
    values = []
    for _ in range(trials):
        for path in cache_paths:
            evict_from_page_cache(path)
        started = time.perf_counter()
        reader()
        values.append(time.perf_counter() - started)
    return values


def _open_archive(path: Path):
    index = json.loads((path / "index.json").read_text())
    if str(index.get("compression", "")).startswith("quant8_"):
        return QuantizedActivationArchive(path)
    if index.get("compression") in CODECS:
        return CodecActivationArchive(path)
    return ActivationArchive(path)


def _bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def _lossy_error(expected, actual) -> dict[str, float]:
    """Check metadata exactly and return value error for a lossy archive read."""
    expected_metadata, expected_bits = expected
    actual_metadata, actual_values = actual
    for name in expected_metadata:
        if not np.array_equal(expected_metadata[name], actual_metadata[name]):
            raise AssertionError(f"Parquet and archive metadata differ for {name}")
    original = _bits_to_float32(expected_bits).astype(np.float64)
    if original.shape != actual_values.shape:
        raise AssertionError("Parquet and archive activation shapes differ")
    errors = original - actual_values.astype(np.float64)
    return {
        "relative_frobenius_error": float(np.linalg.norm(errors) / (np.linalg.norm(original) + 1e-12)),
        "max_abs_error": float(np.abs(errors).max(initial=0.0)),
    }


def _assert_equal(expected, actual) -> None:
    expected_metadata, expected_bits = expected
    actual_metadata, actual_bits = actual
    if not np.array_equal(expected_bits, actual_bits):
        raise AssertionError("Parquet and archive activation bits differ")
    for name in expected_metadata:
        if not np.array_equal(expected_metadata[name], actual_metadata[name]):
            raise AssertionError(f"Parquet and archive metadata differ for {name}")


def _benchmark_query(
    *,
    game_id: str,
    row_group: int,
    query_name: str,
    query: dict[str, Any],
    source_metadata: dict[str, np.ndarray],
    source_bits: np.ndarray,
    dataset_root: Path,
    parquet_path: Path,
    archive_path: Path,
    parquet_bytes: dict[str, int],
    archive: ActivationArchive | QuantizedActivationArchive | CodecActivationArchive,
    iterations: int,
    cold_trials: int,
) -> dict[str, Any]:
    expected = _select_rows(source_metadata, source_bits, query)

    def parquet_reader():
        return _select_rows(*load_row_group(game_id, row_group, dataset_root), query)

    def archive_reader():
        return archive.read(**query)

    # Verify values and metadata outside the timed loops; lossy archives report their error instead.
    _assert_equal(expected, parquet_reader())
    lossy = isinstance(archive, (QuantizedActivationArchive, CodecActivationArchive))
    if lossy:
        fidelity = _lossy_error(expected, archive_reader())
    else:
        _assert_equal(expected, archive_reader())
        fidelity = {"relative_frobenius_error": 0.0, "max_abs_error": 0.0}

    parquet_cold_times = _cold_timings(parquet_reader, [parquet_path], cold_trials) if cold_trials else []
    archive_cold_times = _cold_timings(archive_reader, [archive_path], cold_trials) if cold_trials else []
    parquet_times = _timings(parquet_reader, iterations)
    archive_times = _timings(archive_reader, iterations)
    parquet_median = statistics.median(parquet_times)
    archive_median = statistics.median(archive_times)
    result = {
        "game_id": game_id,
        "row_group": row_group,
        "query_name": query_name,
        "query": query,
        "rows_in_source_row_group": int(source_bits.shape[0]),
        "rows_returned": int(expected[1].shape[0]),
        "dimensions": int(source_bits.shape[1]),
        "iterations": iterations,
        "cold_trials": cold_trials,
        "timing_mode": (
            "cold: posix_fadvise(DONTNEED) eviction immediately before each single read, one trial per measurement; "
            "warm: one discarded warm-up followed by repeated reads with no deliberate eviction. "
            "Cold eviction is best-effort and may be a no-op on some filesystems (see archive/parquet cold_read_times_s)."
        ),
        "timed_work": "read selected rows and return copied activation values plus metadata",
        "parquet_selection_strategy": "read full Parquet row group with canonical loader, then filter in memory",
        "archive_selection_strategy": "load chunk metadata, memory-map activation chunks, filter, and copy selected rows",
        "roundtrip_exact": not lossy,
        "archive_fidelity": fidelity,
        "selected_activation_sha256": _sha256(expected[1]),
        "parquet": {
            "cold_read_times_s": parquet_cold_times,
            "cold_median_read_time_s": statistics.median(parquet_cold_times) if parquet_cold_times else None,
            "median_read_time_s": parquet_median,
            "read_times_s": parquet_times,
            "rows_per_second_at_median": expected[1].shape[0] / parquet_median if parquet_median else None,
            **parquet_bytes,
        },
        "archive": {
            "path": str(archive_path),
            "archive_size_bytes_including_index_and_metadata": int(archive.index["archive_size_bytes"]),
            "cold_read_times_s": archive_cold_times,
            "cold_median_read_time_s": statistics.median(archive_cold_times) if archive_cold_times else None,
            "median_read_time_s": archive_median,
            "read_times_s": archive_times,
            "rows_per_second_at_median": expected[1].shape[0] / archive_median if archive_median else None,
            "compression": archive.index["compression"],
        },
        "archive_speedup_over_parquet": parquet_median / archive_median if archive_median else None,
        "environment": environment_snapshot(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", default="low_0056")
    parser.add_argument("--row-group", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("GIBBON_DATA_ROOT", DEFAULT_ROOT)))
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--token-start", type=int, default=0)
    parser.add_argument("--token-stop", type=int, default=8)
    parser.add_argument("--layer", type=int, default=11, help="Layer for the single-layer query")
    parser.add_argument(
        "--cold-trials",
        type=int,
        default=3,
        help="Cache-eviction trials per query (0 disables cold-cache measurement)",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.cold_trials < 0:
        parser.error("--cold-trials must be at least 0")
    if args.token_start < 0 or args.token_stop <= args.token_start:
        parser.error("token range must satisfy 0 <= --token-start < --token-stop")

    manifest = json.loads((args.root / "manifest.json").read_text())
    game = next(item for item in manifest["games"] if item["game_id"] == args.game_id)
    parquet_path = args.root / game["activation_file"]
    source_metadata, source_bits = load_row_group(args.game_id, args.row_group, args.root)
    parquet_file = pq.ParquetFile(parquet_path)
    parquet_group = parquet_file.metadata.row_group(args.row_group)
    parquet_bytes = {
        "row_group_compressed_bytes": sum(
            parquet_group.column(i).total_compressed_size for i in range(parquet_group.num_columns)
        ),
        "row_group_uncompressed_bytes": sum(
            parquet_group.column(i).total_uncompressed_size for i in range(parquet_group.num_columns)
        ),
    }
    archive = _open_archive(args.archive)
    archive.verify()
    if archive.index["game_id"] != args.game_id or archive.index["rows"] != source_bits.shape[0]:
        parser.error("archive game/row count does not match the selected Parquet row group")

    queries = [
        ("full_row_group", {"round_index": None, "token_start": None, "token_stop": None, "layer": None, "activation_site": None}),
        ("token_range", {"round_index": args.round_index, "token_start": args.token_start, "token_stop": args.token_stop, "layer": None, "activation_site": None}),
        ("single_layer", {"round_index": None, "token_start": None, "token_stop": None, "layer": args.layer, "activation_site": None}),
        ("layer_token_range", {"round_index": args.round_index, "token_start": args.token_start, "token_stop": args.token_stop, "layer": args.layer, "activation_site": None}),
    ]
    output_dir = args.output_dir or (
        REPO_ROOT / "benchmarks" / "results" / args.game_id / "row_groups" / f"{args.row_group:04d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for query_name, query in queries:
        result = _benchmark_query(
            game_id=args.game_id,
            row_group=args.row_group,
            query_name=query_name,
            query=query,
            source_metadata=source_metadata,
            source_bits=source_bits,
            dataset_root=args.root,
            parquet_path=parquet_path,
            archive_path=args.archive,
            parquet_bytes=parquet_bytes,
            archive=archive,
            iterations=args.iterations,
            cold_trials=args.cold_trials,
        )
        result_path = output_dir / f"parquet_vs_archive_{query_name}.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"result_file: {result_path}")


if __name__ == "__main__":
    main()
