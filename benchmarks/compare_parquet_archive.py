#!/usr/bin/env python3
"""Compare reading one Parquet row group with an uncompressed archive chunk."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.data import load_row_group
from python.load_activations import METADATA_COLUMNS

DEFAULT_ROOT = Path("/srv/gibbon-activations/gpt_oss_pilot_3games")
ARCHIVE_FORMAT = "gibbon-raw-reference-chunk"
ARCHIVE_VERSION = 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _metadata_digest(metadata: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in METADATA_COLUMNS:
        array = metadata[name]
        digest.update(name.encode("utf-8") + b"\0")
        if array.dtype.kind in "OUS":
            for value in array:
                encoded = str(value).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
        else:
            contiguous = np.ascontiguousarray(array)
            digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _archive_chunk_path(root: Path, game_id: str, row_group: int) -> Path:
    return root / game_id / "row_groups" / f"{row_group:04d}"


def _write_archive_chunk(
    chunk_path: Path,
    metadata: dict[str, np.ndarray],
    bits: np.ndarray,
    game_id: str,
    row_group: int,
) -> dict[str, Any]:
    chunk_path.mkdir(parents=True, exist_ok=True)
    activation_path = chunk_path / "activations.npy"
    metadata_path = chunk_path / "metadata.npz"
    np.save(activation_path, bits, allow_pickle=False)
    safe_metadata = {
        name: values.astype(str) if values.dtype.kind in "OUS" else values
        for name, values in metadata.items()
    }
    np.savez(metadata_path, **safe_metadata)

    index = {
        "format": ARCHIVE_FORMAT,
        "version": ARCHIVE_VERSION,
        "game_id": game_id,
        "row_group": row_group,
        "rows": int(bits.shape[0]),
        "dimensions": int(bits.shape[1]),
        "activation_file": activation_path.name,
        "activation_dtype": "little-endian uint16 containing exact BF16 bits",
        "activation_shape": list(bits.shape),
        "activation_sha256": _sha256(bits.tobytes()),
        "metadata_file": metadata_path.name,
        "metadata_fields": METADATA_COLUMNS,
        "metadata_sha256": _metadata_digest(metadata),
        "compression": "none",
    }
    (chunk_path / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    index["archive_size_bytes"] = sum(
        (chunk_path / filename).stat().st_size
        for filename in (activation_path.name, metadata_path.name, "index.json")
    )
    return index


def _select_rows(
    metadata: dict[str, np.ndarray],
    bits: np.ndarray,
    *,
    round_index: int | None,
    token_start: int | None,
    token_stop: int | None,
    layer: int | None,
    activation_site: str | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    mask = np.ones(len(metadata["round"]), dtype=bool)
    if round_index is not None:
        mask &= metadata["round"] == round_index
    if token_start is not None:
        mask &= metadata["token_idx"] >= token_start
    if token_stop is not None:
        mask &= metadata["token_idx"] < token_stop
    if layer is not None:
        mask &= metadata["layer"] == layer
    if activation_site is not None:
        mask &= metadata["activation_site"] == activation_site
    selected_metadata = {name: values[mask] for name, values in metadata.items()}
    return selected_metadata, np.asarray(bits[mask]).copy()


def _read_archive_chunk(
    chunk_path: Path,
    *,
    round_index: int | None = None,
    token_start: int | None = None,
    token_stop: int | None = None,
    layer: int | None = None,
    activation_site: str | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    index = json.loads((chunk_path / "index.json").read_text())
    if index.get("format") != ARCHIVE_FORMAT or index.get("version") != ARCHIVE_VERSION:
        raise ValueError(f"Unsupported archive format in {chunk_path}")
    with np.load(chunk_path / index["metadata_file"], allow_pickle=False) as metadata_file:
        metadata = {name: metadata_file[name] for name in index["metadata_fields"]}
    bits = np.load(chunk_path / index["activation_file"], mmap_mode="r", allow_pickle=False)
    return _select_rows(
        metadata,
        bits,
        round_index=round_index,
        token_start=token_start,
        token_stop=token_stop,
        layer=layer,
        activation_site=activation_site,
    )


def _assert_equal(
    expected_metadata: dict[str, np.ndarray],
    expected_bits: np.ndarray,
    actual_metadata: dict[str, np.ndarray],
    actual_bits: np.ndarray,
) -> None:
    if not np.array_equal(expected_bits, actual_bits):
        raise AssertionError("Activation BF16 bit patterns changed in archive round trip")
    for name in METADATA_COLUMNS:
        if not np.array_equal(expected_metadata[name], actual_metadata[name]):
            raise AssertionError(f"Metadata field changed in archive round trip: {name}")


def _time_reads(reader, iterations: int) -> list[float]:
    reader()  # Warm the filesystem cache before collecting repeated read timings.
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        reader()
        timings.append(time.perf_counter() - started)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", default="low_0056")
    parser.add_argument("--row-group", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("GIBBON_DATA_ROOT", DEFAULT_ROOT)))
    parser.add_argument("--archive-root", type=Path, default=Path.home() / "gibbon-work" / "archives" / "raw_reference_v1")
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--round-index", type=int)
    parser.add_argument("--token-start", type=int)
    parser.add_argument("--token-stop", type=int)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--activation-site", choices=("post_attention", "post_mlp"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if (args.token_start is None) != (args.token_stop is None):
        parser.error("--token-start and --token-stop must be provided together")
    if args.token_start is not None:
        if args.round_index is None:
            parser.error("token_idx restarts each round; --round-index is required for a token range")
        if args.token_start < 0 or args.token_stop <= args.token_start:
            parser.error("token range must satisfy 0 <= --token-start < --token-stop")

    source_manifest = json.loads((args.root / "manifest.json").read_text())
    record = next(item for item in source_manifest["games"] if item["game_id"] == args.game_id)
    parquet_path = args.root / record["activation_file"]
    source_metadata, source_bits = load_row_group(args.game_id, args.row_group, args.root)

    chunk_path = _archive_chunk_path(args.archive_root, args.game_id, args.row_group)
    archive_index = _write_archive_chunk(
        chunk_path, source_metadata, source_bits, args.game_id, args.row_group
    )

    selection = {
        "round_index": args.round_index,
        "token_start": args.token_start,
        "token_stop": args.token_stop,
        "layer": args.layer,
        "activation_site": args.activation_site,
    }

    def parquet_reader():
        metadata, bits = load_row_group(args.game_id, args.row_group, args.root)
        return _select_rows(metadata, bits, **selection)

    def archive_reader():
        return _read_archive_chunk(chunk_path, **selection)

    # Verify values and labels before timing either implementation.
    archive_metadata, archive_bits = archive_reader()
    expected_metadata, expected_bits = _select_rows(source_metadata, source_bits, **selection)
    _assert_equal(expected_metadata, expected_bits, archive_metadata, archive_bits)
    parquet_metadata, parquet_bits = parquet_reader()
    _assert_equal(expected_metadata, expected_bits, parquet_metadata, parquet_bits)

    parquet_times = _time_reads(parquet_reader, args.iterations)
    archive_times = _time_reads(archive_reader, args.iterations)
    parquet_file = pq.ParquetFile(parquet_path)
    parquet_group = parquet_file.metadata.row_group(args.row_group)
    parquet_compressed_size = sum(
        parquet_group.column(index).total_compressed_size
        for index in range(parquet_group.num_columns)
    )
    parquet_uncompressed_size = sum(
        parquet_group.column(index).total_uncompressed_size
        for index in range(parquet_group.num_columns)
    )

    parquet_median = statistics.median(parquet_times)
    archive_median = statistics.median(archive_times)
    result = {
        "game_id": args.game_id,
        "row_group": args.row_group,
        "rows_in_source_row_group": int(source_bits.shape[0]),
        "rows_returned": int(expected_bits.shape[0]),
        "dimensions": int(source_bits.shape[1]),
        "query": selection,
        "iterations": args.iterations,
        "timing_mode": "repeated reads after one warm-up read; filesystem cache state not controlled",
        "parquet_selection_strategy": "read full row group with canonical loader, then filter in memory",
        "archive_selection_strategy": "read metadata for selection, then copy matching rows from a memory-mapped activation array",
        "roundtrip_exact": True,
        "activation_sha256": archive_index["activation_sha256"],
        "metadata_sha256": archive_index["metadata_sha256"],
        "parquet": {
            "median_read_time_s": parquet_median,
            "read_times_s": parquet_times,
            "rows_per_second_at_median": expected_bits.shape[0] / parquet_median,
            "row_group_compressed_bytes": int(parquet_compressed_size),
            "row_group_uncompressed_bytes": int(parquet_uncompressed_size),
        },
        "raw_archive": {
            "path": str(chunk_path),
            "median_read_time_s": archive_median,
            "read_times_s": archive_times,
            "rows_per_second_at_median": expected_bits.shape[0] / archive_median,
            "activation_bytes_returned": int(expected_bits.nbytes),
            "archive_size_bytes_including_labels_and_index": archive_index["archive_size_bytes"],
            "compression": "none",
        },
        "archive_speedup_over_parquet": parquet_median / archive_median,
    }
    output_path = args.output or (
        REPO_ROOT
        / "benchmarks"
        / "results"
        / args.game_id
        / "row_groups"
        / f"{args.row_group:04d}"
        / "parquet_vs_archive_token_range.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"result_file: {output_path}")


if __name__ == "__main__":
    main()