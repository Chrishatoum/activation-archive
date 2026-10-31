#!/usr/bin/env python3
"""Evaluate Zarr v3 indexed sharding against the current NumPy-chunk archive format.

PROJECT_PIPELINE.md Stage 2 calls for this comparison before choosing a final
reference layout. This script builds a Zarr v3 array (chunked + sharded, Zstd
compressed) from an existing uncompressed reference archive's activation bits,
then runs the same four query shapes used throughout this project
(full row group, token range, single layer, layer + token range) with the
same cold/warm protocol as `compare_archive_reads.py`, so the two formats are
compared on equal terms rather than against a different benchmark harness.

This is an evaluation prototype, not a new permanent archive implementation:
per Stage 2's own rule, a second parallel API is only worth building if the
measurement says to switch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.environment import environment_snapshot, evict_from_page_cache
from benchmarks.compare_archive_reads import _select_rows, _sha256, _timings, _cold_timings
from python.activation_archive import ActivationArchive, METADATA_COLUMNS


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build_zarr_array(
    source_path: Path,
    destination_path: Path,
    *,
    chunk_rows: int,
    shard_rows: int,
    zstd_level: int = 3,
) -> dict[str, Any]:
    """Write an existing reference archive's activation bits into a Zarr v3 array.

    Chunk/shard row sizes default to the source archive's own chunk size, so the
    comparison isolates the storage *backend* (Zarr's sharded store vs. our
    hand-rolled per-file npy/zstd chunks + JSON index) rather than re-deciding
    chunk granularity at the same time.
    """
    source = ActivationArchive(source_path)
    if source.index.get("compression") != "none":
        raise ValueError("Zarr evaluation source must be an uncompressed reference archive")
    source.verify()
    destination_path = Path(destination_path)
    if destination_path.exists():
        raise FileExistsError(f"Destination already exists: {destination_path}")
    destination_path.mkdir(parents=True)

    rows, dimensions = int(source.index["rows"]), int(source.index["dimensions"])
    started = time.perf_counter()
    array = zarr.create_array(
        store=str(destination_path / "activations.zarr"),
        shape=(rows, dimensions),
        dtype="uint16",
        chunks=(chunk_rows, dimensions),
        shards=(shard_rows, dimensions),
        compressors=zarr.codecs.ZstdCodec(level=zstd_level),
        zarr_format=3,
    )
    # Stream one archive chunk (bounded by chunk_rows_max) at a time, so building
    # never holds the full row group's activation bits in memory at once, matching
    # the project's "avoid an uncontrolled full-size intermediate copy" rule.
    metadata_parts: dict[str, list[np.ndarray]] = {name: [] for name in METADATA_COLUMNS}
    offset = 0
    hasher = hashlib.sha256()
    for chunk_metadata, chunk_bits in source.iter_chunks():
        batch_rows = chunk_bits.shape[0]
        array[offset : offset + batch_rows, :] = chunk_bits
        for name in METADATA_COLUMNS:
            metadata_parts[name].append(chunk_metadata[name])
        hasher.update(np.ascontiguousarray(chunk_bits).tobytes())
        offset += batch_rows
    if offset != rows:
        raise ValueError(f"Streamed {offset} rows but the source archive reports {rows}")
    write_time = time.perf_counter() - started
    metadata = {name: np.concatenate(parts) for name, parts in metadata_parts.items()}

    metadata_path = destination_path / "metadata.npz"
    with metadata_path.open("wb") as output:
        np.savez(output, **metadata)

    manifest = {
        "format": "zarr_v3_evaluation_prototype",
        "game_id": source.index["game_id"],
        "rows": int(rows),
        "dimensions": int(dimensions),
        "chunk_rows": chunk_rows,
        "shard_rows": shard_rows,
        "zstd_level": zstd_level,
        "write_time_s": write_time,
        "activations_sha256": hasher.hexdigest(),
    }
    (destination_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["archive_size_bytes"] = _directory_bytes(destination_path)
    return manifest


class ZarrEvaluationReader:
    """Minimal reader matching the `.read(**query)` interface the other archives share."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.array = zarr.open_array(str(self.path / "activations.zarr"), mode="r")
        with np.load(self.path / "metadata.npz", allow_pickle=False) as stored:
            self.metadata = {name: stored[name] for name in METADATA_COLUMNS}

    def read(self, **query) -> tuple[dict[str, np.ndarray], np.ndarray]:
        mask = np.ones(len(self.metadata["round"]), dtype=bool)
        if query.get("round_index") is not None:
            mask &= self.metadata["round"] == query["round_index"]
        if query.get("token_start") is not None:
            mask &= self.metadata["token_idx"] >= query["token_start"]
            mask &= self.metadata["token_idx"] < query["token_stop"]
        if query.get("layer") is not None:
            mask &= self.metadata["layer"] == query["layer"]
        if query.get("activation_site") is not None:
            mask &= self.metadata["activation_site"] == query["activation_site"]
        rows = np.flatnonzero(mask)
        bits = self._read_rows(rows)
        selected_metadata = {name: values[mask] for name, values in self.metadata.items()}
        return selected_metadata, bits

    def _read_rows(self, rows: np.ndarray) -> np.ndarray:
        """Use a plain contiguous slice whenever possible; fancy (oindex) indexing otherwise.

        A real reader would make this same distinction — scattered-row fancy
        indexing has materially higher overhead in zarr-python's v3 API than a
        contiguous slice, so always using oindex (as an earlier version of this
        script did) understates Zarr's performance on exactly the full-row-group
        and layer-only queries that are contiguous or near-contiguous in practice.
        """
        if len(rows) == 0:
            return np.empty((0, self.array.shape[1]), dtype=self.array.dtype)
        if rows[-1] - rows[0] + 1 == len(rows):  # contiguous run
            return self.array[rows[0] : rows[-1] + 1, :]
        return self.array.oindex[rows, :]


def benchmark(
    zarr_path: Path,
    reference_path: Path,
    *,
    iterations: int,
    cold_trials: int,
    round_index: int,
    token_start: int,
    token_stop: int,
    layer: int,
) -> list[dict[str, Any]]:
    reference = ActivationArchive(reference_path)
    source_metadata, source_bits = reference.read()
    reader = ZarrEvaluationReader(zarr_path)
    if reader.manifest["game_id"] != reference.index["game_id"] or reader.manifest["rows"] != reference.index["rows"]:
        raise ValueError("Zarr array does not match the reference archive's game/row count")

    queries = [
        ("full_row_group", {"round_index": None, "token_start": None, "token_stop": None, "layer": None, "activation_site": None}),
        ("token_range", {"round_index": round_index, "token_start": token_start, "token_stop": token_stop, "layer": None, "activation_site": None}),
        ("single_layer", {"round_index": None, "token_start": None, "token_stop": None, "layer": layer, "activation_site": None}),
        ("layer_token_range", {"round_index": round_index, "token_start": token_start, "token_stop": token_stop, "layer": layer, "activation_site": None}),
    ]
    results = []
    for query_name, query in queries:
        expected_metadata, expected_bits = _select_rows(source_metadata, source_bits, query)

        def reader_fn():
            return reader.read(**query)

        metadata_out, bits_out = reader_fn()
        if not np.array_equal(bits_out, expected_bits):
            raise AssertionError(f"Zarr read does not match the reference archive for query {query_name}")
        for name in METADATA_COLUMNS:
            if not np.array_equal(metadata_out[name], expected_metadata[name]):
                raise AssertionError(f"Metadata mismatch for {name} in query {query_name}")

        cold_times = _cold_timings(reader_fn, [zarr_path], cold_trials) if cold_trials else []
        warm_times = _timings(reader_fn, iterations)
        results.append({
            "query_name": query_name,
            "query": query,
            "rows_returned": int(expected_bits.shape[0]),
            "roundtrip_exact": True,
            "cold_read_times_s": cold_times,
            "cold_median_read_time_s": statistics.median(cold_times) if cold_times else None,
            "median_read_time_s": statistics.median(warm_times),
            "read_times_s": warm_times,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Existing uncompressed reference archive to source bits from")
    parser.add_argument("zarr_destination", type=Path, help="New Zarr v3 evaluation directory")
    parser.add_argument("--chunk-rows", type=int, default=256, help="Zarr chunk size in rows (codec granularity)")
    parser.add_argument("--shard-rows", type=int, default=1024, help="Zarr shard size in rows (file granularity)")
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--cold-trials", type=int, default=3)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--token-start", type=int, default=0)
    parser.add_argument("--token-stop", type=int, default=8)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-zarr", action="store_true", help="Don't delete the Zarr directory after benchmarking")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the array and exit, skipping the read benchmark (for a clean build-only peak-RSS measurement)",
    )
    args = parser.parse_args()

    build_info = build_zarr_array(
        args.reference,
        args.zarr_destination,
        chunk_rows=args.chunk_rows,
        shard_rows=args.shard_rows,
    )
    if args.build_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"build": build_info}, indent=2) + "\n")
        print(json.dumps(build_info, indent=2))
        if not args.keep_zarr:
            shutil.rmtree(args.zarr_destination, ignore_errors=True)
        return
    try:
        queries = benchmark(
            args.zarr_destination,
            args.reference,
            iterations=args.iterations,
            cold_trials=args.cold_trials,
            round_index=args.round_index,
            token_start=args.token_start,
            token_stop=args.token_stop,
            layer=args.layer,
        )
    finally:
        if not args.keep_zarr:
            shutil.rmtree(args.zarr_destination, ignore_errors=True)

    result = {
        "reference_archive": str(args.reference),
        "build": build_info,
        "queries": queries,
        "environment": environment_snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"build": build_info, "queries": [{k: v for k, v in q.items() if k != "read_times_s"} for q in queries]}, indent=2))
    print(f"result_file: {args.output}")


if __name__ == "__main__":
    main()
