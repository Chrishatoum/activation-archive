"""Zarr v3-backed archive for exact activation bit patterns.

Reference (uncompressed) and lossless-compressed tiers are both a single
Zarr v3 array (chunked + sharded, optionally Zstandard-coded) plus one
metadata.npz, rather than a hand-rolled directory of per-chunk .npy/.npz
files. This replaced the original per-chunk format after a measured
evaluation (see ZARR_EVALUATION.md) showed Zarr reads 2-2.4x faster at a
~3% storage cost for the same query shapes this project uses. The lossy
archives (python/quantized_archive.py, python/codec_archive.py) keep their
own custom per-chunk format — they were evaluated separately and have
calibration/exception data that doesn't map onto a plain Zarr array.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
import zarr.codecs

from python.load_activations import METADATA_COLUMNS, _binary2_as_uint16


ARCHIVE_FORMAT = "gibbon-activation-archive"
ARCHIVE_VERSION = 3
COMPRESSED_ARCHIVE_VERSION = 4
ZARR_ARRAY_NAME = "activations.zarr"
METADATA_FILE = "metadata.npz"


def _metadata_digest(metadata: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in METADATA_COLUMNS:
        values = np.asarray(metadata[name])
        digest.update(name.encode("utf-8") + b"\0")
        if values.dtype.kind in "OUS":
            for value in values:
                encoded = str(value).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
        else:
            values = np.ascontiguousarray(values)
            digest.update(values.dtype.str.encode("ascii") + b"\0")
            digest.update(values.tobytes())
    return digest.hexdigest()


def _save_metadata(path: Path, metadata: dict[str, np.ndarray]) -> None:
    safe_metadata = {
        name: values.astype(str) if values.dtype.kind in "OUS" else values
        for name, values in metadata.items()
    }
    with path.open("wb") as output:
        np.savez(output, **safe_metadata)


def _index_size(directory: Path, index_path: Path, index: dict[str, Any]) -> int:
    for _ in range(3):
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        size = sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())
        if index.get("archive_size_bytes") == size:
            return size
        index["archive_size_bytes"] = size
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())


def compress_archive(
    source_path: Path,
    destination_path: Path,
    *,
    compression_level: int = 3,
) -> dict[str, Any]:
    """Create an exact Zstandard-compressed copy of an uncompressed archive."""
    source = ActivationArchive(source_path)
    if source.index.get("compression") != "none":
        raise ValueError("Source archive must be uncompressed")
    source.verify()
    destination_path = Path(destination_path)
    if destination_path.exists():
        raise FileExistsError(f"Archive destination already exists: {destination_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.tmp-", dir=destination_path.parent)
    )
    try:
        rows, dimensions = int(source.index["rows"]), int(source.index["dimensions"])
        chunk_rows = int(source.index["chunk_rows_max"])
        array = zarr.create_array(
            store=str(temporary_path / ZARR_ARRAY_NAME),
            shape=(rows, dimensions),
            dtype="<u2",
            chunks=(chunk_rows, dimensions),
            shards=(chunk_rows, dimensions),
            compressors=zarr.codecs.ZstdCodec(level=compression_level),
            zarr_format=3,
        )
        hasher = hashlib.sha256()
        offset = 0
        for _, bits in source.iter_chunks():
            array[offset : offset + bits.shape[0], :] = bits
            hasher.update(np.ascontiguousarray(bits).tobytes())
            offset += bits.shape[0]
        if offset != rows:
            raise ValueError(f"Streamed {offset} rows but the source archive reports {rows}")
        shutil.copyfile(source.path / METADATA_FILE, temporary_path / METADATA_FILE)

        index: dict[str, Any] = {
            **source.index,
            "version": COMPRESSED_ARCHIVE_VERSION,
            "compression": "zstd",
            "compression_level": compression_level,
            "activations_sha256": hasher.hexdigest(),
        }
        index.pop("archive_size_bytes", None)
        index_path = temporary_path / "index.json"
        index["archive_size_bytes"] = _index_size(temporary_path, index_path, index)
        os.replace(temporary_path, destination_path)
        return index
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


class ArchiveWriter:
    """Write fixed-width activation rows into a Zarr v3 array, in bounded-size batches."""

    def __init__(self, path: Path, game_id: str, chunk_rows: int = 4096):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"Archive destination already exists: {self.path}")
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        self.game_id = game_id
        self.chunk_rows = chunk_rows
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{self.path.name}.tmp-", dir=self.path.parent)
        )
        self._array: zarr.Array | None = None
        self._dimensions: int | None = None
        self._metadata_dtypes: dict[str, str] | None = None
        self._metadata_parts: dict[str, list[np.ndarray]] = {name: [] for name in METADATA_COLUMNS}
        self._hasher = hashlib.sha256()
        self._rows = 0
        self._finished = False

    def append(self, metadata: dict[str, np.ndarray], bits: np.ndarray) -> None:
        if self._finished:
            raise RuntimeError("Cannot append after the archive is finalized")
        missing = set(METADATA_COLUMNS) - metadata.keys()
        if missing:
            raise ValueError(f"Missing metadata fields: {sorted(missing)}")
        bits = np.asarray(bits)
        if bits.ndim != 2 or bits.dtype.kind != "u" or bits.dtype.itemsize != 2:
            raise ValueError("Activation data must be a 2D uint16 array of BF16 bit patterns")
        if bits.shape[0] == 0:
            return
        if self._dimensions is None:
            self._dimensions = int(bits.shape[1])
            self._array = zarr.create_array(
                store=str(self._temporary_path / ZARR_ARRAY_NAME),
                shape=(0, self._dimensions),
                dtype="<u2",
                chunks=(self.chunk_rows, self._dimensions),
                shards=(self.chunk_rows, self._dimensions),
                zarr_format=3,
            )
        elif bits.shape[1] != self._dimensions:
            raise ValueError("Activation dimension count changed within the archive")
        normalized_metadata = {name: np.asarray(metadata[name]) for name in METADATA_COLUMNS}
        for name, values in normalized_metadata.items():
            if values.ndim != 1 or len(values) != bits.shape[0]:
                raise ValueError(f"Metadata field {name} must have one value per activation row")
        metadata_dtypes = {name: values.dtype.str for name, values in normalized_metadata.items()}
        if self._metadata_dtypes is None:
            self._metadata_dtypes = metadata_dtypes
        elif metadata_dtypes != self._metadata_dtypes:
            raise ValueError("Metadata dtypes changed within the archive")

        bits = np.ascontiguousarray(bits, dtype="<u2")
        offset = self._rows
        self._array.resize((offset + bits.shape[0], self._dimensions))
        self._array[offset : offset + bits.shape[0], :] = bits
        self._hasher.update(bits.tobytes())
        for name, values in normalized_metadata.items():
            self._metadata_parts[name].append(values)
        self._rows += int(bits.shape[0])

    def finalize(self) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("Archive is already finalized")
        if not self._rows or self._dimensions is None:
            raise ValueError("Cannot finalize an empty archive")
        metadata = {name: np.concatenate(parts) for name, parts in self._metadata_parts.items()}
        _save_metadata(self._temporary_path / METADATA_FILE, metadata)
        index: dict[str, Any] = {
            "format": ARCHIVE_FORMAT,
            "version": ARCHIVE_VERSION,
            "game_id": self.game_id,
            "rows": self._rows,
            "dimensions": self._dimensions,
            "activation_dtype": "little-endian uint16 containing exact BF16 bits",
            "metadata_fields": METADATA_COLUMNS,
            "metadata_dtypes": self._metadata_dtypes,
            "chunk_rows_max": self.chunk_rows,
            "compression": "none",
            "activations_sha256": self._hasher.hexdigest(),
            "metadata_sha256": _metadata_digest(metadata),
        }
        index_path = self._temporary_path / "index.json"
        index["archive_size_bytes"] = _index_size(self._temporary_path, index_path, index)
        os.replace(self._temporary_path, self.path)
        self._finished = True
        return index

    def abort(self) -> None:
        if not self._finished:
            shutil.rmtree(self._temporary_path, ignore_errors=True)
            self._finished = True

    def __enter__(self) -> "ArchiveWriter":
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        if not self._finished:
            self.abort()


class ActivationArchive:
    """Read and query an activation archive backed by a Zarr v3 array."""

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.index = json.loads((self.path / "index.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Archive index is missing or invalid: {self.path}") from error
        if self.index.get("format") != ARCHIVE_FORMAT:
            raise ValueError(f"Unsupported archive format: {self.index.get('format')}")
        if self.index.get("version") not in (ARCHIVE_VERSION, COMPRESSED_ARCHIVE_VERSION):
            raise ValueError(f"Unsupported archive version: {self.index.get('version')}")
        expected_compression = "none" if self.index["version"] == ARCHIVE_VERSION else "zstd"
        if self.index.get("compression") != expected_compression:
            raise ValueError("Archive version and compression codec do not match")
        if self.index.get("metadata_fields") != METADATA_COLUMNS:
            raise ValueError("Archive metadata schema does not match this reader")
        rows, dimensions = int(self.index["rows"]), int(self.index["dimensions"])
        if not rows:
            raise ValueError("Archive contains no rows")
        self._array = zarr.open_array(str(self.path / ZARR_ARRAY_NAME), mode="r")
        if tuple(self._array.shape) != (rows, dimensions) or self._array.dtype.itemsize != 2:
            raise ValueError(f"Invalid activation array in {self.path}")
        with np.load(self.path / METADATA_FILE, allow_pickle=False) as stored:
            self.metadata = {name: stored[name] for name in METADATA_COLUMNS}
        for name, values in self.metadata.items():
            if values.ndim != 1 or len(values) != rows:
                raise ValueError(f"Invalid metadata dimensions in {self.path}")

    def _read_rows(self, start: int, stop: int) -> np.ndarray:
        return np.asarray(self._array[start:stop, :])

    def iter_chunks(
        self,
        *,
        round_index: int | None = None,
        token_start: int | None = None,
        token_stop: int | None = None,
        layer: int | None = None,
        activation_site: str | None = None,
    ):
        """Yield selected metadata and activation rows one bounded-size batch at a time."""
        if (token_start is None) != (token_stop is None):
            raise ValueError("token_start and token_stop must be provided together")
        if token_start is not None and (token_start < 0 or token_stop <= token_start):
            raise ValueError("token range must satisfy 0 <= token_start < token_stop")
        rows = int(self.index["rows"])
        batch_rows = int(self.index["chunk_rows_max"])
        for offset in range(0, rows, batch_rows):
            stop = min(offset + batch_rows, rows)
            metadata = {name: values[offset:stop] for name, values in self.metadata.items()}
            mask = np.ones(stop - offset, dtype=bool)
            if round_index is not None:
                mask &= metadata["round"] == round_index
            if token_start is not None:
                mask &= metadata["token_idx"] >= token_start
                mask &= metadata["token_idx"] < token_stop
            if layer is not None:
                mask &= metadata["layer"] == layer
            if activation_site is not None:
                mask &= metadata["activation_site"] == activation_site
            if not mask.any():
                continue
            bits = self._read_rows(offset, stop)
            selected_metadata = {name: values[mask] for name, values in metadata.items()}
            yield selected_metadata, np.asarray(bits[mask]).copy()

    def read(
        self,
        *,
        round_index: int | None = None,
        token_start: int | None = None,
        token_stop: int | None = None,
        layer: int | None = None,
        activation_site: str | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Collect a query into memory; use iter_chunks for bounded-memory reads."""
        selected_metadata: dict[str, list[np.ndarray]] = {name: [] for name in METADATA_COLUMNS}
        selected_bits: list[np.ndarray] = []
        for metadata, bits in self.iter_chunks(
            round_index=round_index,
            token_start=token_start,
            token_stop=token_stop,
            layer=layer,
            activation_site=activation_site,
        ):
            selected_bits.append(bits)
            for name in METADATA_COLUMNS:
                selected_metadata[name].append(metadata[name])
        result_metadata = {
            name: np.concatenate(values)
            if values
            else np.empty(0, dtype=np.dtype(self.index["metadata_dtypes"][name]))
            for name, values in selected_metadata.items()
        }
        result_bits = (
            np.concatenate(selected_bits, axis=0)
            if selected_bits
            else np.empty((0, int(self.index["dimensions"])), dtype="<u2")
        )
        return result_metadata, result_bits

    def verify(self) -> None:
        rows = int(self.index["rows"])
        batch_rows = int(self.index["chunk_rows_max"])
        hasher = hashlib.sha256()
        total_rows = 0
        for offset in range(0, rows, batch_rows):
            stop = min(offset + batch_rows, rows)
            bits = self._read_rows(offset, stop)
            hasher.update(np.ascontiguousarray(bits).tobytes())
            total_rows += stop - offset
        if hasher.hexdigest() != self.index["activations_sha256"]:
            raise ValueError(f"Activation checksum mismatch in {self.path}")
        if _metadata_digest(self.metadata) != self.index["metadata_sha256"]:
            raise ValueError(f"Metadata checksum mismatch in {self.path}")
        if total_rows != rows:
            raise ValueError("Archive row count does not match its index")


def build_archive_from_parquet(
    parquet_path: Path,
    archive_path: Path,
    game_id: str,
    *,
    batch_rows: int = 1024,
    row_group: int | None = None,
) -> dict[str, Any]:
    """Convert Parquet to the reference archive using bounded record batches."""
    parquet = pq.ParquetFile(parquet_path)
    feature_columns = [name for name in parquet.schema.names if name not in METADATA_COLUMNS]
    if not feature_columns:
        raise ValueError("Parquet file has no activation dimension columns")
    row_groups = list(range(parquet.metadata.num_row_groups)) if row_group is None else [row_group]
    if any(group < 0 or group >= parquet.metadata.num_row_groups for group in row_groups):
        raise ValueError("Requested Parquet row group is out of range")

    with ArchiveWriter(archive_path, game_id, chunk_rows=batch_rows) as writer:
        for batch in parquet.iter_batches(
            batch_size=batch_rows,
            row_groups=row_groups,
            columns=METADATA_COLUMNS + feature_columns,
            use_threads=True,
        ):
            metadata = {
                name: batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
                for name in METADATA_COLUMNS
            }
            bits = np.empty((batch.num_rows, len(feature_columns)), dtype="<u2")
            for index, name in enumerate(feature_columns):
                column = batch.column(batch.schema.get_field_index(name))
                bits[:, index] = _binary2_as_uint16(pa.chunked_array([column]))
            writer.append(metadata, bits)
        return writer.finalize()
