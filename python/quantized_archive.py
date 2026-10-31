"""Affine Quant8 archive for pilot activation chunks.

Scales are global, per (layer, activation_site) group, or per dimension within
each (layer, activation_site) group. They are fitted either on the source
archive itself (in-sample) or on separate calibration archives (held-out).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from python.activation_archive import (
    ActivationArchive,
    METADATA_COLUMNS,
    _metadata_digest,
    _save_metadata,
)


QUANT8_ARCHIVE_VERSION = 3
QUANT8_FORMAT = "gibbon-activation-archive"
QUANT8_CODECS = {
    "global": "quant8_global_affine",
    "layer_site": "quant8_layer_site_affine",
    "layer_site_dim": "quant8_layer_site_dim_affine",
}
PARAMETERS_FILE = "parameters.npz"
EXCEPTIONS_FILE = "exceptions.npz"

Groups = dict[str, dict[str, np.ndarray]]


def _group_key(granularity: str, layer: Any, activation_site: Any) -> str:
    return "all" if granularity == "global" else f"{int(layer)}:{activation_site}"


def _group_rows(granularity: str, metadata: dict[str, np.ndarray]) -> Iterator[tuple[str, np.ndarray]]:
    """Yield each parameter group's key and row indices within a chunk."""
    rows = len(metadata["layer"])
    if granularity == "global":
        yield "all", np.arange(rows)
        return
    layers, layer_inverse = np.unique(metadata["layer"], return_inverse=True)
    sites, site_inverse = np.unique(metadata["activation_site"], return_inverse=True)
    combined = layer_inverse * len(sites) + site_inverse
    for code in np.unique(combined):
        layer, site = layers[code // len(sites)], sites[code % len(sites)]
        yield _group_key(granularity, layer, site), np.flatnonzero(combined == code)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _index_size(directory: Path, index_path: Path, index: dict[str, Any]) -> int:
    for _ in range(3):
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        size = sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())
        if index.get("archive_size_bytes") == size:
            return size
        index["archive_size_bytes"] = size
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())


def fit_quant8_groups(archive_paths: Sequence[Path], granularity: str) -> Groups:
    """Fit affine minimum/scale parameters from the value ranges of reference archives.

    Per-dimension parameters are rounded to float32, the dtype they are stored in,
    so encoding and decoding use identical values.
    """
    if granularity not in QUANT8_CODECS:
        raise ValueError(f"Unsupported Quant8 granularity: {granularity}")
    per_dimension = granularity == "layer_site_dim"
    ranges: dict[str, list[np.ndarray]] = {}
    for path in archive_paths:
        archive = ActivationArchive(path)
        if archive.index.get("compression") != "none":
            raise ValueError("Quant8 calibration archives must be uncompressed reference archives")
        for metadata, bits in archive.iter_chunks():
            values = _bf16_bits_to_float32(bits)
            if not np.isfinite(values).all():
                raise ValueError("Quant8 requires finite activation values")
            for key, rows in _group_rows(granularity, metadata):
                selected = values[rows]
                low = selected.min(axis=0) if per_dimension else np.array([selected.min()])
                high = selected.max(axis=0) if per_dimension else np.array([selected.max()])
                bounds = ranges.setdefault(key, [low.astype(np.float64), high.astype(np.float64)])
                np.minimum(bounds[0], low, out=bounds[0])
                np.maximum(bounds[1], high, out=bounds[1])
    groups: Groups = {}
    for key, (low, high) in sorted(ranges.items()):
        scale = np.where(high > low, (high - low) / 255.0, 1.0)
        if per_dimension:
            low, scale = low.astype(np.float32).astype(np.float64), scale.astype(np.float32).astype(np.float64)
        groups[key] = {"minimum": low, "scale": scale}
    return groups


def _save_parameters(path: Path, groups: Groups) -> None:
    keys = sorted(groups)
    with path.open("wb") as output:
        np.savez(
            output,
            keys=np.array(keys),
            minimum=np.stack([groups[key]["minimum"] for key in keys]).astype(np.float32),
            scale=np.stack([groups[key]["scale"] for key in keys]).astype(np.float32),
        )


def _load_parameters(path: Path) -> Groups:
    with np.load(path, allow_pickle=False) as stored:
        return {
            str(key): {"minimum": stored["minimum"][i].astype(np.float64), "scale": stored["scale"][i].astype(np.float64)}
            for i, key in enumerate(stored["keys"])
        }


def compress_archive_quant8(
    source_path: Path,
    destination_path: Path,
    *,
    chunk_rows: int | None = None,
    granularity: str = "global",
    calibration_paths: Sequence[Path] | None = None,
    store_clipped_exceptions: bool = False,
) -> dict[str, Any]:
    """Quantize an uncompressed reference archive to affine uint8 codes.

    Without ``calibration_paths`` the parameters are fitted on the source itself
    (in-sample). With them, parameters are fitted only on those archives, which
    must come from other games; source values outside the fitted range are clipped.

    With ``store_clipped_exceptions``, every clipped (row, dimension) also has its
    exact original BF16 bit pattern stored in a per-chunk sparse exception table.
    The decoder restores those exact values after dequantizing the clipped uint8
    code, so held-out clipping no longer sets the archive's max absolute error or
    distorts the variance/cosine metrics of the dimensions it falls in.
    """
    if granularity not in QUANT8_CODECS:
        raise ValueError(f"Unsupported Quant8 granularity: {granularity}")
    source = ActivationArchive(source_path)
    if source.index.get("compression") != "none":
        raise ValueError("Quant8 source must be an uncompressed reference archive")
    source.verify()
    destination_path = Path(destination_path)
    if destination_path.exists():
        raise FileExistsError(f"Archive destination already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if calibration_paths:
        calibration_games = sorted({ActivationArchive(path).index["game_id"] for path in calibration_paths})
        if source.index["game_id"] in calibration_games:
            raise ValueError("Held-out calibration archives must not include the source game")
        groups = fit_quant8_groups(calibration_paths, granularity)
        calibration = {
            "mode": "held_out",
            "game_ids": calibration_games,
            "archives": [str(Path(path)) for path in calibration_paths],
        }
    else:
        groups = fit_quant8_groups([source_path], granularity)
        calibration = {"mode": "in_sample", "game_ids": [source.index["game_id"]], "archives": [str(source_path)]}

    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.tmp-", dir=destination_path.parent)
    )
    chunks: list[dict[str, Any]] = []
    squared_error_sum = 0.0
    squared_value_sum = 0.0
    max_abs_error = 0.0
    clipped_values = 0
    exception_bytes_total = 0
    encode_started = time.perf_counter()
    try:
        for chunk_index, (metadata, source_bits) in enumerate(source.iter_chunks()):
            chunk_name = f"chunk-{chunk_index:08d}"
            values = _bf16_bits_to_float32(source_bits).astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Quant8 requires finite activation values")
            codes = np.empty(values.shape, dtype=np.uint8)
            exception_rows: list[np.ndarray] = []
            exception_dims: list[np.ndarray] = []
            exception_bits: list[np.ndarray] = []
            for key, rows in _group_rows(granularity, metadata):
                if key not in groups:
                    raise ValueError(f"Calibration data has no parameters for group {key}")
                raw_codes = np.rint((values[rows] - groups[key]["minimum"]) / groups[key]["scale"])
                clipped_mask = (raw_codes < 0) | (raw_codes > 255)
                clipped_values += int(np.count_nonzero(clipped_mask))
                if store_clipped_exceptions and clipped_mask.any():
                    local_rows, dims = np.nonzero(clipped_mask)
                    exception_rows.append(rows[local_rows].astype(np.uint32))
                    exception_dims.append(dims.astype(np.uint16))
                    exception_bits.append(source_bits[rows[local_rows], dims])
                group_codes = raw_codes.clip(0, 255)
                codes[rows] = group_codes
                errors = values[rows] - (group_codes * groups[key]["scale"] + groups[key]["minimum"])
                if store_clipped_exceptions:
                    # Exceptions restore the exact value on decode, so they contribute no error.
                    errors = np.where(clipped_mask, 0.0, errors)
                squared_error_sum += float(np.square(errors).sum())
                max_abs_error = max(max_abs_error, float(np.abs(errors).max(initial=0.0)))
            squared_value_sum += float(np.square(values).sum())

            chunk_path = temporary_path / chunk_name
            chunk_path.mkdir()
            code_path = chunk_path / "codes.npy"
            with code_path.open("wb") as output:
                np.save(output, codes, allow_pickle=False)
            metadata_file_name = "metadata.npz"
            output_metadata_path = chunk_path / metadata_file_name
            _save_metadata(output_metadata_path, metadata)
            chunk_entry: dict[str, Any] = {
                "chunk": chunk_name,
                "rows": int(codes.shape[0]),
                "activation_file": code_path.name,
                "codes_sha256": _digest(codes),
                "metadata_file": metadata_file_name,
                "metadata_sha256": _metadata_digest(metadata),
            }
            stored_bytes = code_path.stat().st_size + output_metadata_path.stat().st_size
            if exception_rows:
                exception_path = chunk_path / EXCEPTIONS_FILE
                order = np.argsort(np.concatenate(exception_rows))
                exc_rows = np.concatenate(exception_rows)[order]
                exc_dims = np.concatenate(exception_dims)[order]
                exc_bits = np.concatenate(exception_bits)[order]
                with exception_path.open("wb") as output:
                    np.savez(output, rows=exc_rows, dims=exc_dims, bits=exc_bits)
                exception_size = exception_path.stat().st_size
                exception_bytes_total += exception_size
                stored_bytes += exception_size
                chunk_entry["exceptions_file"] = EXCEPTIONS_FILE
                chunk_entry["exception_count"] = int(exc_rows.shape[0])
                chunk_entry["exceptions_sha256"] = hashlib.sha256(exception_path.read_bytes()).hexdigest()
            chunk_entry["stored_bytes"] = stored_bytes
            chunks.append(chunk_entry)

        encode_time = time.perf_counter() - encode_started
        index: dict[str, Any] = {
            "format": QUANT8_FORMAT,
            "version": QUANT8_ARCHIVE_VERSION,
            "game_id": source.index["game_id"],
            "rows": int(source.index["rows"]),
            "dimensions": int(source.index["dimensions"]),
            "source_activation_dtype": "bfloat16",
            "encoded_activation_dtype": "uint8",
            "metadata_fields": METADATA_COLUMNS,
            "metadata_dtypes": source.index["metadata_dtypes"],
            "compression": QUANT8_CODECS[granularity],
            "scale_granularity": granularity,
            "calibration": calibration,
            "chunk_rows_max": chunk_rows or source.index["chunk_rows_max"],
            "relative_frobenius_error": float(
                np.sqrt(squared_error_sum) / (np.sqrt(squared_value_sum) + 1e-12)
            ),
            "max_abs_error": max_abs_error,
            "clipped_values": clipped_values,
            "clipped_fraction": clipped_values / max(1, int(source.index["rows"]) * int(source.index["dimensions"])),
            "clipped_exceptions_stored": bool(store_clipped_exceptions),
            "exception_bytes_total": exception_bytes_total,
            "encode_time_s": encode_time,
            "chunks": chunks,
        }
        if granularity == "layer_site_dim":
            parameters_path = temporary_path / PARAMETERS_FILE
            _save_parameters(parameters_path, groups)
            index["parameters_file"] = PARAMETERS_FILE
            index["parameters_sha256"] = hashlib.sha256(parameters_path.read_bytes()).hexdigest()
            index["parameter_bytes"] = parameters_path.stat().st_size
        else:
            index["groups"] = {
                key: {"minimum": float(group["minimum"][0]), "scale": float(group["scale"][0])}
                for key, group in groups.items()
            }
            index["parameter_bytes"] = 16 * len(groups)
        index_path = temporary_path / "index.json"
        index["archive_size_bytes"] = _index_size(temporary_path, index_path, index)
        os.replace(temporary_path, destination_path)
        return index
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


class QuantizedActivationArchive:
    """Reader for the lossy affine Quant8 archive version."""

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.index = json.loads((self.path / "index.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Archive index is missing or invalid: {self.path}") from error
        if self.index.get("format") != QUANT8_FORMAT or self.index.get("version") != QUANT8_ARCHIVE_VERSION:
            raise ValueError("Unsupported Quant8 archive format or version")
        if self.index.get("compression") not in QUANT8_CODECS.values():
            raise ValueError("Unsupported Quant8 archive codec")
        self.granularity = self.index.get("scale_granularity", "global")
        if "parameters_file" in self.index:
            groups = _load_parameters(self.path / self.index["parameters_file"])
        else:
            # Archives written before per-group scales store one top-level minimum and scale.
            stored_groups = self.index.get("groups") or {
                "all": {"minimum": self.index["minimum"], "scale": self.index["scale"]}
            }
            groups = {
                key: {"minimum": np.array([group["minimum"]]), "scale": np.array([group["scale"]])}
                for key, group in stored_groups.items()
            }
        self.groups = {
            key: {name: values.astype(np.float32) for name, values in group.items()}
            for key, group in groups.items()
        }

    def iter_chunks(self, **query):
        token_start = query.get("token_start")
        token_stop = query.get("token_stop")
        if (token_start is None) != (token_stop is None):
            raise ValueError("token_start and token_stop must be provided together")
        if token_start is not None and (token_start < 0 or token_stop <= token_start):
            raise ValueError("token range must satisfy 0 <= token_start < token_stop")
        for chunk in self.index["chunks"]:
            chunk_path = self.path / chunk["chunk"]
            with np.load(chunk_path / chunk["metadata_file"], allow_pickle=False) as stored:
                metadata = {name: stored[name] for name in METADATA_COLUMNS}
            mask = np.ones(int(chunk["rows"]), dtype=bool)
            if query.get("round_index") is not None:
                mask &= metadata["round"] == query["round_index"]
            if token_start is not None:
                mask &= metadata["token_idx"] >= token_start
                mask &= metadata["token_idx"] < token_stop
            if query.get("layer") is not None:
                mask &= metadata["layer"] == query["layer"]
            if query.get("activation_site") is not None:
                mask &= metadata["activation_site"] == query["activation_site"]
            if not mask.any():
                continue
            codes = np.load(chunk_path / chunk["activation_file"], mmap_mode="r", allow_pickle=False)
            selected_metadata = {name: values_[mask] for name, values_ in metadata.items()}
            values = codes[mask].astype(np.float32)
            for key, rows in _group_rows(self.granularity, selected_metadata):
                group = self.groups[key]
                values[rows] = values[rows] * group["scale"] + group["minimum"]
            if "exceptions_file" in chunk:
                with np.load(chunk_path / chunk["exceptions_file"], allow_pickle=False) as stored:
                    exc_rows, exc_dims, exc_bits = stored["rows"], stored["dims"], stored["bits"]
                keep = mask[exc_rows]
                if keep.any():
                    positions = np.cumsum(mask) - 1
                    values[positions[exc_rows[keep]], exc_dims[keep]] = _bf16_bits_to_float32(exc_bits[keep])
            yield selected_metadata, values

    def read(self, **query) -> tuple[dict[str, np.ndarray], np.ndarray]:
        metadata_chunks: dict[str, list[np.ndarray]] = {name: [] for name in METADATA_COLUMNS}
        value_chunks = []
        for metadata, values in self.iter_chunks(**query):
            value_chunks.append(values)
            for name in METADATA_COLUMNS:
                metadata_chunks[name].append(metadata[name])
        metadata = {
            name: np.concatenate(arrays) if arrays else np.empty(0, dtype=np.dtype(self.index["metadata_dtypes"][name]))
            for name, arrays in metadata_chunks.items()
        }
        values = np.concatenate(value_chunks, axis=0) if value_chunks else np.empty((0, self.index["dimensions"]), dtype=np.float32)
        return metadata, values

    def verify(self) -> None:
        if "parameters_file" in self.index:
            digest = hashlib.sha256((self.path / self.index["parameters_file"]).read_bytes()).hexdigest()
            if digest != self.index["parameters_sha256"]:
                raise ValueError(f"Quant8 parameter checksum mismatch in {self.path}")
        row_count = 0
        for chunk in self.index["chunks"]:
            chunk_path = self.path / chunk["chunk"]
            codes = np.load(chunk_path / chunk["activation_file"], mmap_mode="r", allow_pickle=False)
            with np.load(chunk_path / chunk["metadata_file"], allow_pickle=False) as stored:
                metadata = {name: stored[name] for name in METADATA_COLUMNS}
            if codes.shape != (int(chunk["rows"]), int(self.index["dimensions"])) or codes.dtype != np.uint8:
                raise ValueError(f"Invalid Quant8 code array in {chunk_path}")
            if _digest(codes) != chunk["codes_sha256"]:
                raise ValueError(f"Quant8 code checksum mismatch in {chunk_path}")
            if _metadata_digest(metadata) != chunk["metadata_sha256"]:
                raise ValueError(f"Metadata checksum mismatch in {chunk_path}")
            if "exceptions_file" in chunk:
                exceptions_path = chunk_path / chunk["exceptions_file"]
                if hashlib.sha256(exceptions_path.read_bytes()).hexdigest() != chunk["exceptions_sha256"]:
                    raise ValueError(f"Quant8 exceptions checksum mismatch in {chunk_path}")
                with np.load(exceptions_path, allow_pickle=False) as stored:
                    if stored["rows"].shape[0] != int(chunk["exception_count"]):
                        raise ValueError(f"Quant8 exception count mismatch in {chunk_path}")
            row_count += int(chunk["rows"])
        if row_count != int(self.index["rows"]):
            raise ValueError("Quant8 archive row count does not match its index")
