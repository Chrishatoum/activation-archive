"""Lossy codec archives: reduced precision, FP8, zfp, and SZ3 over pilot activation chunks.

Each archive chunk is split into blocks, one per (layer, activation_site) group,
because value ranges differ by orders of magnitude between layers. Each block is
encoded independently, and its per-block parameters (scale, error bound) and byte
range are stored in the index. A query decodes only the blocks it touches.
No parameters are fitted on other data, so no calibration split is needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zstandard

from python.activation_archive import (
    ActivationArchive,
    METADATA_COLUMNS,
    _metadata_digest,
    _save_metadata,
)
from python.quantized_archive import _bf16_bits_to_float32, _group_rows, _index_size


CODEC_ARCHIVE_VERSION = 4
CODEC_FORMAT = "gibbon-activation-archive"
PAYLOAD_FILE = "payload.bin"
FP8_E4M3_MAX = 448.0


def _truncate_bf16(bits: np.ndarray, mantissa_bits: int) -> np.ndarray:
    """Round BF16 bit patterns to ``mantissa_bits`` of the 7 mantissa bits (nearest, ties to even)."""
    drop = 7 - mantissa_bits
    if drop == 0:
        return bits.copy()
    wide = bits.astype(np.uint32)
    half = np.uint32(1 << (drop - 1))
    lsb = (wide >> np.uint32(drop)) & np.uint32(1)
    rounded = (wide + half - np.uint32(1) + lsb) >> np.uint32(drop) << np.uint32(drop)
    return rounded.astype(np.uint16)


def _encode_bf16_trunc(values: np.ndarray, bits: np.ndarray, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    truncated = _truncate_bf16(bits, int(params["mantissa_bits"]))
    payload = zstandard.ZstdCompressor(level=int(params.get("zstd_level", 3))).compress(truncated.tobytes())
    return payload, {}


def _decode_bf16_trunc(payload: bytes, shape: tuple[int, int], block: dict[str, Any]) -> np.ndarray:
    raw = zstandard.ZstdDecompressor().decompress(payload)
    return _bf16_bits_to_float32(np.frombuffer(raw, dtype="<u2").reshape(shape))


def _encode_fp8_e4m3(values: np.ndarray, bits: np.ndarray, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    import torch

    absmax = float(np.abs(values).max(initial=0.0))
    scale = absmax / FP8_E4M3_MAX if absmax > 0 else 1.0
    encoded = torch.from_numpy(values / np.float32(scale)).to(torch.float8_e4m3fn)
    return encoded.view(torch.uint8).numpy().tobytes(), {"scale": scale}


def _decode_fp8_e4m3(payload: bytes, shape: tuple[int, int], block: dict[str, Any]) -> np.ndarray:
    import torch

    codes = torch.from_numpy(np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy())
    return codes.view(torch.float8_e4m3fn).to(torch.float32).numpy() * np.float32(block["scale"])


def _encode_zfp_rate(values: np.ndarray, bits: np.ndarray, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    import zfpy

    return zfpy.compress_numpy(np.ascontiguousarray(values), rate=float(params["rate"])), {}


def _decode_zfp_rate(payload: bytes, shape: tuple[int, int], block: dict[str, Any]) -> np.ndarray:
    import zfpy

    return zfpy.decompress_numpy(payload).reshape(shape)


def _encode_sz3_abs(values: np.ndarray, bits: np.ndarray, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    import pysz

    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    bound = float(params["tolerance"]) * rms if rms > 0 else float(params["tolerance"])
    config = pysz.szConfig()
    config.errorBoundMode = pysz.szErrorBoundMode.ABS
    config.absErrorBound = bound
    compressed, _ = pysz.sz.compress(np.ascontiguousarray(values), config)
    return compressed.tobytes(), {"abs_error_bound": bound}


def _decode_sz3_abs(payload: bytes, shape: tuple[int, int], block: dict[str, Any]) -> np.ndarray:
    import pysz

    decoded, _ = pysz.sz.decompress(np.frombuffer(payload, dtype=np.uint8), np.float32, shape)
    return decoded


CODECS: dict[str, tuple[Callable, Callable, set[str]]] = {
    "bf16_trunc_zstd": (_encode_bf16_trunc, _decode_bf16_trunc, {"mantissa_bits", "zstd_level"}),
    "fp8_e4m3_block_scale": (_encode_fp8_e4m3, _decode_fp8_e4m3, set()),
    "zfp_fixed_rate": (_encode_zfp_rate, _decode_zfp_rate, {"rate"}),
    "sz3_abs_rms": (_encode_sz3_abs, _decode_sz3_abs, {"tolerance"}),
}


def _codec_version(codec: str) -> str:
    if codec == "zfp_fixed_rate":
        import zfpy

        return f"zfpy {getattr(zfpy, '__version__', 'unknown')}"
    if codec == "sz3_abs_rms":
        from importlib.metadata import version

        return f"pysz {version('pysz')}"
    if codec == "fp8_e4m3_block_scale":
        import torch

        return f"torch {torch.__version__}"
    return f"zstandard {zstandard.__version__}"


def compress_archive_codec(
    source_path: Path,
    destination_path: Path,
    codec: str,
    **params: Any,
) -> dict[str, Any]:
    """Encode an uncompressed reference archive with a lossy codec, one block per (chunk, layer, site)."""
    if codec not in CODECS:
        raise ValueError(f"Unsupported codec: {codec}")
    encode, _, allowed = CODECS[codec]
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unsupported parameters for {codec}: {sorted(unknown)}")
    if codec == "bf16_trunc_zstd" and not 0 <= int(params.get("mantissa_bits", -1)) <= 7:
        raise ValueError("bf16_trunc_zstd requires 0 <= mantissa_bits <= 7")
    source = ActivationArchive(source_path)
    if source.index.get("compression") != "none":
        raise ValueError("Codec source must be an uncompressed reference archive")
    source.verify()
    destination_path = Path(destination_path)
    if destination_path.exists():
        raise FileExistsError(f"Archive destination already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.tmp-", dir=destination_path.parent)
    )
    chunks: list[dict[str, Any]] = []
    encode_started = time.perf_counter()
    try:
        for chunk_index, (metadata, bits) in enumerate(source.iter_chunks()):
            chunk_name = f"chunk-{chunk_index:08d}"
            values = _bf16_bits_to_float32(bits)
            if not np.isfinite(values).all():
                raise ValueError(f"{codec} requires finite activation values")

            chunk_path = temporary_path / chunk_name
            chunk_path.mkdir()
            payload_path = chunk_path / PAYLOAD_FILE
            blocks = []
            offset = 0
            with payload_path.open("wb") as output:
                for key, rows in _group_rows("layer_site", metadata):
                    payload, block_params = encode(np.ascontiguousarray(values[rows]), bits[rows], params)
                    output.write(payload)
                    blocks.append({"group": key, "rows": int(len(rows)), "offset": offset, "bytes": len(payload), **block_params})
                    offset += len(payload)
            metadata_file_name = "metadata.npz"
            output_metadata_path = chunk_path / metadata_file_name
            _save_metadata(output_metadata_path, metadata)
            chunks.append({
                "chunk": chunk_name,
                "rows": int(bits.shape[0]),
                "activation_file": PAYLOAD_FILE,
                "payload_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                "metadata_file": metadata_file_name,
                "metadata_sha256": _metadata_digest(metadata),
                "blocks": blocks,
                "stored_bytes": payload_path.stat().st_size + output_metadata_path.stat().st_size,
            })

        index: dict[str, Any] = {
            "format": CODEC_FORMAT,
            "version": CODEC_ARCHIVE_VERSION,
            "game_id": source.index["game_id"],
            "rows": int(source.index["rows"]),
            "dimensions": int(source.index["dimensions"]),
            "source_activation_dtype": "bfloat16",
            "metadata_fields": METADATA_COLUMNS,
            "metadata_dtypes": source.index["metadata_dtypes"],
            "compression": codec,
            "codec_parameters": params,
            "codec_version": _codec_version(codec),
            "block_layout": "one block per (chunk, layer, activation_site); rows in source order",
            "calibration": {"mode": "none", "note": "per-block parameters computed from the encoded block and stored"},
            "chunk_rows_max": source.index["chunk_rows_max"],
            "encode_time_s": time.perf_counter() - encode_started,
            "chunks": chunks,
        }
        index_path = temporary_path / "index.json"
        index["archive_size_bytes"] = _index_size(temporary_path, index_path, index)
        os.replace(temporary_path, destination_path)
        return index
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


class CodecActivationArchive:
    """Reader for lossy codec archives (version 4)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.index = json.loads((self.path / "index.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Archive index is missing or invalid: {self.path}") from error
        if self.index.get("format") != CODEC_FORMAT or self.index.get("version") != CODEC_ARCHIVE_VERSION:
            raise ValueError("Unsupported codec archive format or version")
        if self.index.get("compression") not in CODECS:
            raise ValueError(f"Unsupported codec: {self.index.get('compression')}")
        self._decode = CODECS[self.index["compression"]][1]

    def iter_chunks(self, **query):
        token_start = query.get("token_start")
        token_stop = query.get("token_stop")
        if (token_start is None) != (token_stop is None):
            raise ValueError("token_start and token_stop must be provided together")
        if token_start is not None and (token_start < 0 or token_stop <= token_start):
            raise ValueError("token range must satisfy 0 <= token_start < token_stop")
        dimensions = int(self.index["dimensions"])
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
            values = np.empty((int(chunk["rows"]), dimensions), dtype=np.float32)
            blocks = {block["group"]: block for block in chunk["blocks"]}
            with (chunk_path / chunk["activation_file"]).open("rb") as payload_file:
                for key, rows in _group_rows("layer_site", metadata):
                    if not mask[rows].any():
                        continue
                    block = blocks[key]
                    payload_file.seek(block["offset"])
                    payload = payload_file.read(block["bytes"])
                    values[rows] = self._decode(payload, (len(rows), dimensions), block)
            yield {name: values_[mask] for name, values_ in metadata.items()}, values[mask]

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
        row_count = 0
        for chunk in self.index["chunks"]:
            chunk_path = self.path / chunk["chunk"]
            payload_path = chunk_path / chunk["activation_file"]
            if hashlib.sha256(payload_path.read_bytes()).hexdigest() != chunk["payload_sha256"]:
                raise ValueError(f"Codec payload checksum mismatch in {chunk_path}")
            if sum(block["bytes"] for block in chunk["blocks"]) != payload_path.stat().st_size:
                raise ValueError(f"Codec block table does not cover the payload in {chunk_path}")
            if sum(block["rows"] for block in chunk["blocks"]) != int(chunk["rows"]):
                raise ValueError(f"Codec block rows do not match chunk rows in {chunk_path}")
            with np.load(chunk_path / chunk["metadata_file"], allow_pickle=False) as stored:
                metadata = {name: stored[name] for name in METADATA_COLUMNS}
            if _metadata_digest(metadata) != chunk["metadata_sha256"]:
                raise ValueError(f"Metadata checksum mismatch in {chunk_path}")
            row_count += int(chunk["rows"])
        if row_count != int(self.index["rows"]):
            raise ValueError("Codec archive row count does not match its index")
