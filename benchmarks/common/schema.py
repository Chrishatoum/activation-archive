"""Canonical benchmark result schema for activation-storage experiments."""

from __future__ import annotations

from typing import Any

from benchmarks.common.metrics import compression_ratio

REQUIRED_RESULT_KEYS = (
    "method",
    "game_id",
    "row_group",
    "rows",
    "dimensions",
    "data_root",
    "raw_size_bytes",
    "encoded_size_bytes",
    "compression_ratio",
    "read_time_s",
    "encode_time_s",
    "decode_time_s",
    "rows_per_second",
    "relative_frobenius_error",
    "max_abs_error",
    "sha256",
    "roundtrip_exact",
    "lossless",
    "lossy",
    "metadata_fields",
    "method_params",
)


def make_result(
    *,
    method: str,
    game_id: str,
    row_group: int,
    rows: int,
    dimensions: int,
    raw_size_bytes: int,
    encoded_size_bytes: int,
    read_time_s: float,
    encode_time_s: float = 0.0,
    decode_time_s: float = 0.0,
    rows_per_second: float | None = None,
    relative_frobenius_error: float = 0.0,
    max_abs_error: float = 0.0,
    sha256: str | None = None,
    roundtrip_exact: bool = True,
    metadata_fields: list[str] | None = None,
    method_params: dict[str, Any] | None = None,
    data_root: str | None = None,
) -> dict[str, Any]:
    """Create a canonical result dictionary for a single benchmark run."""
    if metadata_fields is None:
        metadata_fields = []
    if method_params is None:
        method_params = {}
    if rows_per_second is None:
        rows_per_second = float(rows) / read_time_s if read_time_s > 0 else float("inf")

    return {
        "method": method,
        "game_id": game_id,
        "row_group": int(row_group),
        "rows": int(rows),
        "dimensions": int(dimensions),
        "data_root": data_root or "",
        "raw_size_bytes": int(raw_size_bytes),
        "encoded_size_bytes": int(encoded_size_bytes),
        "compression_ratio": compression_ratio(raw_size_bytes, encoded_size_bytes),
        "read_time_s": float(read_time_s),
        "encode_time_s": float(encode_time_s),
        "decode_time_s": float(decode_time_s),
        "rows_per_second": float(rows_per_second),
        "relative_frobenius_error": float(relative_frobenius_error),
        "max_abs_error": float(max_abs_error),
        "sha256": sha256 or "",
        "roundtrip_exact": bool(roundtrip_exact),
        "lossless": bool(roundtrip_exact),
        "lossy": not bool(roundtrip_exact),
        "metadata_fields": sorted(metadata_fields),
        "method_params": dict(method_params),
    }


def validate_result(result: dict[str, Any]) -> None:
    """Check that a result dictionary satisfies the shared benchmark contract."""
    missing = [key for key in REQUIRED_RESULT_KEYS if key not in result]
    if missing:
        raise ValueError(f"Benchmark result missing required keys: {missing}")
