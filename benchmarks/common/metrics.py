"""Shared metric helpers for compression experiments."""

from __future__ import annotations

from typing import Any


def relative_frobenius_error(original: Any, reconstructed: Any, eps: float = 1e-12) -> float:
    """Return the relative Frobenius error between two arrays."""
    original = original.astype(float)
    reconstructed = reconstructed.astype(float)
    diff = original - reconstructed
    denom = float((original**2).sum() ** 0.5) + eps
    numer = float((diff**2).sum() ** 0.5)
    return numer / denom


def max_abs_error(original: Any, reconstructed: Any) -> float:
    """Return the maximum absolute error across the two arrays."""
    return float(abs(original - reconstructed).max())


def compression_ratio(raw_size: int, encoded_size: int) -> float:
    """Return the compressed/raw ratio, where smaller values indicate better compression."""
    if raw_size <= 0:
        return 0.0
    return float(encoded_size) / float(raw_size)
