#!/usr/bin/env python3
"""Measure a lossy archive's distortion against its reference archive, per (layer, site).

Reports the proposal's fidelity metrics for every (layer, activation_site) group:
relative Frobenius error, maximum absolute error, absolute-error quantiles,
mean and variance changes, and pairwise inner-product changes on sampled rows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.environment import environment_snapshot
from python.activation_archive import ActivationArchive
from python.codec_archive import CODECS, CodecActivationArchive
from python.quantized_archive import QuantizedActivationArchive

EPSILON = 1e-12


def _open_lossy(path: Path):
    index = json.loads((path / "index.json").read_text())
    if str(index.get("compression", "")).startswith("quant8_"):
        return QuantizedActivationArchive(path)
    if index.get("compression") in CODECS:
        return CodecActivationArchive(path)
    raise ValueError(f"Unsupported lossy archive codec: {index.get('compression')}")


def group_metrics(original: np.ndarray, decoded: np.ndarray, sample_rows: int, rng: np.random.Generator) -> dict[str, float]:
    """Fidelity metrics for one group's rows (both arrays are float64, rows x dimensions)."""
    errors = original - decoded
    absolute = np.abs(errors)
    mean = original.mean(axis=0)
    decoded_mean = decoded.mean(axis=0)
    variance = original.var(axis=0)
    decoded_variance = decoded.var(axis=0)
    # Dimensions with zero variance have no meaningful ratio; skip them.
    varying = variance > EPSILON
    variance_ratio = decoded_variance[varying] / variance[varying]

    rows = rng.choice(original.shape[0], size=min(sample_rows, original.shape[0]), replace=False)
    sample, decoded_sample = original[rows], decoded[rows]
    gram = sample @ sample.T
    decoded_gram = decoded_sample @ decoded_sample.T
    norms = np.sqrt(np.diag(gram)) + EPSILON
    decoded_norms = np.sqrt(np.diag(decoded_gram)) + EPSILON
    cosine = gram / np.outer(norms, norms)
    decoded_cosine = decoded_gram / np.outer(decoded_norms, decoded_norms)
    off_diagonal = ~np.eye(len(rows), dtype=bool)
    cosine_change = np.abs(cosine - decoded_cosine)[off_diagonal] if len(rows) > 1 else np.zeros(1)

    return {
        "rows": int(original.shape[0]),
        "relative_frobenius_error": float(np.linalg.norm(errors) / (np.linalg.norm(original) + EPSILON)),
        "max_abs_error": float(absolute.max(initial=0.0)),
        "abs_error_p50": float(np.quantile(absolute, 0.5)),
        "abs_error_p99": float(np.quantile(absolute, 0.99)),
        "abs_error_p999": float(np.quantile(absolute, 0.999)),
        "value_abs_p50": float(np.quantile(np.abs(original), 0.5)),
        "relative_mean_shift": float(np.linalg.norm(mean - decoded_mean) / (np.linalg.norm(mean) + EPSILON)),
        "variance_ratio_median": float(np.median(variance_ratio)) if variance_ratio.size else 1.0,
        "variance_ratio_worst_deviation": float(np.abs(variance_ratio - 1).max(initial=0.0)),
        "inner_product_relative_error": float(
            np.linalg.norm(gram - decoded_gram) / (np.linalg.norm(gram) + EPSILON)
        ),
        "cosine_abs_change_median": float(np.median(cosine_change)),
        "cosine_abs_change_max": float(cosine_change.max(initial=0.0)),
        "inner_product_sample_rows": int(len(rows)),
    }


def evaluate(reference_path: Path, lossy_path: Path, sample_rows: int, seed: int) -> dict[str, Any]:
    reference = ActivationArchive(reference_path)
    lossy = _open_lossy(lossy_path)
    lossy.verify()
    if reference.index["game_id"] != lossy.index["game_id"] or reference.index["rows"] != lossy.index["rows"]:
        raise ValueError("Lossy archive does not match the reference archive's game/row count")
    rng = np.random.default_rng(seed)
    groups = []
    squared_error = squared_value = 0.0
    max_abs_error = 0.0
    present = set(zip(reference.metadata["layer"].tolist(), reference.metadata["activation_site"].tolist()))
    for layer, site in sorted(present):
        metadata, bits = reference.read(layer=layer, activation_site=site)
        decoded_metadata, decoded = lossy.read(layer=layer, activation_site=site)
        for name in metadata:
            if not np.array_equal(metadata[name], decoded_metadata[name]):
                raise AssertionError(f"Metadata differs for {name} in layer {layer} {site}")
        original = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        metrics = group_metrics(original, decoded.astype(np.float64), sample_rows, rng)
        groups.append({"layer": layer, "activation_site": site, **metrics})
        squared_error += float(np.square(original - decoded).sum())
        squared_value += float(np.square(original).sum())
        max_abs_error = max(max_abs_error, metrics["max_abs_error"])

    def worst(metric: str) -> dict[str, Any]:
        group = max(groups, key=lambda item: item[metric])
        return {"value": group[metric], "layer": group["layer"], "activation_site": group["activation_site"]}

    relative_errors = [group["relative_frobenius_error"] for group in groups]
    return {
        "reference_archive": str(reference_path),
        "lossy_archive": str(lossy_path),
        "game_id": reference.index["game_id"],
        "codec": lossy.index["compression"],
        "scale_granularity": lossy.index.get("scale_granularity"),
        "calibration": lossy.index.get("calibration", {"mode": "in_sample"}),
        "archive_size_bytes": int(lossy.index["archive_size_bytes"]),
        "reference_size_bytes": int(reference.index["archive_size_bytes"]),
        "codec_parameters": lossy.index.get("codec_parameters"),
        "parameter_bytes": int(lossy.index.get("parameter_bytes", 0)),
        "clipped_fraction": lossy.index.get("clipped_fraction"),
        "encode_time_s": lossy.index.get("encode_time_s"),
        "overall": {
            "relative_frobenius_error": float(np.sqrt(squared_error) / (np.sqrt(squared_value) + EPSILON)),
            "max_abs_error": max_abs_error,
            "group_relative_frobenius_median": float(np.median(relative_errors)),
        },
        "worst_group": {
            metric: worst(metric)
            for metric in (
                "relative_frobenius_error",
                "relative_mean_shift",
                "variance_ratio_worst_deviation",
                "inner_product_relative_error",
                "cosine_abs_change_max",
            )
        },
        "inner_product_sample_rows": sample_rows,
        "seed": seed,
        "environment": environment_snapshot(),
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Uncompressed reference archive")
    parser.add_argument("lossy", type=Path, help="Lossy archive built from that reference")
    parser.add_argument("--output", type=Path, required=True, help="Result JSON path")
    parser.add_argument("--sample-rows", type=int, default=256, help="Rows per group for inner-product checks")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = evaluate(args.reference, args.lossy, args.sample_rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("lossy_archive", "overall", "worst_group", "clipped_fraction")}, indent=2))
    print(f"result_file: {args.output}")


if __name__ == "__main__":
    main()
