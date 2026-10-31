#!/usr/bin/env python3
"""Global affine 8-bit quantization benchmark for one Parquet row group."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.data import load_row_group
from benchmarks.common.metrics import max_abs_error, relative_frobenius_error
from benchmarks.common.paths import row_group_result_path
from benchmarks.common.schema import make_result, validate_result

DEFAULT_ROOT = Path("/srv/gibbon-activations/gpt_oss_pilot_3games")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", type=str, default="low_0056")
    parser.add_argument("--row-group", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("GIBBON_DATA_ROOT", DEFAULT_ROOT)))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    read_started = time.perf_counter()
    metadata, bits = load_row_group(args.game_id, args.row_group, args.root)
    read_time = time.perf_counter() - read_started

    raw_bytes = bits.tobytes()
    original = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if not np.isfinite(original).all():
        parser.error("quant8 requires finite activation values")

    encode_started = time.perf_counter()
    minimum = float(original.min())
    maximum = float(original.max())
    scale = (maximum - minimum) / 255.0 if maximum > minimum else 1.0
    codes = np.rint((original - minimum) / scale).clip(0, 255).astype(np.uint8)
    parameter_bytes = np.asarray([minimum, scale], dtype="<f8").tobytes()
    encode_time = time.perf_counter() - encode_started

    decode_started = time.perf_counter()
    decoded_minimum, decoded_scale = np.frombuffer(parameter_bytes, dtype="<f8")
    reconstructed = codes.astype(np.float64) * decoded_scale + decoded_minimum
    decode_time = time.perf_counter() - decode_started

    reconstructed_float32 = reconstructed.astype(np.float32)
    original_float32 = original.astype(np.float32)
    roundtrip_exact = np.array_equal(
        reconstructed_float32.view(np.uint32),
        original_float32.view(np.uint32),
    )
    encoded_size_bytes = codes.nbytes + len(parameter_bytes)
    result = make_result(
        method="quant8",
        game_id=args.game_id,
        row_group=args.row_group,
        rows=int(bits.shape[0]),
        dimensions=int(bits.shape[1]),
        raw_size_bytes=len(raw_bytes),
        encoded_size_bytes=encoded_size_bytes,
        read_time_s=read_time,
        encode_time_s=encode_time,
        decode_time_s=decode_time,
        relative_frobenius_error=relative_frobenius_error(original, reconstructed),
        max_abs_error=max_abs_error(original, reconstructed),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        roundtrip_exact=roundtrip_exact,
        metadata_fields=sorted(metadata.keys()),
        method_params={
            "quantization": "global_affine_minmax",
            "bits_per_value": 8,
            "minimum": minimum,
            "scale": scale,
            "parameter_bytes": len(parameter_bytes),
        },
        data_root=str(args.root),
    )
    validate_result(result)

    output_path = args.output or row_group_result_path(args.game_id, args.row_group, "quant8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
