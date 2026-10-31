#!/usr/bin/env python3
"""Stream exact BF16 activation bits from one Parquet row group."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_ROOT = Path("/srv/gibbon-activations/gpt_oss_pilot_3games")
METADATA_COLUMNS = ["game_id", "round", "layer", "activation_site", "token_idx"]


def _binary2_as_uint16(column: pa.ChunkedArray) -> np.ndarray:
    """Decode Arrow binary(2) values without changing their BF16 bits."""
    array = column.combine_chunks()
    if array.null_count:
        raise ValueError("Activation columns must not contain null values")
    if not (
        pa.types.is_fixed_size_binary(array.type)
        or pa.types.is_binary(array.type)
        or pa.types.is_large_binary(array.type)
    ):
        raise TypeError(f"Expected a binary activation column, got {array.type}")

    if pa.types.is_fixed_size_binary(array.type):
        if array.type.byte_width != 2:
            raise ValueError("Activation values are not exactly two bytes wide")
        _, data_buffer = array.buffers()
        return np.frombuffer(
            data_buffer, dtype="<u2", count=len(array), offset=array.offset * 2
        ).copy()

    offset_dtype = "<i8" if pa.types.is_large_binary(array.type) else "<i4"
    _, offset_buffer, data_buffer = array.buffers()
    offsets = np.frombuffer(
        offset_buffer,
        dtype=offset_dtype,
        count=len(array) + 1,
        offset=array.offset * np.dtype(offset_dtype).itemsize,
    )
    if not np.all(np.diff(offsets) == 2):
        raise ValueError("Activation values are not exactly two bytes wide")

    # Ramiel/NCC are little-endian. Copy so the result owns stable memory after
    # the Arrow table leaves scope.
    return np.frombuffer(
        data_buffer, dtype="<u2", count=len(array), offset=int(offsets[0])
    ).copy()


def read_row_group_bits(
    parquet_path: Path, row_group: int, dimension_indices: Sequence[int] | None = None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return metadata arrays and exact BF16 bits with shape [rows, dimensions]."""
    parquet = pq.ParquetFile(parquet_path)
    feature_columns = [c for c in parquet.schema.names if c not in METADATA_COLUMNS]
    if dimension_indices is not None:
        feature_columns = [feature_columns[i] for i in dimension_indices]

    table = parquet.read_row_group(
        row_group, columns=METADATA_COLUMNS + feature_columns, use_threads=True
    )
    metadata = {
        name: table[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in METADATA_COLUMNS
    }
    bits = np.empty((table.num_rows, len(feature_columns)), dtype="<u2")
    for index, name in enumerate(feature_columns):
        bits[:, index] = _binary2_as_uint16(table[name])
    return metadata, bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_id", choices=["high_0056", "low_0056", "high_0066"])
    parser.add_argument("--row-group", type=int, default=0)
    parser.add_argument("--dimensions", type=int, nargs="*", default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("GIBBON_DATA_ROOT", DEFAULT_ROOT)),
    )
    args = parser.parse_args()

    manifest = json.loads((args.root / "manifest.json").read_text())
    record = next(g for g in manifest["games"] if g["game_id"] == args.game_id)
    parquet_path = args.root / record["activation_file"]
    metadata, bits = read_row_group_bits(
        parquet_path, args.row_group, args.dimensions
    )

    print(f"file: {parquet_path}")
    print(f"row group: {args.row_group}")
    print(f"exact bits: shape={bits.shape}, dtype={bits.dtype}, bytes={bits.nbytes:,}")
    print(
        "metadata: "
        f"rounds={np.unique(metadata['round']).tolist()}, "
        f"layers={np.unique(metadata['layer']).tolist()}, "
        f"sites={np.unique(metadata['activation_site']).tolist()}"
    )

    if args.gpu:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available in this environment")
        tensor = torch.from_numpy(bits).view(torch.bfloat16).to("cuda")
        print(
            f"GPU tensor: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"device={tensor.device}"
        )


if __name__ == "__main__":
    main()
