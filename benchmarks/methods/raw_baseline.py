#!/usr/bin/env python3
"""Raw BF16 baseline benchmark for one Parquet row group."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.data import load_row_group
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

    t0 = time.perf_counter()
    metadata, bits = load_row_group(args.game_id, args.row_group, args.root)
    read_time = time.perf_counter() - t0

    raw_bytes = bits.tobytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    result = make_result(
        method="raw_baseline",
        game_id=args.game_id,
        row_group=args.row_group,
        rows=int(bits.shape[0]),
        dimensions=int(bits.shape[1]),
        raw_size_bytes=len(raw_bytes),
        encoded_size_bytes=len(raw_bytes),
        read_time_s=read_time,
        rows_per_second=float(bits.shape[0]) / read_time if read_time > 0 else float("inf"),
        relative_frobenius_error=0.0,
        max_abs_error=0.0,
        sha256=sha256,
        roundtrip_exact=True,
        metadata_fields=sorted(metadata.keys()),
        method_params={},
        data_root=str(args.root),
    )
    validate_result(result)

    output_path = args.output or row_group_result_path(args.game_id, args.row_group, "raw_baseline")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
