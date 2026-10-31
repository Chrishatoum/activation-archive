#!/usr/bin/env python3
"""Measure row-group loading throughput and peak host/GPU memory."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.data import load_row_group
from benchmarks.common.paths import row_group_result_path

DEFAULT_ROOT = Path("/srv/gibbon-activations/gpt_oss_pilot_3games")


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports ru_maxrss in KiB.
    return int(value * 1024)


def _current_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", default="low_0056")
    parser.add_argument("--row-group", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("GIBBON_DATA_ROOT", DEFAULT_ROOT)))
    parser.add_argument("--gpu", action="store_true", help="also copy the loaded BF16 tensor to CUDA and measure GPU memory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_rss = _current_rss_bytes()
    baseline_peak_rss = _peak_rss_bytes()
    read_started = time.perf_counter()
    metadata, bits = load_row_group(args.game_id, args.row_group, args.root)
    read_time = time.perf_counter() - read_started
    peak_rss = _peak_rss_bytes()

    rows, dimensions = bits.shape
    activation_bytes = int(bits.nbytes)
    result = {
        "game_id": args.game_id,
        "row_group": args.row_group,
        "rows": int(rows),
        "dimensions": int(dimensions),
        "activation_bytes": activation_bytes,
        "metadata_fields": sorted(metadata),
        "read_time_s": read_time,
        "rows_per_second": float(rows / read_time) if read_time else None,
        "activation_mib_per_second": activation_bytes / (1024**2) / read_time if read_time else None,
        "host_memory": {
            "baseline_rss_bytes": baseline_rss,
            "process_peak_rss_bytes": peak_rss,
            "baseline_process_peak_rss_bytes": baseline_peak_rss,
            "new_high_water_bytes_during_load": max(0, peak_rss - baseline_peak_rss),
            "measurement_note": "Process peak RSS includes Python and imported libraries; new_high_water is the increase above the pre-load high-water mark.",
        },
        "gpu_memory": None,
        "data_root": str(args.root),
    }

    if args.gpu:
        try:
            import torch
        except ImportError as exc:
            parser.error(f"--gpu requires PyTorch: {exc}")
        if not torch.cuda.is_available():
            parser.error("--gpu was requested but CUDA is unavailable")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tensor = torch.from_numpy(bits).view(torch.bfloat16)
        transfer_started = time.perf_counter()
        gpu_tensor = tensor.to("cuda")
        torch.cuda.synchronize()
        transfer_time = time.perf_counter() - transfer_started
        result["gpu_memory"] = {
            "device": torch.cuda.get_device_name(),
            "transfer_time_s": transfer_time,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        del gpu_tensor

    output_path = args.output or row_group_result_path(args.game_id, args.row_group, "loader_measurement")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"result_file: {output_path}")


if __name__ == "__main__":
    main()