# Shared benchmark utilities

This directory contains helpers shared by all benchmark scripts.

## Shared responsibilities

The common code should provide:
- dataset selection helpers for game and row-group loading,
- row-group streaming logic for Parquet chunk access,
- metric functions for storage, error, and timing,
- a consistent CLI interface shared across all methods.

## Streaming contract

Benchmarks must be run on a fixed data slice so that different encoders are compared on identical inputs. The default behavior is to benchmark one row group, for example:

```bash
python benchmarks/methods/raw_baseline.py --game-id low_0056 --row-group 0
```

A later full-stream mode should iterate through all row groups for the selected game, but only after the single-row-group benchmark is working. The data choice should remain fixed within each run: same game, same row group, same tensor payload, different compression method only.

## Expected interface

Each benchmark script should accept the same arguments:
- `--game-id`
- `--row-group`
- `--root`
- `--stream-all`

This contract makes the benchmark pipeline reproducible and keeps result files comparable across methods.

## Canonical result schema

Every method should emit the same top-level JSON structure so that we can compare raw, lossless, and lossy encoders fairly. The shared helper in `benchmarks/common/schema.py` defines the contract.

Required keys:

- `method`: method name string (for example `raw_baseline`, `zstd_lossless`, `quant8`)
- `game_id`: dataset/game identifier
- `row_group`: Parquet row-group index
- `rows`: integer number of rows loaded
- `dimensions`: integer tensor width per row
- `data_root`: dataset root or empty string if not known
- `raw_size_bytes`: original BF16 payload size before encoding
- `encoded_size_bytes`: size of the encoded/transformed payload
- `compression_ratio`: `encoded_size_bytes / raw_size_bytes`
- `read_time_s`: time spent reading the row group from the dataset
- `encode_time_s`: time spent encoding/transcoding
- `decode_time_s`: time spent decoding/reconstructing
- `rows_per_second`: throughput for the data-loading phase
- `relative_frobenius_error`: relative reconstruction error (0 for lossless methods)
- `max_abs_error`: worst-case absolute error (0 for lossless methods)
- `sha256`: hash of the original or encoded payload, depending on the method contract
- `roundtrip_exact`: whether reconstruction returns the exact original values
- `lossless`: computed as `roundtrip_exact` for this input slice
- `lossy`: computed as the inverse of `roundtrip_exact` for this input slice
- `metadata_fields`: list of metadata keys attached to the row group
- `method_params`: nested dict of method-specific settings

The result should be generated via `make_result(...)` and validated via `validate_result(...)` before writing JSON to `benchmarks/results/`.

## Result organization

By default, save one JSON file per method and row group using this layout:

```text
benchmarks/results/<game_id>/row_groups/<row_group:04d>/<method>[_<variant>].json
```

This keeps method results for the same data slice together. Method variants that affect results, such as zstd compression level, belong in the filename and in `method_params`. Use the shared `row_group_result_path(...)` helper to construct default paths. An explicit `--output` can override the default. Reserve a separate `full_game/` directory for future whole-game aggregate results; do not mix those with individual row-group measurements.
