# Benchmark methods

This directory contains one benchmark script per compression strategy.

Implemented:
- raw_baseline.py
- zstd_lossless.py
- quant8.py

`quant4.py`, `block_scale.py`, `zfp.py`, and `sz3.py` were removed
(2026-11-03): they were never implemented past a `NotImplementedError`
stub, and ZFP/SZ3/blockwise scaling ended up implemented directly in
`python/codec_archive.py` instead (see FORMAT_SPEC.md's codec table).

Each script should:
1. load the same row group from the same game,
2. run one encoding/decoding strategy,
3. record size, error, and timing metrics,
4. write JSON results under benchmarks/results/.

## Streaming convention

Benchmarks are run on a row-group slice of the Parquet data, not on the entire dataset by default. A row group is a chunk of rows inside a Parquet file and is the natural unit for streaming access. The script should therefore accept explicit benchmark inputs such as:

```bash
python benchmarks/methods/raw_baseline.py --game-id low_0056 --row-group 0
```

This isolates one chunk and keeps the comparison fair. The same script should also support a full-stream mode later for iterating over all row groups in the selected game, for example:

```bash
python benchmarks/methods/raw_baseline.py --game-id low_0056 --stream-all
```

The important rule is that each run keeps the same game and row-group selection for all methods. Only the encoding strategy should change between scripts. This ensures the benchmark compares the same tensor data under different storage methods rather than mixing different input slices.

## Shared CLI contract

All benchmark scripts should accept the same core arguments:
- `--game-id`: the dataset/game to benchmark
- `--row-group`: the Parquet row group to load
- `--root`: the dataset root (default: GIBBON_DATA_ROOT)
- `--stream-all`: optional flag to iterate over every row group in the chosen game

The common loader and metric helpers in `benchmarks/common/` should enforce this same interface so each method script remains modular and comparable.

## Shared result contract

All methods must write the same JSON record shape. Use the helper in `benchmarks/common/schema.py` rather than handcrafting a custom dict per script. The canonical fields are:

- `method`, `game_id`, `row_group`, `rows`, `dimensions`
- `data_root`, `raw_size_bytes`, `encoded_size_bytes`, `compression_ratio`
- `read_time_s`, `encode_time_s`, `decode_time_s`, `rows_per_second`
- `relative_frobenius_error`, `max_abs_error`
- `sha256`, `roundtrip_exact`, `lossless`, `lossy`
- `metadata_fields`, `method_params`

This makes raw, zstd, quantized, and lossy methods directly comparable without reinterpreting different output schemas.

`lossless` and `lossy` describe the measured round-trip for that result: the shared schema derives them from `roundtrip_exact`. They are not separately asserted by each method.

`quant8.py` uses global affine min-max quantization over one row group. Its encoded size includes one uint8 code per activation plus the float64 minimum and scale needed to decode it; reported errors are measured against the original BF16 values.

`zstd_lossless.py` compresses the raw BF16 byte payload with zstandard (default compression level 3). The compressed size excludes Parquet/source metadata and represents only the encoded activation payload. Its `sha256` is the hash of the original payload, matching the raw baseline convention.

By default, row-group results are grouped under `benchmarks/results/<game_id>/row_groups/<row_group:04d>/`. Methods using configurable settings include the setting in the filename, for example `zstd_lossless_level_3.json`. Pass `--output` to select a different path.
