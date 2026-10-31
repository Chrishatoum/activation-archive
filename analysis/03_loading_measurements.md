# Row-Group Loading Measurements

This note records a first measurement of reading activation data from the source
Parquet files. It covers row group 0 from one small run and one large run; it is
not a measurement of a complete game.

## What was measured

The loader read each selected Parquet row group into CPU memory (RAM). With GPU
measurement enabled, it then copied the same BF16 values from RAM to an NVIDIA
GeForce RTX 4090. The measured activation payload size is calculated from the
number of values: rows × 2,880 dimensions × 2 bytes per BF16 value.

## Results

| Run and row group | Rows | Activation payload | Parquet read time | Read throughput | Increase in process peak RAM | GPU copy time | GPU memory allocated / reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `low_0056`, 0 | 1,392 | 7.64 MiB | 0.200 s | 38.2 MiB/s | 141.4 MiB | 0.072 s | 7.64 / 20.0 MiB |
| `high_0056`, 0 | 24,576 | 135.0 MiB | 1.359 s | 99.3 MiB/s | 1,043.6 MiB | 0.059 s | 136.0 / 136.0 MiB |

GPU memory was measured on the RTX 4090. “Allocated” is memory used by the
activation tensor; “reserved” is memory held by PyTorch's CUDA allocator.

## What the results suggest

The larger row group loaded faster per MiB in this run, but its process used
about 1.0 GiB more peak RAM while holding 135 MiB of activation values. This
suggests that Parquet decoding, temporary buffers, and the loader's conversion
may use substantially more RAM than the final activation array. That is worth
investigating when designing a streaming reader.

The GPU allocation is close to the activation payload size, as expected for a
BF16 tensor copy. GPU use is optional for storage compression; this measurement
only shows the cost of moving a loaded chunk to the GPU for later computation.

## How to reproduce

```bash
python benchmarks/measure_row_group.py --game-id low_0056 --row-group 0 --gpu
python benchmarks/measure_row_group.py --game-id high_0056 --row-group 0 --gpu
```

The measurement JSON files are [low_0056 row group 0](../benchmarks/results/low_0056/row_groups/0000/loader_measurement.json)
and [high_0056 row group 0](../benchmarks/results/high_0056/row_groups/0000/loader_measurement.json).

## Limits of this measurement

- Each row group was measured once; timings may vary between runs.
- Cold versus warm filesystem cache was not controlled.
- Process peak RAM includes Python and imported libraries. The reported increase
  is the change in the process high-water mark after loading, not a direct
  measurement of memory used only by the activation values.
- Only row group 0 from `low_0056` and `high_0056` was measured. We have not
  measured every row group, a complete game, or other hardware.
- This measures loading from the existing Parquet files. It does not yet
  measure a new compressed archive.
