# Low_0056 row-group 0 benchmark summary

## Dataset
- Game: `low_0056`
- Row group: `0`
- Rows: `1392`
- Dimensions per row: `2880`

## Baseline raw storage
- Raw size: `8,017,920 bytes`
- Read time: `0.294710 s`
- Throughput: `4,723.3 rows/s`

## Lossless zlib compression
- Compressed size: `6,440,941 bytes`
- Compression ratio: `0.803` (compressed/raw)
- Encode time: `0.512325 s`
- Decode time: `0.045383 s`

## Quality check
- Round-trip equality: `True`
- Reconstruction quality: `exact`
- This confirms the compressed form preserved the original BF16 bytes exactly.

## Interpretation
This first benchmark shows that lossless zlib compression can reduce the storage footprint of the row group while preserving exact numerical fidelity. The reduction is modest for this small prototype but demonstrates the pipeline works and is ready for broader method comparisons.
