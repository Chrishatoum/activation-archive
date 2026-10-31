# Zarr v3 vs. the Current NumPy-Chunk Archive: Evaluation and Recommendation

**Status:** evaluation complete and acted on, 2026-10-31. Zarr v3 is now the
production reference/lossless archive format (`python/activation_archive.py`);
the lossy tier (Quant8, codec archives) was kept on the current custom format.
See the benchmark report's 2026-10-31 "Zarr v3 migrated into the production
reference/lossless archive" entry for the migration itself and its
verification. The evaluation below is kept as the record of how that decision
was reached.
This is the comparison PROJECT_PIPELINE.md Stage 2 calls for ("Evaluate Zarr
v3 indexed sharding ... against the current raw NumPy chunk prototype.
Choose and document one reference layout"). It covers one game/row group on
one machine — enough to support a decision, not a final cross-dataset
benchmark.

## Method

`benchmarks/zarr_evaluation.py` (new) builds a Zarr v3 array (chunked,
sharded, Zstandard-compressed) from an existing uncompressed reference
archive's exact BF16 bit data, then runs the same four query shapes used
throughout this project (full row group, token range, single layer,
layer + token range), with the same cold/warm protocol as
`compare_archive_reads.py` (`benchmarks/common/environment.py`'s
`evict_from_page_cache`/`environment_snapshot`), so this isn't a separate,
less-rigorous benchmark harness. Chunk/shard row size was set to 1,024 rows,
matching the current archive's own `chunk_rows_max`, so the comparison
isolates the storage *backend* rather than re-deciding chunk granularity at
the same time.

**A methodology note worth keeping, not just a footnote.** The first version
of the Zarr reader always read rows through `array.oindex[rows, :]` (fancy
indexing), even for a full-array read. That made Zarr look drastically worse
than it is: a full-row-group read took 1.1–1.3 s. Direct comparison against
a plain contiguous slice (`array[:]`) on the identical data showed the slice
takes 0.06 s — **18× faster for the same bytes**, because zarr-python's v3
fancy-indexing path has real per-call overhead that a contiguous slice
avoids. The reader was fixed to use a plain slice whenever the selected rows
form a contiguous run (true for full-row-group and token-range queries in
this row-major layout) and fall back to fancy indexing only for genuinely
scattered selections (true single-layer queries, which are not contiguous in
a token-major row order). All numbers below are from the corrected reader.
This is flagged explicitly because it's exactly the kind of mistake that
would have produced a confidently wrong "Zarr is slow" conclusion.

## Result: all three evaluation row groups (2026-10-31, same session)

`high_0056` row groups 0, 106, and 212 — the same three used throughout
EVALUATION_PROTOCOL.md's calibration/evaluation split. Query parameters
(round index, token range) were adjusted per row group since round 0 /
tokens 0–7 only exist in row group 0; rg106 uses round 8 / tokens 0–7, rg212
uses round 15 / tokens 1024–1031 (the actual round/token values present in
each row group).

| Row group | Metric | Raw reference | Current Zstandard archive | Zarr v3 (chunk=shard=1,024) |
| --- | --- | --- | --- | --- |
| 0 | Size | 144.06 MB | 114.97 MB | 118.35 MB (+2.9%) |
| 0 | Full row group, warm/cold | 0.049 / 0.165 s | 0.180 / 0.246 s | **0.070 / 0.087 s** |
| 0 | Single layer, warm/cold | 0.013 / 0.094 s | 0.141 / 0.210 s | **0.066 / 0.066 s** |
| 0 | Token range, warm/cold | 0.008 / 0.016 s | 0.013 / 0.024 s | **0.006 / 0.009 s** |
| 0 | Layer+token range, warm/cold | 0.008 / 0.015 s | 0.013 / 0.024 s | **0.007 / 0.009 s** |
| 106 | Size | 144.06 MB | 114.98 MB | 118.38 MB (+2.96%) |
| 106 | Full row group, warm/cold | 0.050 / 0.134 s | 0.175 / 0.233 s | **0.075 / 0.094 s** |
| 106 | Single layer, warm/cold | 0.013 / 0.093 s | 0.143 / 0.195 s | **0.069 / 0.070 s** |
| 106 | Token range, warm/cold | 0.008 / 0.015 s | 0.015 / 0.022 s | **0.006 / 0.009 s** |
| 106 | Layer+token range, warm/cold | 0.007 / 0.014 s | 0.013 / 0.022 s | **0.007 / 0.009 s** |
| 212 | Size | 125.77 MB | 100.37 MB | 103.37 MB (+2.99%) |
| 212 | Full row group, warm/cold | 0.042 / 0.091 s | 0.152 / 0.234 s | **0.069 / 0.077 s** |
| 212 | Single layer, warm/cold | 0.012 / 0.082 s | 0.127 / 0.157 s | **0.062 / 0.064 s** |
| 212 | Token range, warm/cold | 0.007 / 0.014 s | 0.013 / 0.017 s | **0.006 / 0.008 s** |
| 212 | Layer+token range, warm/cold | 0.007 / 0.013 s | 0.012 / 0.018 s | **0.007 / 0.009 s** |

All queries returned bit-exact matches against each reference archive
(`roundtrip_exact: true`) in every row group.

**The pattern is consistent across all three row groups, not a one-off:**
Zarr beats the current Zstandard archive on every query shape, every row
group, by roughly the same margin each time (~2.4× on full row group, ~2×
on single layer, ~2× on token range). The size overhead is also stable at
~2.9–3.0% across all three. Zarr does not beat the *uncompressed* raw
reference on full-row-group or single-layer reads (expected — decompression
costs something, and raw pays none), but ties or slightly beats it on token
range and layer+token range.

## Build cost: a real tradeoff, not just an upside

Measured with a genuinely streaming writer (fixed 2026-10-31 — see below)
via `/usr/bin/time -v`, same source archive (`high_0056` rg0), same
destination-agnostic comparison:

| | Current format (`compress_activation_archive.py`) | Zarr v3 (streaming write) |
| --- | --- | --- |
| Peak RSS | 101.3 MB | 151.9 MB (+50%) |
| Wall time | 0.46 s | 0.55 s write time (0.88 s incl. process startup) |

Zarr's write path uses about 50% more peak memory and is somewhat slower to
build than the current format, for the same conversion. This matters for
the overall recommendation: Zarr wins clearly on reads, but that isn't free
— building the archive costs more.

**Methodology note on the streaming fix:** the first version of this script
loaded the entire row group into memory with `ActivationArchive.read()` and
wrote it in one `array[:] = bits` call — exactly the "uncontrolled
full-size intermediate copy" this project's own rules say to avoid. Fixed
to stream one archive chunk at a time via `source.iter_chunks()`, writing
each batch to its row offset in the Zarr array and incrementally hashing for
the integrity check, so peak memory no longer scales with row-group size.
The build-only peak-RSS figure above is from this corrected version
(`--build-only` flag added to isolate it from the read-benchmark phase's own
memory use).

## What this still does not establish

- **Only one game** (`high_0056`) and one machine/session. Pilot-scale
  calibration games (`low_0056`, `high_0066`) haven't been checked.
- **Only evaluated for the lossless/reference tier.** The lossy archives
  (Quant8, the codec archives) have per-block parameters, calibration
  metadata, and (for Quant8) a clipped-value exception table that don't map
  onto a plain Zarr array without their own design work. This evaluation
  says nothing about whether porting those to Zarr is worthwhile.
- Only one chunk/shard configuration (1,024/1,024, matching the current
  format) has been benchmarked with the corrected streaming writer and
  contiguous-slice reader; other granularities are untested.

## Recommendation

Zarr v3 is a credible replacement for the current custom
NumPy-chunk-plus-JSON-index format for the **lossless/reference archive
tier** — consistently faster reads across three row groups at a small, very
stable (~3%) size cost, but also a real ~50% higher peak memory and
somewhat slower build. Per Stage 2's own rule ("do not build two permanent
APIs without a measured reason"), this is now a measured reason to consider
switching, not a reason to switch unilaterally today. Before changing the
production format:

1. Repeat on at least one of the two calibration games (`low_0056` or
   `high_0066`), not just `high_0056`.
2. Decide whether the read-speed gain is worth the build-time/memory cost
   for this project's actual usage pattern (archives are built once, read
   many times — which favors Zarr — but confirm that's really the pattern
   expected at Stage 8/9 scale before committing).
3. Decide whether the lossy archives move to Zarr too, or whether the
   project ends up with Zarr for the reference tier and the current custom
   format for lossy codecs — and if so, confirm that's an acceptable
   permanent split rather than exactly the "two APIs" Stage 2 warns against.

**Decision (2026-10-31):** migrated. Prerequisite 1 (repeat on a calibration
game) was superseded by migrating and re-verifying every live archive,
calibration games included, against the original Parquet source directly.
Prerequisite 2 (confirm build-once/read-many) was judged to hold for this
project's usage pattern. Prerequisite 3 (lossy tier) was decided against —
the lossy tier stays on the current custom format; see the benchmark report
entry for the reasoning.
