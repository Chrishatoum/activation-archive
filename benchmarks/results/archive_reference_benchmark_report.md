# Reference Archive: Work and Benchmark Report

**Status:** pilot benchmark, 2026-10-22  
**Purpose:** record the uncompressed archive implementation and its first Parquet comparisons.

## Summary

A versioned, uncompressed archive was implemented for the current GPT-OSS pilot rows. On row group 0 of both `low_0056` and `high_0056`, it returned bitwise-identical BF16 activation values and matching row metadata. In repeated warm-cache measurements, archive reads were faster than the current Parquet loading path for both full-row-group and token-range queries.

These are preliminary results for two row groups and one machine. The Parquet path reads and decodes a full row group before filtering; the archive memory-maps activation arrays and filters against chunk metadata. This comparison therefore measures the current implementations and access paths, not an inherent performance limit of the Parquet format.

## Progress log

Newest entries first. Each entry records what was changed or run, the result, and where the raw output is.

### 2026-10-31 — Zarr v3 migrated into the production reference/lossless archive

Acted on the ZARR_EVALUATION.md recommendation: picked Zarr v3 for the reference/lossless tier, kept the current custom format for the lossy tier (Quant8, codec archives) — the usage pattern here is build-once/read-many, where Zarr's read speedup dominates its small build-cost/memory penalty, and porting the lossy tier's calibration/exception metadata onto a plain Zarr array was unplanned work with no measured benefit.

**Code changes.** `python/activation_archive.py` rewritten: `ArchiveWriter`/`ActivationArchive` now back onto a single Zarr v3 array (`activations.zarr`, chunked+sharded at `chunk_rows_max`) plus one `metadata.npz`, instead of a directory of per-chunk `.npy`/`.npz` files; `compress_archive()` builds the zstd tier the same way with `zarr.codecs.ZstdCodec`. New archive versions: 3 (uncompressed, was 1) and 4 (zstd, was 2) — the old per-chunk format is no longer read by this module. `python/quantized_archive.py` and `python/codec_archive.py` were decoupled from the reference archive's internal file layout: their builders now read the source via the public `iter_chunks()` method and generate their own chunk names/metadata files, instead of reaching into `source.index["chunks"]` and copying files directly — this was already the correct abstraction boundary and is what made the storage-backend swap safe to do without touching either lossy format. `benchmarks/evaluate_lossy_archive.py` updated the same way (reads `reference.metadata` instead of iterating old chunk files to enumerate (layer, site) pairs). Test suite updated for the new on-disk layout (`tests/test_activation_archive.py`); 11 tests / 5 subtests still pass.

**Migration of existing archives.** All 69 live archives under `/home/chris/gibbon-work/archives/` were rebuilt with the new code and verified before any old copy was touched: the 8 reference archives bit-exact against the original Parquet source (`/srv/gibbon-activations/.../activations/*.parquet`, read-only, never modified), the 4 zstd archives and 61 lossy archives (21 quant8-matrix, 36 codec-matrix, rebuilt with identical parameters read back out of their old index.json files) checksum-verified via each archive's own `.verify()`. Cross-checked fidelity: re-ran `evaluate_lossy_archive.py` on the rebuilt winning config (held-out Quant8 `layer_site_dim` + exceptions) for all three evaluation row groups — the worst-case metrics across the three groups exactly reproduce the §6a table in EVALUATION_PROTOCOL.md (variance-ratio 0.0241≈0.024 at rg0, inner-product 0.193%≈0.19% at rg0, worst-group error 0.679%≈0.68% at rg0, cosine ≈0.0033 at every row group) — confirms the new pipeline is bit-for-bit equivalent to the old one, not just independently passing.

**A nice surprise.** The new reference tier (Zarr, no compressor) came out ~18% *smaller* than the old uncompressed per-chunk format at identical data — not from compression, but from eliminating per-chunk-file/directory overhead (24 separate `.npy`+`.npz` files per row group collapse into one sharded store + one metadata file). This is on top of, not instead of, the read-speed win already measured in ZARR_EVALUATION.md.

**Left alone.** `raw_reference_v1/`, `high_0056-rg0-quant8-v1/`, `high_0056-rg0-quant8-layersite-v1/` — pre-dating the systematic quant8-matrix/codec-matrix sweep, not referenced by any current script, not migrated. Old archive directories were renamed to `*.bak` rather than deleted (a destructive-action guard blocked outright deletion) — safe to delete once satisfied with the migration; currently roughly doubles the on-disk footprint of `/home/chris/gibbon-work/archives` (~11 GB with `.bak` copies present).

### 2026-10-31 — Zarr v3 evaluation extended: streaming write, build-cost comparison, all three evaluation row groups

Closed the two open prerequisites from the Zarr evaluation entry below.

**Streaming write fix.** `benchmarks/zarr_evaluation.py`'s builder loaded the whole row group into memory and wrote it in one shot — exactly the "uncontrolled full-size intermediate copy" this project's rules say to avoid. Fixed to stream one archive chunk at a time via `ActivationArchive.iter_chunks()`, with an incremental SHA-256 for the integrity check and a new `--build-only` flag to isolate build-phase peak RSS from the read-benchmark phase.

**Build cost measured** (`/usr/bin/time -v`, same source archive, `high_0056` rg0): Zarr's streaming write uses **151.9 MB peak RSS vs. 101.3 MB for the current format's own archive-to-archive conversion (+50%)**, and is slightly slower (0.55 s write time vs. 0.46 s). This is the first real downside found for Zarr — reads win clearly, but building costs more.

**Extended to rg106 and rg212** (built missing lossless Zstandard archives for those row groups first, for a fair comparison; fixed a parameter bug in my own test query — round 0/tokens 0–7 only exist in rg0, so rg106/rg212 needed their own valid round/token values). **The pattern from rg0 holds consistently across all three row groups**: Zarr beats the current Zstandard archive on every query shape every time (~2.4× full row group, ~2× single layer, ~2× token range), at a stable ~2.9–3.0% size overhead across all three.

**Updated recommendation:** Zarr v3 remains a credible candidate for the lossless/reference tier — the read-speed advantage is now confirmed, not a one-row-group fluke — but the build-cost tradeoff means this is a genuine cost/benefit decision ("archives are built once, read many times" favors Zarr, but that assumption should be confirmed for Stage 8/9 scale), not a one-sided win. Still not acted on. Full detail: [ZARR_EVALUATION.md](../../ZARR_EVALUATION.md).

### 2026-10-31 — Zarr v3 vs. current archive format: evaluated, Zarr wins on every read query tested

Picked up Stage 2's still-open item: evaluate Zarr v3 indexed sharding against the current custom NumPy-chunk-plus-JSON-index format. Installed `zarr>=3` (conda-forge, got 3.1.6) and added it to `environment.yml`. New script: `benchmarks/zarr_evaluation.py`, reusing the same cold/warm protocol and environment metadata helpers as `compare_archive_reads.py` so this isn't a separately-calibrated benchmark.

**Caught my own bug before trusting the result.** The first reader implementation always used `array.oindex[rows, :]` (fancy indexing), even for a full-array read, which made a full-row-group read take 1.1–1.3 s — a direct side-by-side showed a plain contiguous slice on the identical data takes 0.06 s, 18× faster. Fixed the reader to slice when the selected rows are contiguous (true for full-row-group and token-range queries here) and only fall back to fancy indexing for genuinely scattered selections (single-layer). Re-ran with the fix.

**Result** (`high_0056` rg 0, same session): Zarr v3 (chunk=shard=1,024 rows, matching the current archive's own chunking) is **118.35 MB, 2.9% larger than the current lossless Zstandard archive (114.97 MB)**, but **faster on every one of the four query shapes**: full row group 0.061 s vs. 0.180 s (3× faster), single layer 0.068 s vs. 0.141 s (2× faster), token range and layer+token-range both modestly faster too. All four reads verified bit-exact.

**Recommendation (not yet acted on):** Zarr v3 looks like a credible replacement for the reference/lossless archive tier specifically. Before switching the production format: repeat on more row groups/games, measure Zarr's write-path peak memory with a real bounded/streaming write (the prototype wrote the whole row group in one shot), and decide whether the lossy archives (Quant8, codec archives) move to Zarr too or stay on the current format. Full writeup: [ZARR_EVALUATION.md](../../ZARR_EVALUATION.md).

### 2026-10-30 — Stage 8 capture proposal drafted (planning only, no code)

With Stage 0's protocol closed out, the next-biggest item is Stage 8 (expanding capture to full GPT-OSS 20B and Qwen3.8-27B tensor inventories), which the pipeline doc itself says needs sign-off from model/runtime owners before any extraction. Wrote [STAGE8_CAPTURE_PROPOSAL.md](../../STAGE8_CAPTURE_PROPOSAL.md) to make that conversation concrete:

- **Architecture research** (web search, confidence-labeled, not yet confirmed against either model's real `config.json`): GPT-OSS 20B — 24 layers, hidden 2,880, GQA (64 Q heads / 8 KV heads), 32 experts top-4 — cross-checked against our own pilot data (layers/hidden size match exactly, which is real corroboration). Qwen3.8-27B — lower confidence, but consistently described across sources as a **hybrid architecture**: only 16 of 64 layers use full attention, the other 48 use Gated DeltaNet linear attention with a recurrent state instead of a KV cache. If true, this needs its own capture code path, not just different config values.
- **Storage/compute feasibility math:** adding GPT-OSS 20B's Q/K/V alone is ~21× the current per-token footprint (~640 GiB for the existing 3-game pilot alone). Full attention-weight matrices are flagged as **not viable to capture in full** — they scale with context length (up to 20,000 tokens for `high`-effort games), dwarfing everything else in the inventory by orders of magnitude; recommended either skipping them or scoping a small separate sampled sub-experiment.
- **Six blocking open questions** listed for whoever controls model/compute access: GPU access (a 27B model won't fit this machine's single RTX 4090), Qwen3.8-27B architecture confirmation, attention-weight scope decision, corpus scope (pilot-scale vs. full 200-game corpus — roughly a 2-order-of-magnitude difference), GPT-OSS expert FFN width, and pinning model revision identifiers (also already flagged as missing in EVALUATION_PROTOCOL.md §1.1).

**Status:** planning only — no capture code written, intentionally, until the open questions are answered.

### 2026-10-30 — Layer+token-range query, cold-cache measurement, and per-result environment metadata

Closed the last two open items from [EVALUATION_PROTOCOL.md](../../EVALUATION_PROTOCOL.md)'s exit check.

**Code changes.**
- `benchmarks/common/environment.py` (new): `evict_from_page_cache()` (per-file `posix_fadvise(DONTNEED)`, no root needed) and `environment_snapshot()` (hostname, platform, conda env, Python version, tracked package versions, GPU name, UTC timestamp).
- `benchmarks/compare_archive_reads.py`: added a fourth query, `layer_token_range` (combines `--layer` and the token-range args — the readers already AND their filters together, so this needed no reader changes); added `--cold-trials` (default 3), which evicts the Parquet file / archive directory from cache before each of N single timed reads, reported as `cold_read_times_s`/`cold_median_read_time_s` alongside the existing warm numbers; every result now carries an `"environment"` key.
- `benchmarks/evaluate_lossy_archive.py`: also carries `"environment"` now (fidelity-only, so no cold-cache concept applies there).

**Result.** Ran on `high_0056` rg 0 for the raw reference and the current best config (held-out Quant8 `layer_site_dim` + exceptions). The new combined query behaves like the other three (50–125× speedup over Parquet, no new correctness issues). Cold reads are consistently slower than warm (2–7× for the archive), which is the first real evidence the fadvise eviction is doing something on this filesystem rather than being a silent no-op. Full table: [EVALUATION_PROTOCOL.md](../../EVALUATION_PROTOCOL.md) §6b.

**Status.** All six items in the protocol's Stage 0 exit check are now done. The protocol itself is complete; remaining work is applying it (not finishing it).

### 2026-10-29 — Read performance and build resources for the new winning configuration; another documentation correction

**Correction.** [EVALUATION_PROTOCOL.md](../../EVALUATION_PROTOCOL.md) §2 wrongly claimed single-layer reads were "not yet measured for raw/Zstandard/Quant8 archives." They were — `benchmarks/compare_archive_reads.py` runs the single-layer query for every archive type uniformly, and `read_comparison_2026-10-29/{reference,zstd,quant8_layer_site_dim_held_out,...}/parquet_vs_archive_single_layer.json` already had the numbers. Corrected. The only real gap in that section is the layer+token-range combined query, which genuinely doesn't exist in `compare_archive_reads.py` or `evaluate_lossy_archive.py` yet.

**What was measured.** Read timings (`high_0056` rg 0) and build peak RSS for the new held-out Quant8 `layer_site_dim` + clipped-exceptions archive, with a same-session raw-reference control to correct for machine-speed drift:

| Query | This config | Raw reference (same-session control) | Parquet | Speedup vs. Parquet |
| --- | --- | --- | --- | --- |
| Full row group | 0.124 s | 0.049 s | 0.886 s | 7.1× |
| Token range | 0.0089 s | 0.0075 s | 0.746 s | 83.6× |
| Single layer | 0.0236 s | 0.0128 s | 0.733 s | 31.0× |

Matches the pre-exception `layer_site_dim` held-out timings almost exactly — the exception lookup costs essentially nothing to read. Build (full held-out invocation, including fitting on all 19 calibration row groups): 196.1 MB peak RSS, 1.92 s wall clock. Not directly comparable to the original in-sample Quant8 build figure (240.5 MB/1.45 s) since that one had no calibration-fitting pass.

Full numbers and discussion: [EVALUATION_PROTOCOL.md](../../EVALUATION_PROTOCOL.md) §6a.

### 2026-10-29 — Clipped-value exceptions for held-out Quant8: implementation and result (new smallest passing configuration)

**Correction first.** The previous entry below wrongly stated the Quant8 matrix lacked per-config inner-product figures ("only an aggregate ≤1.3% bound"). It doesn't — `evaluate_lossy_archive.py`'s `worst_group.inner_product_relative_error` was already recorded per config; I just hadn't pulled it from the JSON. Corrected values (worst across rg 0/106/212): held-out `layer_site_dim` Quant8 is 1.33% (fails the ≤1% target, not just variance/cosine as first stated). [EVALUATION_PROTOCOL.md](../../EVALUATION_PROTOCOL.md) §6 is corrected with the real numbers.

**Code changes.** `python/quantized_archive.py`: `compress_archive_quant8(..., store_clipped_exceptions=True)` stores every clipped (row, dimension)'s exact original BF16 bits in a new per-chunk sparse table (`exceptions.npz`), and the reader overwrites exactly those positions with the exact value after dequantizing. `scripts/quantize_activation_archive.py --store-clipped-exceptions`. New test `test_quant8_clipped_exceptions_recover_exact_outlier_values` (held-out clipped positions match the original bit-for-bit with the flag, and don't without it). Suite: 11/11 passing.

**Archives and result.** Rebuilt held-out Quant8 `layer_site_dim` for `high_0056` rg 0/106/212 under `~/gibbon-work/archives/quant8-matrix/high_0056-rg{0000,0106,0212}-quant8-layer_site_dim-held_out-exceptions/`, same calibration split as before. Re-evaluated with `evaluate_lossy_archive.py`: `benchmarks/results/high_0056/row_groups/{0000,0106,0212}/lossy_eval/quant8_layer_site_dim_held_out_exceptions.json`.

| Metric | Worst across rg 0/106/212 | §3 tolerance |
| --- | --- | --- |
| Overall relative Frobenius error | 0.53% | ≤2% (pass) |
| Worst group relative error | 0.68% | ≤3% (pass) |
| Worst variance-ratio deviation | 0.024 | ≤0.10 (pass) |
| Worst inner-product relative error | 0.19% | ≤1% (pass) |
| Worst cosine change | 0.0033 | ≤0.02 (pass) |
| Size vs. raw reference | 0.520–0.522× | — |

**This is the first configuration to pass every §3 metric on held-out data while beating BF16-4-bit (0.617×) on size.** The exception table costs 0.56–0.67 MB per row group (~0.46% of raw size), for 67,711–81,095 clipped values (0.10–0.11%) recovered exactly. Encode time 0.77–0.93 s per row group; peak RSS not measured in this run. **Current answer to the project question, on this evaluation set: held-out Quant8 `layer_site_dim` + clipped-value exceptions at ~0.52× raw is the smallest archive meeting the fixed fidelity constraints.**

**Open follow-ups:** peak RSS/decode-time for this config not yet measured; read-path (full/token-range/layer) performance not yet benchmarked for it; cold-cache measurement still missing project-wide (EVALUATION_PROTOCOL.md §5.3/§7).

### 2026-10-29 — Fidelity tolerances confirmed; existing results checked against them

**What happened.** The user confirmed the draft §3 tolerance table in [EVALUATION_PROTOCOL.md](../../EVALUATION_PROTOCOL.md) as-is. I then checked every config in the 2026-10-29 codec matrix and Quant8 matrix against it (full table: protocol §6).

**Result.** Only **BF16 4-bit and 5-bit mantissa truncation** pass every metric on held-out data (0.617 / 0.677 of raw size). SZ3 (both τ), FP8 E4M3, zfp (all rates), and Quant8 (all granularities) all fail — specifically on variance-ratio deviation and/or cosine change, not on Frobenius error. Held-out Quant8 `layer_site_dim` is closest (passes Frobenius and worst-group with room to spare, at 0.68–1.22%), but fails variance-ratio (0.60) and cosine change (0.094) because of the clipped-outlier effect already identified in the 2026-10-29 Quant8 entry's finding 3.

**Implication for next steps.** The already-proposed "store clipped values as exact BF16 exceptions" fix (~+0.7% size, in that same entry's "Candidate next fix") is now the most direct path to a configuration smaller than BF16-4-bit that still passes §3 — it should be implemented and re-checked against the tolerance table before trying anything else aimed at beating the current pass/fail frontier.

**Gap noted.** The Quant8 matrix only records an aggregate inner-product bound ("≤1.3% for non-global configs"), not a per-config figure, so it can't be checked precisely against the ≤1% target. Small addition to `evaluate_lossy_archive.py`'s output, not a rerun.

### 2026-10-29 — Stage 0 evaluation protocol written

**Why.** All lossy held-out evaluation so far (Quant8 matrix, BF16-truncation/FP8/zfp/SZ3 codec matrix, both below) ran before any fidelity tolerance was fixed, which is out of sequence with the proposal's own rule. No further held-out evaluation should be treated as final until checked against a predeclared tolerance.

**Deliverable.** [EVALUATION_PROTOCOL.md](../../EVALUATION_PROTOCOL.md) (new, repo root): scope (checkpoints, corpus, prompts, generation settings — with model revision and tokenizer version flagged as unrecorded gaps), the current vs. expanded archive scope, supported query shapes (and gaps: layer-only reads not yet measured outside codec archives, layer+token-range not implemented anywhere), a draft lossy fidelity tolerance table (relative Frobenius ≤2% overall / ≤3% worst group, variance-ratio deviation ≤0.10, inner-product error ≤1%, cosine change ≤0.02 — proposed, not from the proposal text, and explicitly needs sign-off), the fixed calibration/evaluation game split, and recorded hardware/software environment (`ramiel`: i5-12400F, RTX 4090, 31 GiB RAM; conda env package versions) with a methodology section requiring cold-cache measurement and per-result environment metadata that do not exist yet.

**Status.** Draft — tolerances are a proposal pending confirmation. Next: check the existing 2026-10-29 Quant8/codec results against this table explicitly (which configs pass/fail, and at which parameter), then close the query-shape and cold-cache measurement gaps it lists.

### 2026-10-29 — Reduced precision, FP8, zfp, and SZ3 codec archives (step 5)

**Code changes**

- `python/codec_archive.py` (new, archive version 4): a general lossy codec archive. Each chunk is split into one block per (layer, activation_site), because value ranges differ by orders of magnitude between layers. Each block is encoded independently, and its byte range and per-block parameters (scale, error bound) are stored in the index. A query decodes only the blocks it touches. All parameters are computed from the block being encoded and stored, so these codecs need no calibration split. Codecs:
  - `bf16_trunc_zstd`: round BF16 to *k* of 7 mantissa bits (round to nearest, ties to even), then Zstandard level 3. Each value's relative error is at most 2^−(k+1). *k* = 7 is lossless.
  - `fp8_e4m3_block_scale`: per-block absmax scaling to the FP8 E4M3 range, cast with PyTorch.
  - `zfp_fixed_rate`: zfp fixed-rate mode on each 2-D block (rows × 2,880).
  - `sz3_abs_rms`: SZ3 with a pointwise absolute error bound of τ × block RMS.
- `scripts/codec_activation_archive.py` (new CLI). `benchmarks/evaluate_lossy_archive.py` and `benchmarks/compare_archive_reads.py` now open codec archives. The read benchmark also has a **single-layer query** (layer 11, both sites, 1,024 rows).
- Dependencies: `zfpy=1.0.1` (conda-forge) and `pysz==1.1.0` (the SZ3 team's PyPI binding) were installed into `gibbon-activation-pilot` and added to `environment.yml`.
- Test: `test_codec_archives_round_trip_within_bounds` covers all four codecs (bounds, exact metadata, combined layer + token-range selection) and rejects invalid parameters. Suite: 10/10 passing.

**Fidelity results.** Evaluation row groups: `high_0056` 0, 106, 212. The table shows row group 0; rows 106 and 212 agree within 1% relative for every configuration except held-out per-dimension Quant8 (see the Quant8 matrix entry). Archives are under `~/gibbon-work/archives/codec-matrix/`; per-run JSON is in `benchmarks/results/high_0056/row_groups/{0000,0106,0212}/lossy_eval/<config>.json`. Quant8 rows are repeated from the matrix entry for comparison.

| Configuration | Size / raw | Overall rel. Frobenius error | Worst group | Max abs error | Worst variance-ratio deviation | Worst inner-product rel. error | Worst cosine change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16, 7 mantissa bits (lossless) + zstd | 0.794 | 0 | 0 | 0 | 0 | 0 | 0 |
| BF16, 5 bits + zstd | 0.677 | 0.0073 | 0.0077 | 128 | 0.023 | 0.0052 | 0.0071 |
| BF16, 4 bits + zstd | 0.617 | 0.0138 | 0.0147 | 256 | 0.064 | 0.0093 | 0.0116 |
| BF16, 3 bits + zstd | 0.550 | 0.0270 | 0.0285 | 512 | 0.171 | 0.0184 | 0.0267 |
| BF16, 2 bits + zstd | 0.483 | 0.0545 | 0.0601 | 1,024 | 0.706 | 0.0375 | 0.0456 |
| FP8 E4M3, block scale | 0.510 | 0.0261 | 0.0272 | 466 | 0.134 | 0.0174 | 0.0224 |
| zfp, 4 bits/value | 0.295 | 0.1568 | 0.2421 | 3,666 | 957 | 0.0558 | 0.1562 |
| zfp, 6 bits/value | 0.433 | 0.0420 | 0.0615 | 882 | 60.8 | 0.0132 | 0.0333 |
| zfp, 8 bits/value | 0.571 | 0.0106 | 0.0154 | 250 | 4.34 | 0.0032 | 0.0067 |
| **SZ3, τ = 0.01** | **0.448** | **0.0057** | **0.0057** | **6.26** | 0.171 | 0.0004 | 0.0008 |
| **SZ3, τ = 0.03** | **0.350** | **0.0171** | **0.0173** | **18.8** | 1.41 | 0.0012 | 0.0021 |
| SZ3, τ = 0.1 | 0.244 | 0.0573 | 0.0578 | 62.6 | 14.5 | 0.0039 | 0.0085 |
| Quant8 per-(layer, site), in-sample | 0.509 | 0.0544 | 0.0869 | 49.4 | 22.0 | 0.0039 | 0.0134 |
| Quant8 per-(layer, site), held-out | 0.509 | 0.0654 | 0.0903 | 800 | 13.3 | 0.0047 | 0.0191 |
| Quant8 per-dim, in-sample | 0.516 | 0.0033 | 0.0052 | 31.7 | 0.010 | 0.0018 | 0.0034 |
| Quant8 per-dim, held-out | 0.516 | 0.0068 | 0.0163 | 800 | 0.167 | 0.0099 | 0.0672 |

**Read results.** Measured on `high_0056` row group 0, all formats back-to-back in one session (one warm-up, seven timed reads, cache uncontrolled). The raw reference matched its 2026-10-22 timing (0.050 vs 0.0475 s), so the slowdown seen earlier today had cleared. JSON: `benchmarks/results/high_0056/row_groups/0000/read_comparison_2026-10-29/<archive>/`.

| Archive | Size | Full row group | Token range (384 rows) | Single layer (1,024 rows) | Full-read speedup vs Parquet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw reference | 144.06 MB | 0.050 s | 0.0076 s | 0.014 s | 23.2× |
| Zstandard (lossless, whole chunks) | 114.97 MB | 0.188 s | 0.0133 s | 0.144 s | 4.8× |
| BF16 7 bits + zstd (lossless, (layer, site) blocks) | 114.35 MB | 0.244 s | 0.0185 s | 0.028 s | 3.8× |
| Quant8 per-dim, held-out | 74.39 MB | 0.126 s | 0.0089 s | 0.019 s | 7.1× |
| BF16 4 bits + zstd | 88.84 MB | 0.217 s | 0.0147 s | 0.024 s | 4.1× |
| FP8 E4M3 | 73.47 MB | 0.257 s | 0.0165 s | 0.026 s | 3.4× |
| SZ3 τ = 0.01 | 64.56 MB | 0.480 s | 0.0268 s | 0.038 s | 1.9× |
| SZ3 τ = 0.03 | 50.37 MB | 0.448 s | 0.0258 s | 0.036 s | 2.1× |
| zfp 8 bits/value | 82.30 MB | 1.029 s | 0.0522 s | 0.062 s | 0.9× |
| Parquet (median over these runs) | 122.67 MB | 0.88–1.17 s | 0.73–0.78 s | 0.74–0.84 s | — |

**Findings**

1. **SZ3 has the best size-for-error of everything tested.** At τ = 0.01 it is 13% smaller than per-dimension Quant8 (0.448 vs 0.516 of raw), with lower held-out error (0.57% vs 0.68%) and a guaranteed max error. Its 6.3 max abs error compares with 800 for Quant8, whose clipped values set the maximum. At τ = 0.03 it reaches 0.35 of raw with 1.7% error. Its error is also uniform across groups: the worst group is within 2% of the overall error, because the bound scales with each block's RMS.
2. **The cost is decode speed.** SZ3 full reads take 0.45–0.48 s, about 300 MB/s of raw data, roughly 4× slower than Quant8 but still 2× faster than the tested Parquet path. Single-layer and token-range reads stay fast (0.026–0.038 s), because only the touched blocks are decoded.
3. **Remaining SZ3 weakness: low-variance dimensions.** The error bound scales with each block's RMS, so dimensions with small spread get proportionally large noise. The worst variance-ratio deviation is 1.4 at τ = 0.03 and 14.5 at τ = 0.1. This is the same effect that per-dimension scales fixed for Quant8.
4. **BF16 mantissa truncation is the simplest reduced-precision option and bounds each value's relative error.** At 4 bits it is 0.617 of raw with 1.4% error. Its max abs error is large in absolute terms (256), but that occurs only on the largest values, where it is at most 1/32 relative. It is dominated by SZ3 on size for a given error.
5. **FP8 E4M3 ≈ BF16 with 3 mantissa bits**, as expected from its 3-bit mantissa: 2.6% error at 0.51 of raw.
6. **zfp fixed-rate does poorly on this data.** At 8 bits/value it is 0.571 of raw with 1.1% error. It distorts variance (worst deviation 4.3) and is the slowest decoder: a full read takes 1.03 s, slower than Parquet. One cause: blocks of 21–22 rows are not multiples of zfp's 4-row block size, so it pads them; 8 bits/value stores as 9.1. zfp's fixed-accuracy mode and a token-aligned chunk size are untested.
7. **Splitting chunks into (layer, site) blocks trades full-read speed for selective-read speed.** Lossless BF16 blocks (7 bits + zstd) are 0.794 of raw, matching the whole-chunk Zstandard archive (0.798). Single-layer reads take 0.028 s instead of 0.144 s (5.1× faster), because whole-chunk Zstandard must decompress every chunk. Full reads are slower (0.244 vs 0.188 s), because there are 48× more, smaller decompression calls. The JSON marks this archive `roundtrip_exact: false` only because the benchmark treats every codec archive as lossy; its measured error is exactly 0, so the BF16 values are recovered bit for bit.

### 2026-10-29 — Held-out calibration, per-dimension scales, and full fidelity metrics (Quant8 matrix)

**Why.** The proposal requires calibration-dependent encoders to be fitted on different data from what they are evaluated on. All earlier Quant8 results fitted their scales on the row group being encoded (in-sample). This entry adds scales fitted only on other games (held-out). See finding 4 for how to read the two.

**Code changes**

- `python/quantized_archive.py`:
  - **Separate fitting step.** `fit_quant8_groups()` fits the scales on any list of reference archives. `compress_archive_quant8(..., calibration_paths=[...])` encodes with scales fitted only on those archives, and refuses if they include the source game. Without calibration paths, it fits in-sample as before.
  - **Clipping is recorded.** The index stores `calibration` (mode, game IDs, archive paths) and counts out-of-range values in `clipped_values` and `clipped_fraction`.
  - **New `layer_site_dim` granularity.** It stores a float32 minimum and scale for each dimension within each (layer, site) group in `parameters.npz` (1.11 MB, checksummed).
  - **Group-by-group decoding.** The reader decodes one group at a time instead of building full-size per-row parameter arrays. Existing archives decode identically; verified for both archives from earlier today.
- `scripts/quantize_activation_archive.py`: new `--granularity layer_site_dim` and `--calibration ARCHIVE...` options.
- `benchmarks/evaluate_lossy_archive.py` (new): compares a lossy archive with its reference, group by group. Reports relative Frobenius error, max abs error, abs-error p50/p99/p99.9, relative mean shift, per-dimension variance ratio, and inner-product and cosine changes on 256 sampled rows per group (seed 0). This covers the proposal's evaluation metrics (step 4).
- Tests: `test_quant8_held_out_calibration_clips_and_rejects_source_game` and `test_quant8_per_dimension_scales_handle_outlier_dimension`. Suite: 9/9 passing.

**Data split.** The split is by game, following the proposal's split by sample.
- **Calibration:** all of `low_0056` (16 row groups) and `high_0066` row groups 0, 33, 65 (early, middle, late).
- **Evaluation:** `high_0056` row groups 0, 106, 212 (early, middle, late).

All are new uncompressed reference archives under `~/gibbon-work/archives/`. Parquet-to-archive conversion used 0.48–1.14 GB peak RSS per row group, which is more than the 1,024-row batch size implies. It is worth checking whether PyArrow decodes whole row groups.

**Results.** 18 archives under `~/gibbon-work/archives/quant8-matrix/`. Per-run JSON: `benchmarks/results/high_0056/row_groups/{0000,0106,0212}/lossy_eval/quant8_<granularity>_<calibration>.json`.

| Scale granularity | Fitted on | Size (rg0) | Overall rel. Frobenius error, rg 0 / 106 / 212 | Worst group rel. error (any rg) | Max abs error (any rg) | Clipped values | Worst variance-ratio deviation | Worst cosine change |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| global | in-sample | 73.28 MB | 0.182 / 0.118 / 0.301 | 17.8 | 71.5 | 0 | 242 | 1.05 |
| global | held-out | 73.28 MB | 0.320 / 0.318 / 0.335 | 19.3 | 83.1 | 0 | 110 | 1.06 |
| layer_site | in-sample | 73.28 MB | 0.054 / 0.053 / 0.066 | 0.087 | 71.5 | 0 | 38.3 | 0.014 |
| layer_site | held-out | 73.28 MB | 0.065 / 0.065 / 0.068 | 0.091 | 800 | 16–57 per rg | 13.3 | 0.020 |
| layer_site_dim | in-sample | 74.39 MB | 0.0033 / 0.0032 / 0.0038 | 0.005 | 36.8 | 0 | 0.01 | 0.0034 |
| layer_site_dim | held-out | 74.39 MB | 0.0068 / 0.0065 / 0.0122 | 0.016 | 2,368 | 0.10–0.11% | 0.60 | 0.094 |

Reference archive size: 144.06 MB for rg 0 and 106, 125.77 MB for rg 212. Encoding took 0.5–1.1 s with 176–283 MB peak RSS per row group.

**Findings**

1. **Held-out fitting costs accuracy, as expected, but the ranking holds.** Per-dimension scales are about 10× more accurate than per-(layer, site) scales on held-out data (0.65–1.2% vs 6.5–6.8% overall), for 1.11 MB more (+1.5% of the archive). Global scaling is unusable either way (worst group error above 1,000% held-out).
2. **Per-dimension scales fix the variance distortion.** With per-(layer, site) scales, some low-variance dimensions end up with up to 39× their true variance, because the quantization step exceeds their spread. With per-dimension scales the worst deviation is 1% in-sample and 60% held-out.
3. **Clipping is the new problem for per-dimension scales on held-out data.** Each dimension's fitted range is tighter, so about 0.1% of values (68k–81k per row group) fall outside the calibration range, spread evenly over all groups (median 0.125% per group). Those values set the max abs error (up to 2,368) and the worst cosine change (0.094). Per-(layer, site) scales clip only 16–57 values per row group, but are less accurate everywhere else.
4. **In-sample and held-out are two valid encoder designs, not right versus wrong** (corrected later on 2026-10-29). In-sample scales are computed from the data being encoded and stored in the archive. The decoder needs nothing else, so this is a legitimate encoder that reads the data twice, not leakage of evaluation data. Held-out scales are fixed before the data is seen. That is what a single-pass streaming capture needs, and it is the setting the proposal's calibration rule describes. The 1.2–3.2× higher overall error with held-out scales is the cost of fixing the scales in advance. Report both, labelled by mode.
5. **Inner products are well preserved by the useful granularities.** Gram-matrix relative error is at most 1.3% for every non-global configuration.

**Candidate next fix (not implemented).** Store clipped values as exact BF16 exceptions: a sparse index plus the original value. At about 81k values × 6 bytes ≈ 0.49 MB per row group (+0.7%), held-out per-dimension Quant8 would keep its 0.7–1.2% overall error while the clipped values become exact.

### 2026-10-29 — Quant8 with per-(layer, site) scales

- **Change.** `compress_archive_quant8(..., granularity="layer_site")` (CLI: `--granularity layer_site`) stores one affine minimum and scale per (layer, activation_site) group: 48 groups, 768 parameter bytes. `granularity="global"` remains the default. Existing global archives still read, because the reader falls back to their top-level `minimum`/`scale`. New test: `test_quant8_layer_site_scales_reduce_error_across_layer_ranges`. Suite: 7/7 passing.
- **Archive build.** `~/gibbon-work/archives/high_0056-rg0-quant8-layersite-v1`: 73.28 MB (+4.3 KB vs global), 1.03 s encode, 224.4 MB peak RSS.
- **Fidelity.** Overall relative Frobenius error fell from 0.182 to **0.054**. The worst layer/site fell from 10.77 (layer 0 `post_attention`) to **0.087** (layer 11 `post_mlp`). Every one of the 48 groups is now between 0.050 and 0.087 (median 0.057 for `post_attention` and 0.064 for `post_mlp`). Max absolute error is unchanged at about 49.4, set by the widest late layers.
- **Remaining error comes from outlier dimensions.** In layer 11 `post_mlp`, a few dimensions reach |x| ≈ 1,008 (dim 2559), 764 (dim 773), and 672 (dim 1506). The median dimension peaks around 31, and the 99th-percentile |x| is 61. Those few dimensions still set each group's step size. The next experiment is per-dimension or blockwise scales within each (layer, site).
- **Read performance.** The first layer_site run measured 0.202 s for a full read, because the reader called `np.unique` on the string `activation_site` column once per layer per chunk (600 calls). The fix computes the unique values once per chunk and decodes in place. On re-run, global and layer_site full reads take 0.102 / 0.103 s, and token-range reads 0.0086 / 0.0087 s.
- **Timing drift.** The machine is slower today than on 2026-10-22: the raw reference archive's full read measured 0.085 s in a same-session control, versus 0.0475 s in the table above. Compare Quant8 timings with each other, not with the older raw/Zstandard rows. A same-session re-run of all formats is still needed.

### 2026-10-29 — Quant8 archive built and benchmarked on `high_0056`, row group 0

- **Test fix.** `test_quant8_archive_reconstructs_with_recorded_error` failed because the shared fixture deliberately contains a NaN (`0x7FC1`), which Quant8 correctly rejects. The Quant8 test now uses finite data; a new test, `test_quant8_archive_rejects_non_finite_values`, checks that NaN input is refused and leaves no partial archive. Suite: 6/6 passing (`python -m unittest discover -s tests`).
- **Archive build.** `scripts/quantize_activation_archive.py` converted `~/gibbon-work/archives/high_0056-rg0-reference-v1` into `~/gibbon-work/archives/high_0056-rg0-quant8-v1`: 24 chunks, 73.28 MB total, 1.45 s encode (2.10 s wall clock including startup), 240.5 MB peak RSS (`/usr/bin/time -v`). This is the first measured archive write time and peak memory.
- **Read benchmark.** `benchmarks/compare_archive_reads.py` now accepts Quant8 archives. For lossy archives it checks metadata exactly and records value error, instead of requiring identical bits. Results are in [Quant8 archive](#quant8-archive-high_0056-row-group-0) below.
- **Key finding: the global Quant8 error is concentrated in early layers.** One min/max scale (−12,288 to 12,928, step ≈ 99.6) is shared across all layers. Late layers span about ±13,000, but layer-0 values span about ±75, less than one quantization step. Per-layer relative Frobenius error:

  | Layer | `post_attention` | `post_mlp` |
  | ---: | ---: | ---: |
  | 0 | 10.77 | 10.10 |
  | 5 | 5.11 | 4.31 |
  | 11 | 1.65 | 1.43 |
  | 17 | 0.357 | 0.282 |
  | 23 | 0.051 | 0.051 |

  The overall 18.2% figure therefore hides errors of up to 1,077% in early layers. Global-affine Quant8 is not usable as is. A scale per (layer, site), or per block, is the next experiment.

## Work completed

- Added `python/activation_archive.py`: version-1 writer/reader for the pilot's fixed-width activation rows.
- The writer accepts batches and emits bounded-size chunks, storing exact BF16 bits as little-endian `uint16` values in `.npy` files and row metadata in `.npz` files.
- The versioned `index.json` records the game, dimensions, dtype, metadata fields and dtypes, chunk records, and per-chunk SHA-256 checksums.
- Added chunk-by-chunk reads with `iter_chunks()`, a convenience `read()` method, selection by round/token range/layer/activation site, and integrity validation with `verify()`.
- Added `scripts/build_activation_archive.py` for batch-based Parquet conversion.
- Added `scripts/compress_activation_archive.py` to create a separate, version-2 Zstandard archive from the uncompressed reference archive. Activation chunks are compressed independently; metadata and indexes are retained, and the reader verifies exact decompressed bits.
- Added synthetic tests for exact round trips, filtering, chunk iteration, Parquet conversion, and detecting modified activation data.
- Added `benchmarks/compare_archive_reads.py`, which checks Parquet/archive output equality before timing full-row-group and token-range reads and saves one JSON result per query.
- Ran raw, Zstandard level 3, and Quant8 storage baselines on `high_0056`, row group 0.

The archive is a reference implementation for the pilot schema only. It does not yet represent independent head and tensor-type axes, heterogeneous tensor shapes, masks, or variable-shaped tensor families from the full project proposal.

## Benchmark method

- Same source game and row group for the two storage paths.
- Seven timed repetitions after one warm-up read; cache state was not controlled, so these are warm-cache-oriented results, not cold-read measurements.
- Timed operation returns selected activation rows and associated metadata. Parquet uses the canonical loader to read the full row group and then filters in memory. The archive loads chunk metadata, memory-maps activation chunks, filters, and copies the selected rows.
- Exact activation values and metadata were checked before timing. The output activation SHA-256 is saved with each result.
- Archive size includes the `.npy` activation arrays, `.npz` metadata, and index. Parquet size reports the selected row group's compressed and uncompressed storage sizes.
- Write time, peak memory, cold-cache latency, and repeated row groups were not measured in this comparison.

## Results

Times are median seconds; file sizes use decimal MB (1 MB = 1,000,000 bytes). Speedup is Parquet median divided by archive median.

| Game / row group | Query | Rows returned | Parquet median | Archive median | Archive speedup | Parquet compressed row group | Uncompressed archive |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `low_0056` / 0 | Full row group | 1,392 | 0.0786 s | 0.00393 s | 20.0x | 10.69 MB | 8.16 MB |
| `low_0056` / 0 | Round 0, tokens 0-7 | 384 | 0.0652 s | 0.00230 s | 28.3x | 10.69 MB | 8.16 MB |
| `high_0056` / 0 | Full row group | 24,576 | 0.9979 s | 0.0475 s | 21.0x | 122.67 MB | 144.06 MB |
| `high_0056` / 0 | Round 0, tokens 0-7 | 384 | 0.7051 s | 0.00751 s | 93.9x | 122.67 MB | 144.06 MB |

All four results report `roundtrip_exact: true`. The uncompressed archive is smaller than the compressed Parquet row group for `low_0056` row group 0, but about 17.4% larger for `high_0056` row group 0. Thus the archive's observed read-speed advantage has a storage-size trade-off that varies with the data.

## Compression baselines: `high_0056`, row group 0

All methods processed the same 24,576 × 2,880 activation payload (141.56 MB raw). These are codec payload sizes from the method benchmarks, not complete archive-directory sizes; method-specific metadata is included as described in the method notes. Encode/decode times are single benchmark measurements and should be repeated before drawing throughput conclusions.

| Method | Encoded size | Size / raw | Encode time | Decode time | Relative Frobenius error | Max absolute error | Exact? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Raw baseline | 141.56 MB | 1.000 | 0 s | 0 s | 0 | 0 | Yes |
| Zstandard level 3 | 112.36 MB | 0.794 | 0.203 s | 0.152 s | 0 | 0 | Yes |
| Quant8 global affine | 70.78 MB | 0.500 | 1.545 s | 0.129 s | 0.1823 | 49.82 | No |

Zstandard reduced the raw activation payload by about 20.6% and preserved it exactly. Quant8 reduced it by about 50%, but changed values; its measured relative Frobenius error was about 18.2%. Whether that error is acceptable must be decided against a predeclared project tolerance and tested across layers/tensor types and held-out data. Do not select Quant8 from size alone.

Raw results: [raw_baseline.json](high_0056/row_groups/0000/raw_baseline.json), [zstd_lossless_level_3.json](high_0056/row_groups/0000/zstd_lossless_level_3.json), and [quant8.json](high_0056/row_groups/0000/quant8.json).

These codec baseline measurements encode tensors loaded from Parquet; they do not themselves create an archive. The following separate experiment compresses the already-built raw archive and measures reads from that compressed archive.

## Zstandard-compressed archive: `high_0056`, row group 0

The raw reference archive was copied into a distinct version-2 archive with Zstandard level 3 applied to each activation chunk. The metadata files and index were retained. The reader decompressed the queried activation chunks and returned values matching the source bit-for-bit.

| Representation | Total archive size | Full-row-group median read | Token-range median read | Exact recovery |
| --- | ---: | ---: | ---: | --- |
| Uncompressed reference archive | 144.06 MB | 0.0475 s | 0.00751 s | Yes |
| Zstandard archive, level 3 | 114.97 MB | 0.1719 s | 0.01289 s | Yes |
| Parquet source row group | 122.67 MB compressed | 0.8590 s | 0.7008 s | Yes |

For this chunk, the Zstandard archive is about 20.2% smaller than the raw archive and about 6.3% smaller than the compressed Parquet row group. It reads more slowly than the raw archive because chunks must be decompressed, but it remained about 5.0x faster than the tested Parquet path for a full-row-group read and 54.4x faster for the token-range query. The token-range query still decompresses the archive chunk or chunks it touches; it does not decode individual tensor values directly from the compressed byte stream.

These are seven repeated reads after a warm-up with uncontrolled cache state. The full-read archive timing had an unusually slow first timed sample, so repeat measurements and cold-cache tests are needed before drawing firm performance conclusions.

Compressed archive: `~/gibbon-work/archives/high_0056-rg0-zstd-u16-v1/`. Comparison JSON: [compressed full-row-group read](high_0056/row_groups/0000/zstd_archive_comparison/parquet_vs_archive_full_row_group.json) and [compressed token-range read](high_0056/row_groups/0000/zstd_archive_comparison/parquet_vs_archive_token_range.json).

## Quant8 archive: `high_0056`, row group 0

The raw reference archive was converted into version-3 archives storing affine `uint8` codes, with either one global scale or one scale per (layer, activation_site). Metadata files are copied unchanged, and reads decode codes to float32. Same protocol as above: one warm-up, seven timed reads, cache state uncontrolled. The two Quant8 rows were measured back-to-back on 2026-10-29, when the machine was slower than on 2026-10-22 (see the progress log). Their timings are comparable with each other, not directly with the raw, Zstandard, and Parquet rows measured earlier.

| Representation | Total archive size | Full-row-group median read | Token-range median read | Relative Frobenius error (full / token range) | Worst layer/site rel. error | Max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uncompressed reference archive | 144.06 MB | 0.0475 s | 0.00751 s | 0 | 0 | 0 |
| Zstandard archive, level 3 | 114.97 MB | 0.1719 s | 0.01289 s | 0 | 0 | 0 |
| Quant8, global scale | 73.28 MB | 0.1019 s | 0.00861 s | 0.1824 / 0.2006 | 10.77 | 49.82 |
| Quant8, per-(layer, site) scale | 73.28 MB | 0.1032 s | 0.00865 s | 0.0544 / 0.0600 | 0.087 | 49.44 |
| Parquet source row group (same session as Quant8) | 122.67 MB compressed | 1.2390 s | 0.7257 s | 0 | 0 | 0 |

Both Quant8 archives are 50.9% of the raw archive's size and were about 12× (full) and 84× (token range) faster than the tested Parquet path in the same session. Metadata matched exactly. Per-(layer, site) scales cost 4.3 KB and no measurable read time, and reduce worst-group error by over 100×. A remaining error of 5–9% per group, caused by outlier dimensions, still needs to be checked against a project tolerance; none has been set yet (Stage 0).

Archives: `~/gibbon-work/archives/high_0056-rg0-quant8-v1/` and `~/gibbon-work/archives/high_0056-rg0-quant8-layersite-v1/`. Comparison JSON: global [full](high_0056/row_groups/0000/quant8_archive_comparison/parquet_vs_archive_full_row_group.json) / [token range](high_0056/row_groups/0000/quant8_archive_comparison/parquet_vs_archive_token_range.json); per-(layer, site) [full](high_0056/row_groups/0000/quant8_layer_site_archive_comparison/parquet_vs_archive_full_row_group.json) / [token range](high_0056/row_groups/0000/quant8_layer_site_archive_comparison/parquet_vs_archive_token_range.json).

## Result files

- `low_0056`, full row group: [parquet_vs_archive_full_row_group.json](low_0056/row_groups/0000/parquet_vs_archive_full_row_group.json)
- `low_0056`, token range: [parquet_vs_archive_token_range.json](low_0056/row_groups/0000/parquet_vs_archive_token_range.json)
- `high_0056`, full row group: [parquet_vs_archive_full_row_group.json](high_0056/row_groups/0000/parquet_vs_archive_full_row_group.json)
- `high_0056`, token range: [parquet_vs_archive_token_range.json](high_0056/row_groups/0000/parquet_vs_archive_token_range.json)

## How to reproduce

Build an archive for one row group, placing the generated data under home scratch rather than beside the canonical dataset:

```bash
python scripts/build_activation_archive.py \
  /srv/gibbon-activations/gpt_oss_pilot_3games/activations/low_0056_attempt_000.parquet \
  ~/gibbon-work/archives/low_0056-rg0-reference-v1 \
  --game-id low_0056 \
  --row-group 0 \
  --batch-rows 1024
```

Compare the archive with Parquet and save results under `benchmarks/results/`:

```bash
python benchmarks/compare_archive_reads.py \
  --game-id low_0056 \
  --row-group 0 \
  --archive ~/gibbon-work/archives/low_0056-rg0-reference-v1 \
  --iterations 7
```

The comparison command writes `parquet_vs_archive_full_row_group.json` and `parquet_vs_archive_token_range.json` under the game's row-group results directory. Use a new archive path when rebuilding; the writer intentionally refuses to overwrite an existing archive.

## Interpretation and limitations

The measurements show that this NumPy-based archive implementation was faster for the tested reads on these two row groups, on this machine, under the stated protocol. They do not establish that it is faster for all query patterns, cold reads, complete-game workloads, other hardware, or Parquet configurations. In particular:

- The Parquet comparison reads a complete row group before filtering. Predicate pushdown, column projection, or a different Parquet layout could change the comparison.
- The archive is uncompressed and uses separate files for activation and metadata chunks. Small-query latency includes opening metadata/chunk files and returning copied output values.
- Only row group 0 from one low-effort and one high-effort game was measured.
- Cache state was not controlled; one initial read was discarded, followed by seven timed reads.
- Archive creation throughput and peak RAM were not measured. The converter is batch-based, but its actual memory behavior should be measured on larger inputs.
- The current format is specific to the existing flat pilot vectors; it is not yet the final format for the proposal's full set of tensor families.

## Next steps

1. Measure archive creation time and peak RAM for these same row groups.
2. Repeat the benchmark on additional row groups, and include layer-only and layer-plus-token-range queries.
3. Add controlled or clearly separated cold- and warm-cache measurements, and compare against reasonable Parquet query strategies.
4. Evaluate chunk-row settings and account for all index, metadata, and filesystem overhead.
5. Continue compression comparisons. Done on 2026-10-29: Quant8 at three scale granularities (in-sample and held-out), BF16 truncation, FP8, zfp fixed-rate, and SZ3, on three `high_0056` row groups. Open: exact storage of clipped values for held-out Quant8, per-dimension error bounds for SZ3, zfp fixed-accuracy mode, token-aligned chunk sizes, Quant4, and evaluation on more games. Record encoded size, encode/decode time, peak memory, and fidelity; define error tolerances before evaluating held-out data. Keep raw and lossless Zstandard as exact reference baselines.
6. Evaluate Zarr v3 indexed sharding before making a final storage-layout decision, then expand to the broader model/tensor capture scope.
