# Stage 0 Evaluation Protocol

**Status:** draft, 2026-10-29. This is the Stage 0 deliverable required by
[PROJECT_PIPELINE.md](PROJECT_PIPELINE.md): scope, held-out split, supported
queries, fidelity limits, and benchmark hardware/methodology, fixed before
further held-out evaluation.

**Important retroactive note.** Several held-out lossy evaluations were
already run on 2026-10-29 (Quant8 at three granularities, BF16 mantissa
truncation, FP8 E4M3, zfp fixed-rate, SZ3) before this document existed, in
violation of the proposal's own sequencing rule ("fix tolerances before
evaluating held-out data"). Those results are not invalidated as
measurements, but none of them have been checked against a predeclared
tolerance — because none existed. Section 3 below is the first predeclared
tolerance table; all existing and future lossy results must be checked
against it explicitly, and the comparison must be recorded in the benchmark
report.

## 1. Scope

### 1.1 Model checkpoints and corpus (current pilot)

| Item | Value |
| --- | --- |
| Model | `openai/gpt-oss-20b` |
| Model revision | **not pinned** — the dataset manifest records the HF model id but no revision/commit hash. Gap: record the exact revision (or weight checksum) for any future capture. |
| Source run | `gpt_oss_full_200_seed0_20260602` |
| Pilot subset | `gpt_oss_pilot_3games` (manifest `format_version` 1, created 2026-10-01) |
| Games in scope | `low_0056` (low effort), `high_0056` (high effort), `high_0066` (high effort) |
| Prompt version | 0 |
| Prompts per game | 16 (rounds 0–14, plus one `final_decision` prompt at index 15) |
| Generation settings | `max_new_tokens=20000`, `n_avg=5`, `base_seed=0`, `reasoning_efforts=[low, high]` |
| Tokenizer version | **not separately recorded** in the current manifests. Gap: record it explicitly for any future capture. |
| Checksums | Per-file SHA-256 in `manifest.json`, verified against NCC after transfer |

### 1.2 Archive scope

- **First archive scope (current, Stages 2–7):** the existing pilot exactly as captured — `post_attention` and `post_mlp` sites, 24 layers (`0`–`23`), 2,880 BF16 dimensions per row, flat per-token rows, no head axis. This is what every method implemented so far (raw, Zstandard, Quant8, BF16-truncation, FP8, zfp, SZ3) encodes.
- **Expanded capture scope (future, Stage 8):** GPT-OSS 20B and Qwen3.8-27B, each with its own tensor inventory (residual streams, Q/K/V, attention weights, per-head outputs, intermediate feed-forward activations, routing assignments/scores, recurrent states where applicable). Not yet defined in architecture-specific detail and not started — checkpoints are not yet frozen for this scope. Do not begin Stage 8 capture until this section is filled in with per-architecture tensor lists and capture points, per PROJECT_PIPELINE.md Stage 8.

## 2. Supported query shapes

Within the current pilot archive format, every tested archive variant (raw, Zstandard, Quant8, codec archives) must support, and has been exercised against:

| Query shape | Status |
| --- | --- |
| Complete sample (full row group) | Supported, measured for every archive variant |
| Token range | Supported, measured for every archive variant |
| Single layer (both sites) | Supported and measured for every archive type (`compare_archive_reads.py`'s query list applies uniformly) — **correcting an earlier claim in this document that said it was codec-archive-only; it isn't, see `read_comparison_2026-10-29/` results for reference, Zstandard, and Quant8** |
| Layer + token range | Implemented 2026-10-30 as a fourth query (`layer_token_range`) in `compare_archive_reads.py`, supported uniformly by every archive type since the underlying readers already AND together whatever filters are given. Measured for the reference archive and the current winning Quant8 config; see §6b. |
| Head / tensor-type selection | **Out of scope for the current pilot** — there is no head axis or separate tensor type in the flat `post_attention`/`post_mlp` rows. This becomes required once Stage 8 expands capture. |

Required combination for Stage 9 sign-off: complete sample, token range, and single layer must all work, exactly, on every method that claims archive support — not just on the reference format.

## 3. Fidelity targets (confirmed 2026-10-29)

**Lossless codecs** (raw reference, Zstandard, BF16 7-bit "truncation"): exact
bitwise recovery of the original BF16 bit pattern is the requirement, checked
via `roundtrip_exact` / SHA-256 of the decoded payload. No tolerance is
applicable — any mismatch is a failure.

**Lossy codecs**: the following table was derived from the magnitude of
effects we'd want preserved for downstream interpretability use (mean/variance
structure, pairwise geometry), not from any number stated in the proposal
itself, and confirmed by the user on 2026-10-29 as the acceptance bar for
Stage 7/9. Checked against the 2026-10-29 codec matrix (§6), it is not a
rubber stamp: BF16 4-bit/5-bit truncation pass every metric, while SZ3 (both
τ), FP8, zfp-8bit, and per-dimension Quant8 all fail, specifically on
variance-ratio deviation or cosine change rather than on Frobenius error —
i.e. the "sophisticated" codecs currently distort variance/geometry in ways
plain mantissa truncation does not, even though they have lower raw error.

| Metric | Proposed tolerance | Basis |
| --- | --- | --- |
| Overall relative Frobenius error, per (game, row group) | ≤ 2% | SZ3 τ=0.01 (0.57%) and per-dim Quant8 held-out (0.68%) both clear this with margin; FP8/zfp-8bit (~1.1–2.6%) sit near the edge, which is the point of a real bar |
| Worst stratified group (by layer × site) relative error | ≤ 3% | Allows the observed worst-group inflation (e.g. SZ3 τ=0.01 worst group ≈ overall) without passing badly-behaved outlier layers by averaging |
| Max absolute error | Report only, no fixed bound | Activation magnitude varies by orders of magnitude across layers; a single global bound is not meaningful. Use per-layer max-abs normalized by that layer's RMS instead, once that stratified statistic is computed |
| Worst per-dimension variance-ratio deviation | ≤ 0.10 | SZ3 τ=0.01 (0.17) and τ=0.03 (1.41) both exceed this at the low-variance tail — this is intentionally a real constraint, not one every current method passes |
| Worst pairwise inner-product relative error | ≤ 1% | All SZ3/Quant8-per-dim results already clear this; FP8/zfp-8bit (1.3–1.7%) sit at the edge |
| Worst cosine-similarity change | ≤ 0.02 | Matches the scale of structural changes meaningful for downstream geometric analysis |

**Calibration rule (fixed, not draft):** any encoder that fits scales,
codebooks, or prediction parameters must be fit only on the calibration split
below, and the fitted parameters must be applied unchanged to evaluation
samples. This is already implemented for Quant8 (`calibration_paths`) and
needs no further change; SZ3/zfp/FP8/BF16-truncation currently derive all
parameters from the block being encoded, so no calibration split applies to
them.

## 4. Calibration / evaluation split

Fixed by game id (not row group), to avoid leakage through within-game
autocorrelation:

- **Calibration:** all of `low_0056` (16 row groups) + `high_0066` row groups 0, 33, 65 (early/middle/late).
- **Evaluation (held out):** `high_0056` row groups 0, 106, 212 (early/middle/late).

This is the split already used for the 2026-10-29 Quant8/codec matrix; this
document fixes it as the standing split for the current 3-game pilot rather
than a one-off choice. When the dataset expands beyond 3 games (Stage 8), the
split must be redone at the game level, stratified by `reasoning_effort`
(`low`/`high`) so both strata are represented in calibration and evaluation.

## 5. Benchmark hardware, environment, and methodology

### 5.1 Hardware (recorded 2026-10-29, on `ramiel`)

| Item | Value |
| --- | --- |
| Host | `ramiel` |
| OS / kernel | Ubuntu 24.04, Linux 6.11.0-25-generic x86_64 |
| CPU | Intel Core i5-12400F |
| RAM | 31 GiB |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, driver 550.144.03 |

Earlier results in the benchmark report (2026-10-22 vs 2026-10-29) already
show same-machine timing drift of ~2× on the raw reference read. Going
forward, every benchmark run must record a timestamp and a same-session
control measurement (e.g. re-time the raw reference archive) so cross-date
comparisons can be scaled or discarded rather than compared raw.

### 5.2 Software environment

Conda env `gibbon-activation-pilot` (`environment.yml`): Python 3.12.3,
numpy 2.4.6, pandas 3.0.6, pyarrow 25.0.0, zstandard 0.25.0, torch 2.5.1
(CUDA 12.4, available), zfpy 1.0.1, pysz 1.1.0, R 4.4.

### 5.3 Benchmark methodology (fixed going forward)

- **Warm-cache measurement:** one discarded warm-up read, then ≥7 timed
  repeats; report median and spread (not a single number), as the existing
  `compare_archive_reads.py` protocol already does — extend it to report
  spread, not just the figures currently in the report tables.
- **Cold-cache measurement:** implemented 2026-10-30 in
  `benchmarks/common/environment.py` (`evict_from_page_cache`, using
  `posix_fadvise(DONTNEED)` per file — no root needed) and wired into
  `compare_archive_reads.py` as `--cold-trials` (default 3): evict, then one
  single timed read, repeated across trials, reported separately from the
  warm numbers as `cold_read_times_s`/`cold_median_read_time_s`. This is
  best-effort (a no-op on filesystems where fadvise doesn't evict); the
  2026-10-30 run shows cold consistently slower than warm for both Parquet
  and the archive (e.g. raw-reference full-row-group archive read: 0.166 s
  cold vs. 0.049 s warm), which is evidence the eviction is actually working
  on this machine's filesystem, not just a label. `evaluate_lossy_archive.py`
  measures fidelity, not timing, so cold-cache does not apply to it.
- **Every benchmark JSON result must record environment metadata:**
  implemented 2026-10-30 as `environment_snapshot()` in the same module —
  hostname, platform, conda env, Python version, the tracked package
  versions, GPU name/availability, and a UTC timestamp — added as an
  `"environment"` key to every result from `compare_archive_reads.py` and
  `evaluate_lossy_archive.py`. Not yet added to the older `benchmarks/methods/*.py`
  placeholder scripts, which remain unused prototypes (see `benchmarks/methods/README.md`).
- **Row-group and game coverage:** fidelity results must cover all three
  evaluation row groups (0, 106, 212) before being reported as a headline
  number; a single-row-group number is a pilot check, not a result, and
  should be labeled as such in the report.

## 6. Retroactive check against the 2026-10-29 results

Checked against §3 on 2026-10-29. Figures are from the existing report tables
(codec matrix and Quant8 matrix); this is a pass/fail re-read, not a rerun.

**Codec matrix** (`high_0056` rg 0/106/212):

| Config | Rel. Frobenius (≤2%) | Worst group (≤3%) | Variance-ratio dev. (≤0.10) | Inner-product (≤1%) | Cosine change (≤0.02) | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 7-bit (lossless) | — | — | — | — | — | exact, not a lossy case |
| BF16 5-bit | 0.73% pass | 0.77% pass | 0.023 pass | 0.52% pass | 0.71% pass | **PASS** |
| BF16 4-bit | 1.38% pass | 1.47% pass | 0.064 pass | 0.93% pass | 1.16% pass | **PASS** |
| BF16 3-bit | 2.70% fail | 2.85% fail | 0.171 fail | 1.84% fail | 2.67% fail | FAIL |
| BF16 2-bit | 5.45% fail | 6.01% fail | 0.706 fail | 3.75% fail | 4.56% fail | FAIL |
| FP8 E4M3 | 2.61% fail | 2.72% fail | 0.134 fail | 1.74% fail | 2.24% fail | FAIL |
| zfp 8 bits/value | 1.06% pass | 1.54% pass | 4.34 fail | 0.32% pass | 0.67% pass | FAIL (variance) |
| zfp 6 bits/value | 4.20% fail | 6.15% fail | 60.8 fail | 1.32% fail | 3.33% fail | FAIL |
| zfp 4 bits/value | 15.68% fail | 24.21% fail | 957 fail | 5.58% fail | 15.62% fail | FAIL |
| SZ3 τ=0.01 | 0.57% pass | 0.57% pass | 0.171 fail | 0.04% pass | 0.08% pass | FAIL (variance) |
| SZ3 τ=0.03 | 1.71% pass | 1.73% pass | 1.41 fail | 0.12% pass | 0.21% pass | FAIL (variance) |
| SZ3 τ=0.1 | 5.73% fail | 5.78% fail | 14.5 fail | 0.39% pass | 0.85% pass | FAIL |

**Quant8 matrix** (`high_0056` rg 0/106/212; worst value across all three row groups, read directly from the existing per-config `lossy_eval/*.json` files, not from the summary markdown table, which omitted the inner-product column for this matrix even though the JSON already has it):

| Config | Rel. Frobenius (≤2%, worst rg) | Worst group (≤3%) | Variance-ratio dev. (≤0.10) | Inner-product (≤1%) | Cosine change (≤0.02) | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| global, in-sample | 30.1% fail | 17.8 fail | 242 fail | 695% fail | 1.05 fail | FAIL |
| global, held-out | 33.5% fail | 19.3 fail | 110 fail | 944% fail | 1.06 fail | FAIL |
| layer_site, in-sample | 6.6% fail | 0.087 fail | 38.3 fail | 0.44% pass | 0.014 pass | FAIL — **also invalid: in-sample fit is not a held-out result** |
| layer_site, held-out | 6.8% fail | 0.091 fail | 13.3 fail | 0.47% pass | 0.020 pass (at bound) | FAIL |
| layer_site_dim, in-sample | 0.38% pass | 0.005 pass | 0.01 pass | 0.18% pass | 0.0034 pass | passes numerically, but **invalid: in-sample fit is not a held-out result** |
| layer_site_dim, held-out | 1.22% pass | 0.016 pass | 0.60 fail | 1.33% fail | 0.094 fail | FAIL (variance, inner-product, cosine) |

**Reading this.** Only BF16 mantissa truncation at 4 and 5 bits currently passes every §3 metric on held-out data. Every quantization/codec method that beats it on raw size (SZ3, FP8, zfp, Quant8 at any granularity) fails — held-out Quant8 `layer_site_dim` specifically fails on variance-ratio deviation, inner-product error, and cosine change together, not on Frobenius error, consistent with §3's note that these metrics were chosen to be a real bar rather than one everything clears. The report's own finding 3 ties this failure to clipped outlier values, and proposes storing them as exact BF16 exceptions (~+0.7% size) as an untried fix — see §6a for the result of implementing it.

## 6a. Clipped-value exception fix: implementation and result (2026-10-29)

Implemented in `python/quantized_archive.py`: `compress_archive_quant8(..., store_clipped_exceptions=True)` stores every clipped (row, dimension)'s exact original BF16 bits in a per-chunk sparse table (`exceptions.npz`: row index, dimension index, bits), alongside the usual uint8 codes. The decoder (`QuantizedActivationArchive.iter_chunks`) dequantizes as before, then overwrites exactly those positions with the exact value. Verified bit-exact in `tests/test_activation_archive.py::test_quant8_clipped_exceptions_recover_exact_outlier_values` (held-out clipped positions match the original exactly; without the flag they do not). CLI: `scripts/quantize_activation_archive.py --store-clipped-exceptions`.

Rebuilt held-out Quant8 `layer_site_dim` archives for `high_0056` rg 0/106/212 (calibration: `low_0056` all 16 rg + `high_0066` rg 0/33/65, same split as §4) and re-ran `benchmarks/evaluate_lossy_archive.py` against each reference. Results (worst across the three row groups):

| Metric | Value | §3 tolerance | Result |
| --- | --- | --- | --- |
| Overall relative Frobenius error | 0.53% | ≤2% | pass |
| Worst group relative error | 0.68% | ≤3% | pass |
| Worst variance-ratio deviation | 0.024 | ≤0.10 | pass |
| Worst inner-product relative error | 0.19% | ≤1% | pass |
| Worst cosine change | 0.0033 | ≤0.02 | pass |
| Size vs. raw reference | 0.52× (0.520–0.522 across the three row groups) | — | smaller than BF16 4-bit (0.617×), the previous §6 winner |

**This is the first configuration found that passes every §3 metric on held-out data while beating BF16-4-bit on size.** The exception table costs 0.56–0.67 MB per row group (~0.46% of raw size, close to the report's original "+0.7%" estimate), for 67,711–81,095 clipped values (0.10–0.11% of all values) recovered exactly instead of clipped. Encode time was 0.77–0.93 s per row group (not separately measured before; peak RSS not measured in this run).

This changes the current answer to the project question: on this evaluation set, **held-out Quant8 `layer_site_dim` with clipped-value exceptions, at ~0.52× raw size, is the smallest configuration currently meeting the fidelity constraints in §3** — ahead of BF16-4-bit truncation (0.617×), which was the only passing configuration before this fix.

**Read performance and build resources** (`high_0056` rg 0, same session; `compare_archive_reads.py` with a same-session raw-reference control to correct for the machine-speed drift noted in §5.1):

| Query | This config (median) | Raw reference (same-session control) | Parquet (median) | Speedup vs. Parquet |
| --- | --- | --- | --- | --- |
| Full row group | 0.124 s | 0.049 s | 0.886 s | 7.1× |
| Token range (384 rows) | 0.0089 s | 0.0075 s | 0.746 s | 83.6× |
| Single layer (1,024 rows) | 0.0236 s | 0.0128 s | 0.733 s | 31.0× |

This matches the pre-exception `layer_site_dim` held-out archive's read timings almost exactly (same report, 0.126/0.0089/0.019 s) — the exception lookup adds negligible read overhead, and the archive stays 7–84× faster than Parquet despite being the slowest of the lossy archives to read so far other than SZ3/zfp.

**Build resources** (`/usr/bin/time -v`, full held-out invocation including fitting scales on all 19 calibration row groups, not just the 1 evaluation row group encoded): 196.1 MB peak RSS, 1.92 s wall clock (1.54 s user + 0.22 s system). This is not directly comparable to the original in-sample Quant8 build's 240.5 MB/1.45 s (§ progress log, 2026-10-29 Quant8 entry), which measured only a single-archive in-sample fit with no calibration pass — a like-for-like comparison would need to time the calibration fit and the per-row-group encode separately.

## 6b. Combined layer+token-range query, and cold-cache numbers (2026-10-30)

Added the `layer_token_range` query and cold-cache measurement described in
§2 and §5.3. Ran both on `high_0056` rg 0 for the raw reference archive and
the current winning config (held-out Quant8 `layer_site_dim` + exceptions):

| Query | Config | Warm median | Cold median | Parquet warm median | Parquet cold median | Speedup (warm) |
| --- | --- | --- | --- | --- | --- | --- |
| Full row group | raw reference | 0.049 s | 0.167 s | 0.808 s | 0.932 s | 16.5× |
| Full row group | Quant8 + exceptions | 0.0078 s* | 0.0145 s* | — | — | — |
| Token range | raw reference | 0.0077 s | 0.0162 s | 0.767 s | 0.775 s | 100.0× |
| Single layer | raw reference | 0.0136 s | 0.0964 s | 0.718 s | 0.760 s | 52.9× |
| Layer + token range | raw reference | 0.0072 s | 0.0141 s | 0.898 s | 0.922 s | 125.2× |
| Layer + token range | Quant8 + exceptions | 0.0077 s | 0.0145 s | 0.930 s* | 1.163 s* | 119.7× |

\* These two rows ran in a separate invocation a few minutes earlier than the
raw-reference row in this table; Parquet medians in that run were slower
(0.93–1.56 s) than the same-session raw-reference control, consistent with
the already-documented run-to-run drift on this machine (§5.1) — compare the
Quant8 archive figures with each other, not across the two runs' Parquet
columns. A fully same-session combined run is the next thing to do if this
table needs to support a precise cross-config comparison rather than just
confirming the query and cold-cache mechanics work.

**Reading this.** The combined layer+token-range query behaves like the
other three: no new correctness issues, and speedups in the same 50–125×
range. Cold-cache reads are consistently slower than warm for both Parquet
and the archive (2–7× for the archive, roughly in line with its on-disk
chunk count; smaller for Parquet, whose row-group read was already mostly
sequential I/O), which is the expected direction and the first evidence the
`posix_fadvise` eviction is actually taking effect on this filesystem rather
than silently no-op'ing.

## 7. Exit check

This document satisfies the Stage 0 deliverable (scope, split, queries,
fidelity limits, hardware/methodology). Status as of 2026-10-29:

1. ~~Confirm or revise the fidelity tolerances in §3~~ — done; confirmed by the user 2026-10-29.
2. ~~Re-check the already-run 2026-10-29 lossy results against §3~~ — done; see §6. Current result: only BF16 4-bit/5-bit truncation pass fully; SZ3/FP8/zfp/Quant8 fail (held-out Quant8 `layer_site_dim` on variance-ratio, inner-product, and cosine together, driven by clipped outliers).
3. ~~Implement the clipped-value exception fix for held-out Quant8~~ — done; see §6a.
4. ~~Measure read performance and build resources for the new winning config~~ — done; see §6a. Query-shape coverage turned out to already include single-layer reads for every archive type (§2 corrected); the one real remaining gap is the layer+token-range combined query, not yet implemented anywhere.
5. ~~Implement the layer+token-range combined query~~ — done 2026-10-30; see §6b.
6. ~~Add cold-cache measurement and per-result environment metadata~~ — done 2026-10-30; see §5.3 and §6b. Not yet backfilled into the unused `benchmarks/methods/*.py` placeholders.

All six Stage 0 exit-check items are now done. Remaining work is no longer about this protocol's own completeness — it's applying the protocol (fidelity table, split, query shapes, cold/warm + environment metadata) to new methods and to the Stage 8 capture expansion, per [PROJECT_PIPELINE.md](PROJECT_PIPELINE.md).
