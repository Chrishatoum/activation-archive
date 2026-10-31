# Project Pipeline: Full Transformer Activation Storage

This is the working sequence for completing the semester project. It separates the work into stages with concrete outputs and exit checks, so later compression results are built on a correct, measurable archive.

## Project question

Find the smallest practical archive for complete transformer activation records that meets predeclared numerical-fidelity and access-performance requirements. Measure storage, reconstruction error, encoding/decoding throughput, memory use, and read latency on the same data and hardware.

## Current position

The repository has a three-game GPT-OSS 20B pilot containing BF16 `post_attention` and `post_mlp` vectors. It now has a bounded-batch Parquet-to-archive converter, an uncompressed version-1 reference archive, a lossless Zstandard-compressed version-2 archive, archive readers and integrity checks, and synthetic round-trip tests. The current archive layout stores activation chunks as `.npy`/Zstandard-compressed uint16 payloads and row metadata as `.npz` files.

Read comparisons have been run on row group 0 for `low_0056` and `high_0056`, for both full-row-group and token-range queries. A compressed archive has also been built and compared for `high_0056`, row group 0. Raw, Zstandard, and Quant8 codec baselines were separately run on Parquet-loaded data for that same high-effort row group; only the Zstandard archive experiment creates and measures a compressed archive.

These results are preliminary: they cover two row groups, use repeated warm-cache-oriented timings with uncontrolled cache state, and the Parquet comparison reads the full row group before filtering. Archive write time and peak memory remain unmeasured. The current pilot format covers only its flat fixed-width activation rows; it does not yet represent separate heads, Q/K/V, attention weights, routing data, recurrent states, masks, or heterogeneous variable-shaped tensors. See [TENSOR_INVENTORY.md](TENSOR_INVENTORY.md) and the running [archive benchmark report](benchmarks/results/archive_reference_benchmark_report.md).

## Stage 0: Freeze scope and evaluation rules

Before tuning formats, write down the data and success criteria.

- Define which model checkpoints, model revisions, input corpus version, prompts, tokenizer versions, and generation settings are in scope.
- Define the first archive scope as the existing pilot, then define the expanded capture scope separately for GPT-OSS 20B and Qwen3.8-27B.
- Define the supported query shapes: complete sample, token range, layer, head, and tensor type. Specify combinations that must work.
- Define fidelity targets before inspecting held-out results. Exact lossless recovery is the requirement for lossless codecs; set numerical error tolerances for lossy codecs before final evaluation.
- Split data by sample/input into calibration and evaluation sets whenever an encoder learns scales, codebooks, or prediction parameters. Never fit on evaluation samples.
- Record hardware, OS, library versions, and benchmark protocol. Separate cold-cache and warm-cache measurements where practical.

**Deliverable:** a short evaluation protocol containing scope, held-out split, supported queries, fidelity limits, and benchmark hardware/methodology.

**Exit check:** the same tensors, splits, queries, and metrics can be used for every storage method.

## Stage 1: Validate the existing source dataset

Use the pilot to develop the archive before requesting or extracting a larger dataset.

```bash
export GIBBON_DATA_ROOT=/srv/gibbon-activations/gpt_oss_pilot_3games
python scripts/check_dataset.py
```

- Verify manifest entries, file readability, row counts, row-group counts, metadata ranges, and tensor dimensions for every game.
- Verify all expected tokens, layers, and capture locations are present according to the pilot's actual contract.
- Keep the canonical dataset read-only. Put derived files under `~/gibbon-work/`.
- Save a machine-readable source inventory and checksums so every benchmark input can be traced back to the same source.

**Deliverable:** validated source manifest and an inventory of exactly what the pilot captures and omits.

**Exit check:** validation is repeatable and fails clearly if files or expected records are missing.

## Stage 2: Build the uncompressed reference archive

Finish this before judging compression. The reference archive establishes the layout, index, read API, and exact baseline against which every encoding is checked.

- Turn the current prototype into a reusable writer and reader API instead of keeping archive logic only inside `compare_parquet_archive.py`.
- Make writing bounded-memory: stream source row groups or smaller blocks into archive chunks instead of retaining a whole game or row group in memory.
- Use versioned metadata/index files. Store model and input identifiers, sample/game, token boundaries, layer, head when present, tensor type, shape, dtype, byte order, chunk coordinates, source provenance, and checksums.
- Keep masks, ragged sequence lengths, and tensor-specific shapes explicit. The archive must not assume every tensor is a flat fixed-width vector.
- Adopt a chunked, indexed storage backend suitable for the project. Evaluate Zarr v3 indexed sharding, as proposed, against the current raw NumPy chunk prototype. Choose and document one reference layout based on query and memory requirements; do not build two permanent APIs without a measured reason.

**Status (2026-10-31):** decided and migrated — see [ZARR_EVALUATION.md](ZARR_EVALUATION.md). `python/activation_archive.py` (the reference/lossless tier) now writes and reads a Zarr v3 array instead of the hand-rolled per-chunk .npy/.npz layout; `python/quantized_archive.py` and `python/codec_archive.py` (the lossy tier) keep their own format, decoupled from the reference archive's internals via the public `iter_chunks()` API rather than reaching into its physical chunk files. All 69 archives under `/home/chris/gibbon-work/archives/` were rebuilt with the new code and verified — the 8 reference archives bit-exact against the original Parquet source, the rest fidelity-checked against the report's recorded numbers (exact match on the winning Quant8 config's worst-case metrics across the three evaluation row groups). Old copies kept alongside as `*.bak` pending cleanup.
- Make writes safe against interruption: write temporary chunk/index files, verify them, then publish a completed index. Detect incomplete or incompatible archives on read.
- Implement selectors for sample, token range, layer, head, and tensor type, and allow combinations. Return data and identifiers in a documented order.
- Include indices, labels, scales if any, padding, and per-chunk overhead in total archive size.

**Pilot deliverable status:** a documented version-1 uncompressed format, Python writer/reader API, batch-based conversion command, and scratch archives exist for pilot row groups. This does not complete the final format design for the broader project tensor scope.

**Exit checks:**

- Exact BF16 bitwise equality and exact metadata equality after write/read.
- Every supported selector returns the same values and identifiers as the source.
- Empty ranges, boundary tokens, variable lengths, and missing/invalid selector values behave predictably.
- Peak working memory is bounded by the configured chunk size, rather than total game size.
- Interrupted or corrupted archives are detected rather than silently accepted.

The current tests cover synthetic exact round trips, selectors, chunk iteration, Parquet conversion, Zstandard archive round trips, and modified-data checksum detection. Real-data exact round trips have been checked on `low_0056` and `high_0056` row group 0. Peak memory has not yet been measured, and tests for heterogeneous shapes, masks, truncated files, invalid indexes, and the broader query schema remain.

## Stage 3: Test archive correctness and streaming behavior

Create focused automated tests before broad performance runs.

- Test writer/reader round trips on synthetic tensors with known shapes and dtypes, including BF16 bit patterns, non-finite values, variable-length sequences, masks, and metadata.
- Test a real small pilot row group and a larger `high_0056` row group.
- Test every selector alone and in combinations; compare output order and content to the canonical loader.
- Test checksums, version handling, truncated chunks, and invalid indices.
- Exercise streams across chunk boundaries, including a query that starts in one chunk and ends in another.

**Deliverable:** automated test suite and a reproducible command for running it.

**Exit check:** tests pass without access to the full dataset where possible; real-data integration tests pass when the dataset is mounted.

## Stage 4: Establish a fair Parquet-versus-reference benchmark

The current comparison records full-row-group and token-range reads on two pilot row groups, plus raw-versus-Zstandard archive reads on the larger row group. It remains preliminary rather than comprehensive: Parquet is read as a full row group then filtered, cache state is uncontrolled, and archive creation time and peak memory are not included. Expand the benchmark to compare equivalent operations and include archive creation.

Run on identical source records and hardware, at minimum:

- Complete row-group or sample read.
- Single-layer read.
- Token-range read.
- Layer plus token-range read.
- Head and tensor-type reads once those axes exist in the expanded dataset.
- Small and large samples, including `low_0056` and `high_0056`.

For each method and query, record:

- Total bytes, including data, indices, metadata, padding, and any sidecars.
- Write/encode and read/decode latency and throughput.
- Peak host RAM and, when relevant, peak GPU memory.
- Cold-cache and warm-cache latency where cache state can be controlled; otherwise record the limitation.
- Rows/tensors/bytes returned and exact source/query identity.
- Integrity and fidelity checksums.

Compare equivalent output work. If Parquet must read and filter a whole row group, report that as the current Parquet behavior; do not present it as an inherent limit of Parquet if predicate pushdown, column projection, or a different row-group layout could improve that query. Likewise, include archive index/metadata loading and selection costs.

**Deliverable:** benchmark runner, machine-readable results, and a short report explaining the workload and caveats.

**Exit check:** the benchmark is repeatable and does not compare different records or omit one format's overhead.

## Stage 5: Measure the uncompressed storage/access baseline

Run the validated benchmark over representative pilot data. This stage determines whether the reference layout meets access and memory requirements before compression is added.

- Convert at least one small and one large pilot sample.
- Verify archive total size and checksums.
- Measure sequential full reads and the selective queries defined in Stage 0.
- Measure bounded-memory writing and reading, including crossing chunk boundaries.
- Compare results with source Parquet and explain differences in row layout, compression, caching, and filtering.

**Deliverable:** baseline tables and plots for storage, throughput, peak memory, and query latency.

**Exit check:** reference archive meets correctness and streaming requirements. If not, revise chunking/index layout before adding codecs.

**Current status:** initial read comparisons exist for full-row-group and token-range queries on `low_0056` and `high_0056`, row group 0. Additional row groups, layer-based queries, controlled cache conditions, write throughput, and peak-memory measurements are still needed.

## Stage 6: Add compression and reduced-precision methods

Implement methods as interchangeable encoders/decoders over the same reference chunks and common result schema. Start with baselines, then add methods in increasing complexity.

1. Raw/uncompressed reference.
2. Lossless general-purpose compression, such as Zstandard, applied to the stored tensor chunks.
3. Reduced-precision storage where supported by the data and intended fidelity contract.
4. Global and blockwise quantization, with block axes and scale granularity treated as experimental variables.
5. Error-bounded floating-point codecs such as ZFP and SZ3.
6. Optional low-rank or predictive methods only after the baselines and access requirements are stable.

**Current status:** raw, Zstandard level 3, and Quant8 codec baselines have been measured on Parquet-loaded `high_0056` row group 0. Separately, a real version-2 archive was created by compressing the version-1 archive's activation chunks with Zstandard level 3; exact recovery and reads were verified. Quant8 remains a codec result, not a compressed archive implementation. Quant4, blockwise scaling, ZFP, SZ3, and reduced-precision archive variants remain to be implemented or measured.

For every method:

- Preserve complete tensor coverage; compression must not drop tokens, heads, layers, tensor types, or masked regions.
- Include scales, codebooks, masks, indices, padding, and codec metadata in encoded size.
- Record codec version and all parameters in the archive/index and benchmark result.
- Decode and validate before accepting a result. Lossless methods must reproduce exact original values; lossy methods must report the predeclared metrics.
- Measure encode/decode time, peak memory, compression ratio, and query behavior, not just compression size.
- Fit calibration-dependent parameters on calibration samples only and apply them unchanged to held-out evaluation samples.

**Deliverable:** tested method implementations with one shared result format and results on calibration and held-out data.

**Exit check:** each method is reproducible, complete, and measured with all metadata overhead and required fidelity statistics.

## Stage 7: Evaluate numerical distortion and downstream structure

Relative Frobenius error and maximum absolute error are necessary but not sufficient for tensor families with different scales or outliers.

For each evaluated tensor, stratify results by model, sample, layer, head, tensor type, and sequence length as applicable. Report:

- Relative Frobenius error: `||A - A_hat||_F / (||A||_F + epsilon)`.
- Maximum absolute error and error quantiles/distributions.
- Mean and variance changes.
- Pairwise inner-product changes for representative vectors.
- Exact bitwise match rate for lossless/reference paths.
- Behavior on masked or redundant entries, including whether reconstruction from schema metadata is exact and how the metadata cost is counted.

Fix tolerances before evaluating held-out data. Show the storage/performance frontier and identify which methods satisfy all constraints, rather than selecting a winner from compression ratio alone.

**Deliverable:** held-out fidelity report and storage-distortion-throughput frontier plots.

**Exit check:** the selected configurations meet the predeclared error limits across relevant tensor types and do not hide bad layers/heads in averages.

## Stage 8: Expand capture to the full project scope

Do this after the reference format, tests, and pilot benchmarks are stable. Agree the capture points and feasibility with the model/runtime owners before extraction.

**Status (2026-10-30):** a draft planning document exists — [STAGE8_CAPTURE_PROPOSAL.md](STAGE8_CAPTURE_PROPOSAL.md). It covers architecture research for both models (confidence-labeled, not yet confirmed against either model's real config), a storage/compute feasibility estimate showing why "capture everything" doesn't fit (Q/K/V alone is ~21× the pilot's current per-token footprint; full attention-weight matrices are flagged as not viable at all), and six blocking open questions for the model/runtime owners. No capture code has been written yet — this stage has not started in the sense of extraction, only in the sense of planning what to ask for.

- Freeze GPT-OSS 20B and Qwen3.8-27B checkpoints and record immutable identifiers.
- Freeze and version the input corpus, prompts, tokenization, generation settings, and software/runtime versions.
- Define a tensor inventory for each architecture: residual streams, Q/K/V, attention weights, per-head outputs, intermediate feed-forward activations, recurrent states where applicable, and routing assignments/scores where applicable.
- For every capture point, record tensor type, shape, dtype, axes, masks, token boundaries, and architecture-specific interpretation.
- Validate that all expected samples and capture points are present and that sequence-length-dependent and quadratic attention tensors are represented correctly.
- Write directly through the streaming capture path where possible; avoid creating an uncontrolled full-size intermediate copy.
- Keep the new extraction immutable and versioned; store provenance and checksums in the manifest.

**Deliverable:** new complete dataset, versioned tensor inventory, capture tools, and completeness report for both model families.

**Exit check:** independent checks establish that all required tensor families and identifiers were captured, with no silent truncation or missing masks.

## Stage 9: Run the final reproducible benchmark

Run the frozen benchmark matrix over the pilot and expanded evaluation datasets using the same hardware where possible. Record deviations when hardware or runtime differs.

- Evaluate complete samples and representative selective reads.
- Include cold/warm access, sequential throughput, write/read throughput, and peak memory.
- Evaluate compression methods on held-out inputs and report calibration/evaluation separation.
- Preserve raw result JSON, environment/version data, commands, and plotting scripts.
- Rerun important measurements enough times to report variation, not just a single best timing.

**Deliverable:** reproducible benchmark package and final results tables/plots.

**Exit check:** another person can reproduce the result from documented commands and manifests, without access to undocumented local state.

## Stage 10: Package the final project

- Document the archive format and compatibility/version policy.
- Document the Python API and provide examples for writing, querying, validating, and reconstructing data.
- Document the capture pipeline and how architecture-specific tensor layouts are described.
- Publish tests, benchmark commands, result schemas, and environment lock/version information.
- Write the final report: methods, dataset scope, limitations, storage-distortion-throughput frontier, and which tensor types compress well or poorly.
- Keep data, credentials, and generated large archives out of Git; publish code, metadata schemas, and appropriately sized reproducibility artifacts.

**Deliverable:** archive specification, Python API, capture/reconstruction tools, tests, benchmark suite, result artifacts, and final report.

## What to do next

The archive and first compression prototype now exist. The next work is to define success criteria and close measurement gaps before scaling data collection:

1. ~~Write the Stage 0 evaluation protocol~~ — done as a draft: see [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) (2026-10-29). It covers scope, supported queries, a proposed fidelity tolerance table, the calibration/evaluation split, and hardware/software environment, but the tolerances still need explicit confirmation, and several existing held-out results (Quant8, codec matrix) predate it and must be checked against it retroactively.
2. Measure archive creation time and peak RAM for the existing low/high row-group examples; record the commands and environment versions.
3. Improve benchmark fairness and coverage: add layer-only and layer-plus-token-range queries, test additional row groups, compare reasonable Parquet projections/layouts, and run controlled cold/warm reads where practical.
4. Compare raw and Zstandard archive sizes and reads on more chunks. Measure compression and decompression time and peak memory as part of the archive workflow, not only codec-only payload benchmarks.
5. Implement and evaluate Quant4, blockwise scaling, reduced precision, ZFP, and SZ3 as archive encoders/decoders. Account for indexes and all codec metadata; report error distributions and per-layer/tensor results.
6. Evaluate the current NumPy chunk layout against Zarr v3 indexed sharding, then document the chosen format and query/index contract.
7. Expand capture only after the pilot design is stable: freeze GPT-OSS 20B and Qwen3.8-27B checkpoints and inputs, define complete tensor inventories, and implement streaming capture/reconstruction for all required tensor types, masks, heads, and variable shapes.
8. Run held-out evaluation across representative models and sequence sizes, produce storage-distortion-throughput frontiers, and identify the smallest configuration meeting the fixed fidelity/access constraints.
9. Package the format specification, Python API, capture/reconstruction tools, tests, benchmark commands/results, environment versions, and final report so another researcher can reproduce the findings. **Status (2026-10-31):** format specification written — see [FORMAT_SPEC.md](FORMAT_SPEC.md) (covers all three archive kinds' on-disk layout, index.json schema, and reproduction commands). Still missing: a final written-up report tying the whole pilot together for someone who hasn't read every doc in this repo.

The running pilot measurements and detailed result links are in [benchmarks/results/archive_reference_benchmark_report.md](benchmarks/results/archive_reference_benchmark_report.md). They establish useful hypotheses, but do not yet establish a final winner: only two row groups were used for read tests, cache state was uncontrolled, the Parquet path reads a complete row group before filtering, and the larger proposed tensor families have not yet been captured.
