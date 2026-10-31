# Gibbon activation pilot

Starter project for studying complete activation traces from the probabilistic
reasoning task. The code is small; the dataset remains on `ramiel` and is not
stored in GitHub.

## What is in the pilot?

The dataset contains three complete games from
`gpt_oss_full_200_seed0_20260602`:

| Game | Reasoning effort | Tokens | Activation rows | Parquet size |
| --- | --- | ---: | ---: | ---: |
| `high_0056` | high | 104,709 | 5,026,032 | 23.54 GiB |
| `low_0056` | low | 1,008 | 48,384 | 318.08 MiB |
| `high_0066` | high | 29,619 | 1,421,712 | 6.72 GiB |

For every stored token, each file has all 24 layers, both activation sites
(`post_attention` and `post_mlp`), and all 2,880 BF16 dimensions. No tokens,
layers, sites, or dimensions were sampled. The complete pilot is 30.57 GiB.

See [DATASET.md](DATASET.md) for the schema and completeness guarantees.
See [PROJECT_PIPELINE.md](PROJECT_PIPELINE.md) for the end-to-end project plan,
staged deliverables, validation gates, and immediate next steps.

## 1. Connect to Ramiel

Tailscale must be running and your account must be allowed to use Tailscale
SSH as the Linux user `chris`:

```bash
ssh chris@ramiel
```

If your local SSH configuration does not resolve `ramiel`, use its Tailscale
name as shown in your network's Tailscale Machines page. Do not copy private
keys or passwords into this repository.

Confirm access:

```bash
whoami
ls /srv/gibbon-activations/gpt_oss_pilot_3games
test -r /srv/gibbon-activations/gpt_oss_pilot_3games/manifest.json && echo "dataset readable"
test ! -w /srv/gibbon-activations/gpt_oss_pilot_3games && echo "canonical dataset is read-only"
```

The canonical dataset is intentionally read-only. Put generated outputs in
your home directory, for example `~/gibbon-work/`.

## 2. Clone and create the environment

Clone this private repository after accepting the GitHub invitation:

```bash
git clone git@github.com:gguiomar/interp-struct.git
cd interp-struct
```

The recommended environment is defined in `environment.yml`. Install a
user-local Miniforge distribution if `conda` is not already available, then:

```bash
conda env create -f environment.yml
conda activate gibbon-activation-pilot
```

This installs R, R Markdown, Arrow, Python/PyArrow, and PyTorch with CUDA
support without requiring sudo.

Set the dataset path (the examples use this automatically as their default):

```bash
export GIBBON_DATA_ROOT=/srv/gibbon-activations/gpt_oss_pilot_3games
python scripts/check_dataset.py
```

## 3. First analysis

Render the R Markdown walkthrough:

```bash
Rscript -e "rmarkdown::render('analysis/01_game_overview.Rmd', params=list(game_id='low_0056'))"
```

It produces a 15-round panel showing all four buttons in every round,
including occluded buttons, the selected button, and red/green evidence. It
also reports basic game and activation statistics. Start with `low_0056`; it
is small enough for rapid iteration.

Inspect a Parquet row group and optionally move one block to the GPU:

```bash
python python/load_activations.py low_0056 --row-group 0
python python/load_activations.py low_0056 --row-group 0 --gpu
```

The loader reads one Parquet row group at a time and reconstructs the exact
BF16 bit patterns. It does not materialize a whole large game in RAM or VRAM.

Measure one row group's loading throughput and host memory use with:

```bash
python benchmarks/measure_row_group.py --game-id low_0056 --row-group 0
```

Add `--gpu` to measure the transfer and peak CUDA memory when a GPU is available.
The result JSON is written under `benchmarks/results/<game_id>/row_groups/`.

Build an uncompressed, chunked reference archive from a Parquet file with:

```bash
python scripts/build_activation_archive.py \
  /path/to/activations.parquet \
  ~/gibbon-work/archives/low_0056 \
  --game-id low_0056 \
  --row-group 0 \
  --batch-rows 1024
```

The writer processes bounded record batches and writes a versioned index with
per-chunk integrity hashes. Use `python/activation_archive.py`'s
`ActivationArchive.iter_chunks()` for bounded-memory reads, or `read()` for a
convenient in-memory result when the selected data fits RAM. This version is a
reference archive for the pilot's fixed-width activation rows; it does not yet
represent separate head/tensor axes or heterogeneous variable-shaped tensors.
The archive can be integrity-checked with:

```bash
python -c "from pathlib import Path; from python.activation_archive import ActivationArchive; ActivationArchive(Path('~/gibbon-work/archives/low_0056').expanduser()).verify()"
```

Create a separate lossless Zstandard-compressed copy of an existing raw archive:

```bash
python scripts/compress_activation_archive.py \
  ~/gibbon-work/archives/high_0056-rg0-reference-v1 \
  ~/gibbon-work/archives/high_0056-rg0-zstd-v1 \
  --compression-level 3
```

This compresses each activation chunk independently and keeps its metadata and
index. The reader decompresses only chunks used by a query. Verify either raw
or compressed archives with the same `ActivationArchive(...).verify()` command.

Compare full-row-group and token-range reads against Parquet, using an existing
archive:

```bash
python benchmarks/compare_archive_reads.py \
  --game-id low_0056 \
  --row-group 0 \
  --archive ~/gibbon-work/archives/low_0056-rg0-reference-v1-validation
```

This writes one JSON result per query under
`benchmarks/results/<game_id>/row_groups/<row_group>/`. It verifies exact values
before timing and labels its repeated warm-cache timing protocol in each result.

## Suggested first tasks

1. Render `low_0056` and verify the round panel against the JSON log.
2. Compare the paired `low_0056` and `high_0056` games.
3. Measure row-group loading throughput and peak RAM/VRAM ([initial results](analysis/03_loading_measurements.md)).
4. Prototype a smaller lossless representation while preserving the exact
   BF16 bits and metadata alignment.
5. Record every transformation and validate it by round-trip checksums.

## Working rules

- Never modify files under `/srv/gibbon-activations`.
- Never commit Parquet data, credentials, generated HTML, or environment files.
- Keep derived artifacts in your home directory or a documented scratch path.
- Prefer row-group streaming; `high_0056` is too large for casual full-table
  loading.
- Open a pull request for code or documentation changes.
