# Archive Format Specification

**Status:** 2026-10-31, covers the pilot's three archive kinds as implemented. Written so another researcher could implement a compatible reader/writer from this document alone, without reading the source.

## Shared conventions

Every archive is a directory containing an `index.json` plus the data files it describes. All four archive kinds (reference-uncompressed, reference-zstd, Quant8, codec) share one format identifier:

```json
"format": "gibbon-activation-archive"
```

**The `version` integer is the real discriminator, not the format string.** A reader must check both fields together and reject anything it doesn't recognize; it must not infer the archive kind from `version` alone without also checking `format`, and the project convention is that every `version` value across every archive kind in this project is globally unique (never reused for two different kinds), so a reader that only checks `version` still can't be fooled by a different kind accidentally matching.

| Version | Kind | Compression | Reader |
| --- | --- | --- | --- |
| 1 | reference (legacy, retired 2026-10-31) | none | — |
| 2 | reference (legacy, retired 2026-10-31) | zstd | — |
| 3 | Quant8 | one of `quant8_global_affine` / `quant8_layer_site_affine` / `quant8_layer_site_dim_affine` | `python/quantized_archive.py:QuantizedActivationArchive` |
| 4 | codec | one of `bf16_trunc_zstd` / `fp8_e4m3_block_scale` / `zfp_fixed_rate` / `sz3_abs_rms` | `python/codec_archive.py:CodecActivationArchive` |
| 5 | reference | none | `python/activation_archive.py:ActivationArchive` |
| 6 | reference | zstd | `python/activation_archive.py:ActivationArchive` |

Versions 1/2 are retired: no reader in this codebase opens them any more (see the 2026-10-31 Zarr migration below). If you need to read an archive stamped version 1 or 2, you're holding one built by code older than this document — regenerate it from the original Parquet source with the current `build_archive_from_parquet`.

Every archive's rows carry five parallel metadata columns, in this fixed order, defined once in `python/load_activations.py:METADATA_COLUMNS`:

| Column | Dtype | Meaning |
| --- | --- | --- |
| `game_id` | string | Which game/episode this row's activation came from |
| `round` | int16 | Round index within the game |
| `layer` | int16 | Transformer layer index |
| `activation_site` | string | Where in the layer this was captured (e.g. `post_attention`, `post_mlp`) |
| `token_idx` | int32 | Token position within the round |

Row order is whatever order rows were appended in (for the reference archive: Parquet row order within the requested row group(s)); every other archive kind preserves the reference archive's row order exactly, so row `i` means the same (game, round, layer, site, token) across all archive kinds built from the same source.

Every archive's `index.json` also carries `rows`, `dimensions`, `game_id`, `metadata_fields` (must equal `METADATA_COLUMNS` above), `metadata_dtypes` (numpy dtype string per column), and `archive_size_bytes` (total bytes of every file in the archive directory, index.json included, computed by writing the index up to 3 times until the recorded size and the actual directory size agree — see `_index_size`/the writer `finalize()` methods for the exact fixed-point loop).

## 1. Reference archive (versions 5/6)

`python/activation_archive.py`. The exact, bit-preserving tier every other archive kind is built from.

**Files:**
- `index.json`
- `activations.zarr/` — a Zarr v3 array, shape `(rows, dimensions)`, dtype little-endian uint16 (each uint16 is one BF16 value's raw bits, reinterpret as BF16 by left-shifting into the top 16 bits of a float32: `(bits.astype(uint32) << 16).view(float32)`). Chunked and sharded at `chunk_rows_max` rows per chunk/shard, full width. Version 6 adds a Zstandard codec (`zarr.codecs.ZstdCodec`) on top; version 5 has no array-level compressor.
- `metadata.npz` — one NumPy archive with five arrays, one per metadata column above, each length `rows`, in row order. String columns (`game_id`, `activation_site`) are stored as fixed-width unicode (`np.savez` on `.astype(str)` arrays).

**index.json fields beyond the shared set:**
```json
{
  "format": "gibbon-activation-archive",
  "version": 5,
  "activation_dtype": "little-endian uint16 containing exact BF16 bits",
  "chunk_rows_max": 1024,
  "compression": "none",
  "activations_sha256": "<sha256 of the concatenated row-major activation bytes, in row order>",
  "metadata_sha256": "<sha256 digest, see below>"
}
```
Version 6 additionally has `"compression": "zstd"` and `"compression_level": <int>`.

**`metadata_sha256` digest definition** (`_metadata_digest`): for each column in `METADATA_COLUMNS` order, hash the column name + a null byte, then: for string columns, each value's UTF-8 length (8 bytes little-endian) followed by its UTF-8 bytes; for numeric columns, the dtype string + a null byte followed by the raw contiguous bytes of the whole column. This exact byte-for-byte procedure must be reproduced identically to verify an archive — it is not simply `hash(bytes(array))`.

**Reading:** `ActivationArchive(path).read(round_index=, token_start=, token_stop=, layer=, activation_site=)` — all filters optional and ANDed together; `token_start`/`token_stop` form a half-open `[start, stop)` range and must be given together. `iter_chunks(**same filters)` yields `(metadata_dict, bits_array)` per `chunk_rows_max`-sized batch, for bounded-memory access. `verify()` recomputes both SHA-256 digests over the whole archive and checks row count.

**Writing:** `ArchiveWriter(path, game_id, chunk_rows=4096)` as a context manager; `.append(metadata_dict, bits_2d_uint16)` any number of times (row counts don't need to align to `chunk_rows`), then `.finalize()`. `build_archive_from_parquet(parquet_path, archive_path, game_id, batch_rows=1024, row_group=None)` is the usual entry point from the original Parquet capture.

`compress_archive(source_path, destination_path, compression_level=3)` produces the version-6 zstd tier from an existing version-5 archive, streaming through `iter_chunks()` so peak memory doesn't scale with archive size.

## 2. Quant8 archive (version 3)

`python/quantized_archive.py`. Lossy: each BF16 value is affinely mapped to a uint8 code. Its own chunked directory format, independent of the reference archive's internal storage — built by reading a reference archive only through its public `iter_chunks()` method, never its files directly, which is what let the reference tier's storage backend change (to Zarr) without touching this format.

**Layout:** one subdirectory per chunk, named `chunk-00000000`, `chunk-00000001`, ... in row order, each containing:
- `codes.npy` — uint8 array, shape `(chunk_rows, dimensions)`
- `metadata.npz` — same five columns as above, for this chunk's rows
- `exceptions.npz` (only if any values clipped and `store_clipped_exceptions=True` was used) — three parallel arrays `rows` (uint32, row index within the whole archive), `dims` (uint16), `bits` (uint16, exact original BF16 bits), sorted by `rows`

Plus, at the archive root: `index.json`, and (only for `layer_site_dim` granularity) `parameters.npz`.

**Optional payload compression (2026-11-01, opt-in, not the default).** If built with `payload_compression="zstd"`, every one of the files above (`codes.npy`, `metadata.npz`, `exceptions.npz`, `parameters.npz`) is written as a raw zstd frame of that file's normal bytes, and gets a `.zst` suffix (`codes.npy.zst`, etc.) — the `index.json` `"*_file"` fields always hold the real on-disk filename, so a reader never has to guess. This is a second, independent, fully lossless layer on top of Quant8's own lossy quantization: it compresses the already-quantized bytes, so it changes size and read CPU cost only, never any fidelity metric. Measured on the three evaluation row groups: ~12–13% smaller, +31% full-archive read time (182.6 ms → 239.8 ms, same machine/session). Default is `"none"` (every existing archive and script call is unaffected); not applied to any of the project's shipped archives, since the read-speed cost is a real tradeoff against this project's "fast reads" result, not a strict win.

**Affine quantization.** A *group* is either "all rows" (`global` granularity), or all rows sharing a `(layer, activation_site)` pair (`layer_site`), or further split per output dimension within that pair (`layer_site_dim`). Each group has a `minimum` and `scale` (float32, rounded before use so encode/decode agree exactly): `code = round((value - minimum) / scale)`, clipped to `[0, 255]`; decode: `value = code * scale + minimum`. Parameters are fit either in-sample (on the archive's own data) or held-out (on separate calibration archives, which must not include the source game — enforced at build time).

**index.json fields beyond the shared set:**
```json
{
  "format": "gibbon-activation-archive",
  "version": 3,
  "source_activation_dtype": "bfloat16",
  "encoded_activation_dtype": "uint8",
  "compression": "quant8_global_affine | quant8_layer_site_affine | quant8_layer_site_dim_affine",
  "scale_granularity": "global | layer_site | layer_site_dim",
  "calibration": {"mode": "in_sample | held_out", "game_ids": [...], "archives": [...]},
  "chunk_rows_max": 1024,
  "relative_frobenius_error": 0.0,
  "max_abs_error": 0.0,
  "clipped_values": 0,
  "clipped_fraction": 0.0,
  "clipped_exceptions_stored": false,
  "exception_bytes_total": 0,
  "payload_compression": "none | zstd",
  "groups": {"<key>": {"minimum": 0.0, "scale": 0.0}},
  "chunks": [{"chunk": "chunk-00000000", "rows": 1024, "activation_file": "codes.npy", "codes_sha256": "...", "metadata_file": "metadata.npz", "metadata_sha256": "...", "stored_bytes": 0, "exceptions_file": "exceptions.npz", "exception_count": 0, "exceptions_sha256": "..."}]
}
```
`"groups"` is present for `global`/`layer_site`; for `layer_site_dim` it's replaced by `"parameters_file": "parameters.npz"` + `"parameters_sha256"`, since per-dimension parameters don't fit compactly inline. Group keys are `"all"` for global, else `"<layer>:<activation_site>"`.

**Reading:** `QuantizedActivationArchive(path)` exposes the same `.read()`/`.iter_chunks()`/`.verify()` interface as the reference reader. Exceptions (if present) overwrite their exact (row, dim) positions with the original BF16 value after dequantization, so a clipped value's contribution to every fidelity metric becomes zero rather than the clipped error.

**Writing:** `compress_archive_quant8(source_path, destination_path, chunk_rows=None, granularity="global", calibration_paths=None, store_clipped_exceptions=False, exception_margin=0.0, payload_compression="none")`. `exception_margin` (2026-11-01, opt-in) only stores an exception for a clip landing more than that many code-units past `[0, 255]`; the default `0.0` stores every clip, matching the original behavior.

## 3. Codec archive (version 4)

`python/codec_archive.py`. Lossy: one of four interchangeable codecs (`bf16_trunc_zstd`, `fp8_e4m3_block_scale`, `zfp_fixed_rate`, `sz3_abs_rms`), picked by name, each independently encoding one block per `(chunk, layer, activation_site)` group.

**Layout:** same per-chunk subdirectory convention as Quant8 (`chunk-00000000`, ...), each containing `payload.bin` (the concatenation of every block's encoded bytes in group order) and `metadata.npz`.

**index.json fields beyond the shared set:**
```json
{
  "format": "gibbon-activation-archive",
  "version": 4,
  "source_activation_dtype": "bfloat16",
  "compression": "bf16_trunc_zstd | fp8_e4m3_block_scale | zfp_fixed_rate | sz3_abs_rms",
  "codec_parameters": {},
  "codec_version": "<library name and version used to encode>",
  "block_layout": "one block per (chunk, layer, activation_site); rows in source order",
  "calibration": {"mode": "none", "note": "..."},
  "chunk_rows_max": 1024,
  "encode_time_s": 0.0,
  "chunks": [{"chunk": "chunk-00000000", "rows": 1024, "activation_file": "payload.bin", "payload_sha256": "...", "metadata_file": "metadata.npz", "metadata_sha256": "...", "blocks": [{"group": "<layer>:<site>", "rows": 0, "offset": 0, "bytes": 0}], "stored_bytes": 0}]
}
```
Each block dict also carries codec-specific fields (e.g. `abs_error_bound` for `sz3_abs_rms`) needed to decode that block; see `CODECS` in `python/codec_archive.py` for the exact per-codec parameter set and valid ranges (e.g. `bf16_trunc_zstd` requires `0 <= mantissa_bits <= 7`).

**Reading/writing:** `CodecActivationArchive(path)`, `compress_archive_codec(source_path, destination_path, codec, **params)` — same `iter_chunks()`-based decoupling from the reference archive's internals as Quant8.

## Reproducing an archive from scratch

```bash
# Reference archive from the original Parquet capture
python scripts/build_activation_archive.py <parquet> <archive-dir> --game-id <id> [--row-group N]

# Zstandard-compressed reference tier
python scripts/compress_activation_archive.py <reference-archive> <zstd-archive-dir>

# Quant8 (see scripts/quantize_activation_archive.py --help for calibration/exception flags)
python scripts/quantize_activation_archive.py <reference-archive> <quant8-archive-dir> --granularity layer_site_dim

# Codec archive (see scripts/codec_activation_archive.py --help)
python scripts/codec_activation_archive.py <reference-archive> <codec-archive-dir> --codec <codec-name> [--mantissa-bits N | --rate N | --tolerance N]
```

Every builder verifies its source archive (calls `.verify()`) before encoding, and every archive this project currently ships was itself verified bit-exact against the original Parquet source after the 2026-10-31 Zarr migration (see the benchmark report's migration entry).
