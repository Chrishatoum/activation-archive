# Dataset reference

## Location and layout

The canonical pilot is mounted on Ramiel at:

```text
/srv/gibbon-activations/gpt_oss_pilot_3games/
├── README.md
├── manifest.json
├── source_manifest.json
├── activations/
│   ├── high_0056_attempt_000.parquet
│   ├── low_0056_attempt_000.parquet
│   └── high_0066_attempt_000.parquet
├── logs/
│   ├── high_0056.json
│   ├── low_0056.json
│   └── high_0066.json
└── prompts/
    └── prompt_<game_id>_<round>.txt
```

There are 16 prompts per game: rounds `0` through `14` plus the final decision
prompt at index `15`. Each game log contains 15 evidence rounds and one
`final_decision` object.

## Activation Parquet schema

Each row is one token at one layer and one activation site. The five metadata
columns are:

| Column | Type | Meaning |
| --- | --- | --- |
| `game_id` | string | Game identifier |
| `round` | int16 | Prompt/decision round index |
| `layer` | int16 | Transformer layer, `0`–`23` |
| `activation_site` | string | `post_attention` or `post_mlp` |
| `token_idx` | int32 | Token position within the recorded output |

The remaining 2,880 columns contain one BF16 value each, encoded as its exact
two-byte bit pattern (`fixed_size_binary[2]`). A token therefore contributes 48 rows:
24 layers × 2 sites.

Do not cast through float32 and back when testing losslessness. Compare the
underlying `uint16` bit patterns.

For a plain-language comparison of the paired low- and high-effort runs, see
[analysis/02_paired_game_comparison.md](analysis/02_paired_game_comparison.md).

## Game log schema

Important top-level fields include `game_id`, `reasoning_effort`, task
configuration, cue mapping, biased/correct cue, success, model provenance,
activation metadata, `rounds`, and `final_decision`.

Every item in `rounds` records:

- zero-based `round`;
- `available_cues` (the availability mask);
- per-cue `round_data` with red/green evidence;
- model `choice` and `observed_color`;
- reasoning trace and generated token information;
- prompt path and timing.

The R Markdown example explicitly reconstructs the unavailable/occluded cue
cells instead of silently dropping them.

## Pilot manifest and integrity

`manifest.json` records exact sizes, row counts, token counts, row groups, and
SHA-256 checksums for the selected files. The hashes were checked against NCC
after transfer. `source_manifest.json` is retained unchanged from the full
generation run for provenance.

Totals:

- 3 games
- 135,336 tokens
- 6,496,128 activation rows
- 32,821,752,946 activation-file bytes
- 30.57 GiB including logs, prompts, and manifests

## What this pilot does not contain

It is complete for the three selected games, but it does not contain the other
games from the full 200-game run. This selection is intended for lossless
format design, GPU-loading experiments, and paired-game analysis—not for
estimating population-level statistics over the entire run.
