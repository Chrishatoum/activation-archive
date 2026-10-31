# Pilot Tensor Inventory

This document records what the current pilot actually stores, based on checks
of its manifest, game logs, and all three Parquet schemas. It is separate from
the larger tensor coverage proposed for future extraction.

## Captured Data

| Item | Present in this pilot |
| --- | --- |
| Model | `openai/gpt-oss-20b` |
| Transformer layers | 24, numbered `0` through `23` |
| Capture locations | `post_attention` and `post_mlp` |
| Values per captured row | 2,880 BF16 values, named `activation_dim0000` through `activation_dim2879` |
| Stored value representation | Each value is two bytes (`fixed_size_binary[2]`), preserving its BF16 bits |
| Row identifiers | `game_id`, `round`, `layer`, `activation_site`, and `token_idx` |
| Separate attention-head axis | Not present |

Each Parquet row contains one token's values at one layer and one capture
location. There are five identifier columns and 2,880 activation-value
columns, for 2,885 columns total. One token therefore produces
24 layers × 2 locations = 48 rows. The rows are stored flat; there is no
explicit head dimension.

| Game | Reasoning effort | Generated tokens | Activation rows | Parquet row groups |
| --- | --- | ---: | ---: | ---: |
| `high_0056` | high | 104,709 | 5,026,032 | 213 |
| `low_0056` | low | 1,008 | 48,384 | 16 |
| `high_0066` | high | 29,619 | 1,421,712 | 66 |
| **Total** | | **135,336** | **6,496,128** | |

The game logs and prompts are stored separately from the activation Parquet
files. The logs contain task settings, round choices, observations, generated
text/token information, and final decisions. A future archive must either
include this context or preserve reliable references to it.

## Not Captured

These tensor families are part of the broader project proposal but are not
separate fields in the current pilot Parquet files:

- Query, key, and value tensors.
- Attention-weight tensors.
- Per-head outputs or a head identifier.
- Intermediate MLP activations beyond the `post_mlp` capture point.
- Expert/router assignments, routing scores, or other routing data.
- Architecture-specific recurrent states.

The game logs describe cue availability and round/prompt context, but those
are not activation-value columns in Parquet. This pilot is therefore complete
for its three selected games and two recorded activation locations, not a
complete record of every internal transformer tensor.