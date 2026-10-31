# Paired Game Comparison: `low_0056` and `high_0056`

## Why these two runs are paired

Both logs describe the same game setup: GPT-OSS 20B, base seed `0`, game seed `56`, paired-game index `56`, prompt version `0`, and correct cue `A`. The recorded difference of interest is reasoning effort: `low` versus `high`.

This is one pair of runs. It is useful for understanding this pilot's data, but it is not enough to conclude that high reasoning effort generally performs better.

## Summary

| Measure | `low_0056` | `high_0056` |
| --- | ---: | ---: |
| Reasoning effort | low | high |
| Correct cue | A | A |
| Final choice | A | A |
| Successful final answer | yes | yes |
| Rounds | 15 | 15 |
| Valid available-cue choices | 10 / 15 | 15 / 15 |
| Invalid unavailable-cue choices | 5 / 15 | 0 / 15 |
| Generated tokens | 1,008 | 104,709 |
| Activation rows | 48,384 | 5,026,032 |
| Parquet row groups | 16 | 213 |

Each generated token has 48 activation rows (24 layers × 2 capture locations), which explains the activation-row totals. The high-effort run generated about 104 times as many tokens, so its activation file is also much larger.

## Round-by-round behavior

`NONE` means the chosen cue was unavailable, so no color observation was produced.

| Round | Available cues | Low choice / result | High choice / result | Same choice? |
| ---: | --- | --- | --- | --- |
| 1 | A, D | A / RED | A / RED | yes |
| 2 | A, B, D | B / GREEN | B / GREEN | yes |
| 3 | A, B, D | A / RED | D / RED | no |
| 4 | A, C, D | A / RED | C / GREEN | no |
| 5 | C, D | A / NONE | C / RED | no |
| 6 | B, D | A / NONE | D / RED | no |
| 7 | B, C, D | A / NONE | D / GREEN | no |
| 8 | A, B, D | A / RED | A / RED | yes |
| 9 | B, C | A / NONE | B / RED | no |
| 10 | A, B, D | A / GREEN | A / GREEN | yes |
| 11 | A, C, D | A / RED | C / GREEN | no |
| 12 | A | A / RED | A / RED | yes |
| 13 | A, C, D | A / RED | A / RED | yes |
| 14 | A, B | A / RED | A / RED | yes |
| 15 | D | A / NONE | D / RED | no |

The two runs chose the same cue in 7 of 15 rounds. The low-effort run repeatedly chose `A`, including five rounds when `A` was not available. The high-effort run chose an available cue in every round and explored more cues. Both still identified `A` as the biased cue in the final answer.

## What this tells us about the data

- The logs preserve context needed to interpret activations: available cues, choices, observations, and final answers.
- The high- and low-effort runs have very different sequence lengths. Storage and throughput tests should therefore include both small and large runs.
- Matching round numbers does not mean the model processed identical text. After their choices diverge, the generated content and activation tokens can differ too.
- These are behavioral observations from the game logs. This note does **not** compare activation values or prove a general effect of reasoning effort.

For the tensor fields captured in these runs, see [TENSOR_INVENTORY.md](../TENSOR_INVENTORY.md).
