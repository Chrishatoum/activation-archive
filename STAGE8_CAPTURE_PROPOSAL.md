# Stage 8 Capture Proposal: Expanding to Full GPT-OSS 20B and Qwen3.8-27B Tensor Inventories

**Status:** draft, 2026-10-30. This is a planning document, not an implementation.
Per [PROJECT_PIPELINE.md](PROJECT_PIPELINE.md) Stage 8: "Agree the capture points
and feasibility with the model/runtime owners before extraction." That
conversation hasn't happened yet — this document exists to make it concrete.
Nothing here should be treated as locked in until the open questions in §5 are
answered by whoever owns model/runtime access.

## 1. Why this matters now

Everything built so far (Stages 2–7, [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md))
operates on the existing 3-game pilot, which captures exactly two activation
sites per layer (`post_attention`, `post_mlp`) as flat per-token vectors, with
no head axis, no Q/K/V, no attention weights, and no MoE routing data. Stage 8
is the step that captures the rest. It is also, by a wide margin, the biggest
remaining item in the project (see the time estimate given 2026-10-30): it
needs new compute access, a new capture harness, and — per the numbers in §4 —
a deliberate scope decision, because "capture everything" is not storable.

## 2. Architecture research (confidence-labeled; confirm before implementation)

I don't have either model's actual `config.json` on this machine (no cached
weights, no `transformers` install in this environment), so everything below
is from a 2026-10-30 web search, not from inspecting the model directly.
Treat it as a starting point, not ground truth — re-derive from the real
config before writing capture code.

### 2.1 GPT-OSS 20B — moderate-to-high confidence

Sources: NVIDIA Megatron-Bridge docs, Hugging Face `transformers` model docs,
vLLM recipes — established technical documentation, not secondary blogs.

| Property | Value | Cross-check |
| --- | --- | --- |
| Layers | 24 | **Matches our own pilot data exactly** (`TENSOR_INVENTORY.md`: 24 layers) |
| Hidden size | 2,880 | **Matches our own pilot data exactly** (2,880 activation dims) |
| Attention | Grouped-query: 64 query heads, 8 KV heads/groups, head_dim 64 | Consistent across sources |
| MoE | 32 experts per layer, top-4 routing | Consistent across most sources (one outlier source says 64/2) |
| Total / active params | ~21B total, ~3.6B active per token | Consistent across sources |

The layer count and hidden size matching our already-verified pilot data is a
real cross-check, not a coincidence — it means the rest of this row (GQA
shape, MoE shape) is very likely correct for the same model.

### 2.2 Qwen3.8-27B — lower confidence, and architecturally unusual

Sources here are mostly secondary hosting/blog pages (NVIDIA NIM catalog,
vLLM recipe page, several SEO-style model-spec blogs), not an official model
card I could read directly. The release date these sources give (2026-08-14)
is after my knowledge cutoff, so I can't cross-check this one from what I
already knew — **this section needs direct confirmation from the official
Hugging Face model card / config.json before Stage 8 design proceeds.**

| Property | Value | Confidence |
| --- | --- | --- |
| Layers | 64, in 16 repeated groups | Low-moderate |
| Hidden size | 5,120 | Low-moderate |
| **Attention mix** | Only 16 of 64 layers use full (grouped-query) attention; the other 48 use **Gated DeltaNet** — linear attention with a constant-size recurrent state instead of a growing KV cache | Moderate (matches Alibaba's published Qwen3-Next hybrid-attention design pattern) |
| Params | ~27B language + vision encoder (~28B total) | Low-moderate |
| Context | 262,144 native, extendable via YaRN | Low |

**The important part if this is even approximately right:** this is not a
standard transformer. 48 of 64 layers would have no K/V tensors at all in the
usual sense — they'd have a **recurrent state** (PROJECT_PIPELINE.md Stage 8
literally has a bullet for this: "recurrent states where applicable"). That
means the capture harness needs two different code paths for this one model,
not a single Q/K/V hook applied uniformly. This is worth confirming early,
because it changes the capture design, not just a config parameter.

## 3. Draft tensor inventory

Per PROJECT_PIPELINE.md Stage 8: "residual streams, Q/K/V, attention weights,
per-head outputs, intermediate feed-forward activations, recurrent states
where applicable, and routing assignments/scores where applicable." Mapped to
what each architecture above actually has:

| Capture point | GPT-OSS 20B | Qwen3.8-27B | Notes |
| --- | --- | --- | --- |
| Residual stream (`post_attention`, `post_mlp`) | Already captured (pilot) | New | Same flat-vector format as the pilot extends directly |
| Q, K, V | New — Q: 64×64=4,096/layer; K,V: 8×64=512/layer each (GQA) | New for the 16 full-attention layers only | Per §4, raw storage cost is large even without attention weights |
| Attention weights (softmax matrix) | New, **flagged infeasible as-is** | New, **flagged infeasible as-is** (and only meaningful for 16/64 layers) | Scales with context length; see §4 |
| Per-head output (pre-`o_proj`) | New — same shape as Q (4,096/layer) | New, for full-attention layers | Largely redundant with Q if both are kept; pick one |
| Intermediate FFN / expert activations | New — per active expert (top-4 of 32), width unconfirmed | New — standard FFN per layer | GPT-OSS width needs confirming from config, not estimated here |
| Routing assignments/scores | New — top-4 indices + 32 router logits/layer | N/A (not MoE per the above) | Small; cheap to capture in full |
| Recurrent state | N/A | New — DeltaNet state for 48/64 layers, shape unconfirmed | Needs its own code path; see §2.2 |

For every row that ships, Stage 8's own checklist applies: record tensor
type, shape, dtype, axes, masks, and token boundaries in the manifest, and
validate completeness before trusting any capture run.

## 4. Why "capture everything" doesn't fit — a storage budget

The current pilot captures 2 sites × 2,880 BF16 values/token × 24 layers,
measured at ~242.6 KB/token file-encoded (`DATASET.md`/`TENSOR_INVENTORY.md`
totals), for 3 games / 135,336 tokens / 30.57 GiB.

Adding GPT-OSS 20B's Q/K/V alone (§3, excluding attention weights and FFN
detail) is **~5,120 values/layer × 24 layers = 122,880 values/token**, about
**21× the current per-token footprint**, before compressing anything. Scaled
to just the existing 3-game pilot, that's roughly 640 GiB for Q/K/V alone,
on top of the 30.57 GiB already captured — and the full project's original
corpus has ~67× more games (200 vs. 3) than the pilot.

**Attention weights are the real problem, not a quantification detail.**
Unlike every other row in §3, a softmax attention matrix scales with context
length, not just hidden size: 64 heads × up to-20,000-token context (the
`high` reasoning-effort generation setting already in use, per
`source_manifest.json`) **per query token, per layer.** Even for a single
layer capturing a single query token's full attention row, that's already
comparable in size to that token's entire current residual-stream capture;
capturing it for every token, every head, every layer of a 20,000-token
generation is multiple orders of magnitude larger than everything else in
this document combined. This is not something a cleverer codec fixes — it is
a decision about what to capture at all.

**Recommendation to bring to the model/runtime owners:** do not capture full
attention-weight matrices at the same scale as everything else. Either skip
them entirely (per-head *output* already captures what the attention
mechanism produced, if not how it got there) or capture them only for a
small, explicitly sampled/windowed subset of tokens/layers/heads as a
separate, clearly-scoped sub-experiment — not as part of the main Stage 8
capture. Also decide *now* whether Stage 8 runs on the full historical
200-game corpus or stays at pilot scale (3 games, or a slightly larger fixed
sample) — the difference is roughly two orders of magnitude in both storage
and GPU-hours, and the pipeline document's "expanded capture scope" language
doesn't currently pin this down.

## 5. Open questions for the model/runtime owners (blocking)

1. **Compute access.** Qwen3.8-27B (and likely GPT-OSS 20B with the expanded
   tensor set, given §4) won't run on this machine's single RTX 4090 (24GB).
   The original pilot's `source_manifest.json` shows it was generated on a
   4-GPU node. Is that (or equivalent) available again, and for how long?
2. **Qwen3.8-27B's architecture** — confirm §2.2 against the real model card
   before any capture code is written; the hybrid attention/recurrent-state
   split changes the capture design, not just its parameters.
3. **Attention-weight scope** — confirm the §4 recommendation (skip full
   matrices, or scope a small separate sub-experiment) rather than attempting
   full capture.
4. **Corpus scope** — pilot-scale (3 games) extended to new tensor families,
   vs. the full 200-game corpus. This should be a deliberate choice, not a
   default.
5. **GPT-OSS 20B expert FFN width** — needed to size the routing/FFN rows in
   §3 precisely; not estimated here since I don't have a confirmed number.
6. **Model revision/weight identifiers** — neither model has a pinned
   revision recorded anywhere in this project yet (also flagged in
   [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) §1.1 for the existing
   GPT-OSS 20B pilot). Stage 8 needs this nailed down from the start, not
   discovered after capture, since the pipeline's own rule is to "freeze
   checkpoints and record immutable identifiers" before extraction.

## 6. What's NOT in this document

This is a planning document, not a capture harness. No code has been written
for Stage 8. The actual implementation (streaming capture path, manifest
schema, completeness validation) should wait until §5's questions have
answers — writing capture code against an unconfirmed architecture (Qwen3.8-27B
especially) risks a rewrite once the real config.json is in hand.
