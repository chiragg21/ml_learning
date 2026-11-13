# Model Versions: v0 → v1 → v2 → v3

This project implements a transformer from scratch, one version at a
time, each version adding a handful of real architectural ideas used in
modern LLMs. This file explains what each version contains, why each
piece exists, and where to find it in code. It assumes you know roughly
what a neural network is, but nothing about transformers specifically.

If you just want to *run* something (train a model, generate text), see
[README.md](README.md) instead - this file is about what the models
themselves are made of.

Want more depth than this file gives - actual tensor shapes, line-by-line
mechanism, and the "why can't I just..." questions (can I swap tokenizers
freely? why does the model have to match the tokenizer it was trained
with?)? See [LEARNING.md](LEARNING.md), the deep-dive companion to this
file.

## Before v0: the five-minute transformer primer

A transformer for a task like translation has two halves:

- An **encoder** reads the whole source sentence at once and produces
  one vector per input token ("what does this sentence mean").
- A **decoder** generates the target sentence one token at a time,
  looking at (a) what it's generated so far and (b) the encoder's output,
  via a mechanism called **attention**.

**Attention**, in one sentence: for each position, look at every other
position, decide how relevant each one is (a learned "query vs key"
comparison), and pool their values together weighted by that relevance.
*Self-attention* means a sequence attending to itself (how encoder
layers build context, and how the decoder looks at its own
already-generated tokens). *Cross-attention* means the decoder attending
to the encoder's output (how it looks back at the source sentence).

Everything below is refinements of exactly those two ideas.

---

## v0 - the vanilla transformer

**Code:** `src/models/v0/`, `src/tokenizers/v0/tokenizer_v0.py`
**Try it:** `configs/toy_copy_v0.py` (fast, synthetic), `configs/translation_v0.py` (real)

v0 is the original 2017 "Attention Is All You Need" transformer, as
plainly as possible - nothing fancy, so it's a clean baseline to compare
every later version against.

| Piece | What it is | File |
|---|---|---|
| Tokenizer | Character-level: every character is its own token | `tokenizer_v0.py` |
| Position info | A learned vector added to each token embedding, one per absolute position (0, 1, 2, ...) | `encoder_v0.py` / `decoder_v0.py` |
| Attention | Standard multi-head attention (`nn.MultiheadAttention`) | both |
| Normalization | LayerNorm, applied **after** each sublayer's output is added back ("Post-Norm") | both |
| Feed-forward | Two linear layers with a GELU activation in between | both |
| Training | Plain AdamW, optional linear LR warmup, no decay afterward | `src/training/trainer.py` |

### Post-Norm, visually

```
Input X
   │
   ▼
Attention(X)
   │
   ▼
LayerNorm(X + Attention(X))   <- normalize AFTER adding the residual
   │
   ▼
FFN(...)
   │
   ▼
LayerNorm(... + FFN(...))
   │
   ▼
Output
```

### Why a character tokenizer?

It's the simplest possible tokenizer - no vocabulary-building algorithm,
no out-of-vocabulary tokens, works on any text immediately. The cost:
sequences get long (a 20-word sentence is ~100+ characters), and the
model has to learn spelling from scratch rather than getting whole
words/subwords handed to it. Fine for this project's scale; a real LLM
would use a subword tokenizer (BPE or similar) instead.

### Parameter counting

Every v0 file has the actual parameter-count formula worked out in its
docstring (e.g. `encoder_v0.py`'s `4*D*D + 2*D*F + 9*D + F` per layer) -
worth reading even if you don't care about the exact numbers, since it's
a good way to build intuition for where a transformer's parameters
actually live (mostly: attention's four D×D matrices, and the
feed-forward network's two D×F matrices).

### v0's limitations (why v1 exists)

- The learned position table only works up to the exact `max_seq_len`
  it was trained with - it can't generalize to longer sequences.
- Post-Norm makes very deep stacks harder to train (the residual
  stream gets renormalized every layer instead of staying "clean"),
  which is part of why early transformers topped out shallower than
  today's models.
- GELU FFN, LayerNorm, and a separate (untied) output projection are
  all "fine but not what anyone actually uses anymore."

---

## v1 - Pre-Norm, RMSNorm, RoPE, SwiGLU, weight tying

**Code:** `src/models/v1/` (`modules_v1.py`, `encoder_v1.py`, `decoder_v1.py`, `transformer_v1.py`)
**Try it:** `configs/toy_copy_v1.py`, `configs/translation_v1.py`

v1 doesn't change the overall encoder-decoder *shape* - it's a drop-in
replacement for v0 (same `forward()` signature, same training/inference
code works unchanged). What changes is *how* each piece works, updating
every component to what's actually used by GPT-2-onward/LLaMA/Mistral-era
models. Switching from v0 to v1 (or between datasets) is a one-line
config change - see `configs/toy_copy_v1.py` vs `configs/toy_copy_v0.py`,
which differ only in `version="v1"` and one scheduler line.

| Piece | v0 | v1 | Why change |
|---|---|---|---|
| Normalization | LayerNorm, Post-Norm | RMSNorm, **Pre-Norm** | Trains more stably at depth; RMSNorm is cheaper and just as good |
| Position info | Learned absolute position table | **RoPE** (rotary position embedding) | Zero extra parameters; generalizes beyond the trained sequence length; encodes *relative* position |
| Feed-forward | GELU, 2 matrices | **SwiGLU**, 3 matrices (gated) | Consistently better quality at matched compute (Shazeer, 2020) |
| Output head | Separate learned matrix | **Tied** to the input embedding | Fewer parameters; standard since GPT-2 |
| LR schedule | Linear warmup, no decay | Linear warmup, then **cosine decay** | The de facto standard schedule for training transformers from scratch |

### Pre-Norm, visually (contrast with v0 above)

```
Input X
   │
   ├───────────────┐
   │               │
   ▼               │
RMSNorm(X)         │   <- normalize the INPUT to the sublayer, not its output
   │               │
   ▼               │
Attention(...)      │
   │               │
   └──────►  X + Attention(RMSNorm(X))
                    │
                   ...same pattern for the FFN...
                    │
                    ▼
                 Output
```
(Note there's no norm *after* the last residual add here - instead, v1
applies **one** final RMSNorm after the whole layer stack, inside
`EncoderV1`/`DecoderV1`, to keep the residual stream at a sane scale
before anything downstream reads it.)

The practical effect: in Post-Norm, every layer renormalizes the
accumulated residual stream, which empirically makes gradients harder to
propagate cleanly through very deep stacks. In Pre-Norm, the residual
stream itself is never directly normalized - each sublayer reads a
normalized *copy* of it, but what gets added back is unnormalized. That
keeps a clean gradient path from the output all the way back to the
input, which is a big part of why 50+ layer transformers are trainable
at all.

### RoPE in one paragraph

Instead of *adding* a position vector once at the start (v0), RoPE
*rotates* each attention head's query/key vectors by an angle
proportional to their position, every time attention runs. The reason
this is popular: rotating two vectors by their own absolute positions and
then taking their dot product leaves only the *difference* between those
positions in the result - so the model learns about relative distance
between tokens "for free," and the mechanism works at any sequence
length, not just ones seen in training. See the full derivation/comments
in `src/models/v1/modules_v1.py`'s `RotaryEmbedding` class.

### SwiGLU in one paragraph

A "gated" feed-forward network: one matrix produces a signal, a second
produces a parallel gate that multiplies it elementwise, a third
projects back down. Three matrices instead of GELU's two, but
empirically better at the same total compute budget - this is the FFN
used by LLaMA, Mistral, PaLM, and most current open models. See
`SwiGLUFeedForward` in `modules_v1.py`.

### Weight tying - and a real gotcha worth knowing

v1 reuses the decoder's input embedding matrix as the output projection
(`llm_head`) instead of learning a second, separate one - same
parameters serve both "turn a token ID into a vector" and "turn a vector
back into token probabilities." This saves `vocab_size * d_model`
parameters outright.

The catch, discovered while building this: `nn.Embedding`'s default
initialization (`Normal(0, 1)`) is the wrong scale once that same matrix
is also used as a `Linear` layer's weight (which PyTorch instead
default-initializes much smaller, `~1/sqrt(d_model)`). Left alone, this
made the model's initial loss blow up to the 50s instead of the
expected `~log(vocab_size)` for a freshly initialized model - a
realistic example of how a subtly wrong initialization can silently
sabotage training without throwing any error. `TransformerV1` fixes
this with an explicit `_init_weights` pass (std=0.02 on every
Linear/Embedding, the same convention GPT-2/nanoGPT use) - see the
comment in `transformer_v1.py` for the full explanation.

### "Proper experiment infrastructure"

This isn't a model-code change, but it's part of v1's checklist: the
training/inference pipeline around the model (config-driven
`ModelFactory`/`TokenizerFactory`/`DataFactory`, the experiment-hash-based
artifact system, `Trainer` with mixed precision/`torch.compile`/
checkpointing/MLflow logging) is what makes it possible to switch
versions and datasets with a one-line config change instead of rewriting
scripts. That infrastructure lives in `src/training/trainer.py`,
`src/models/factory.py`, `src/data/factory.py`, `src/tokenizers/factory.py`,
and `artifacts/manager.py` - shared by every model version, v0 included.

---

## Tokenizers (independent of model version)

Tokenizers are chosen separately from the model via `TokenizerConfig`, so
any tokenizer below works with any model version. All share the same
interface (`PAD/UNK/BOS/EOS` ids, `encode`, `decode`, `vocab_size`) and
are registered in `src/tokenizers/factory.py`.

| Key | What it is | File |
|---|---|---|
| `char_v0` | One token per character | `src/tokenizers/v0/tokenizer_v0.py` |
| `bpe_v1` | Byte-level BPE trained on our corpus | `src/tokenizers/v1/tokenizer_v1.py` |
| `wordpiece_v2` | WordPiece trained on our corpus | `src/tokenizers/v2/tokenizer_v2.py` |
| `unigram_v3` | Unigram (SentencePiece-style) trained on our corpus | `src/tokenizers/v3/tokenizer_v3.py` |
| `pretrained_v4` | A tokenizer someone else already trained, loaded from the Hugging Face Hub or a local `tokenizer.json` | `src/tokenizers/v4/tokenizer_v4.py` |

**WordPiece (`wordpiece_v2`)** is the tokenizer BERT uses. It builds its
vocabulary like BPE, but merges the pair that most increases corpus
likelihood instead of the most frequent pair, and at encoding time
greedily matches the longest vocabulary entry from the left of each word.
Continuation pieces carry a `##` prefix ("playing" -> `play`, `##ing`).
Two differences from `bpe_v1` worth knowing: it works on unicode
characters rather than bytes, so a character never seen in training
becomes `<UNK>`; and decoding re-joins words with single spaces, so it
doesn't reproduce the original whitespace exactly.
**Try it:** `configs/toy_copy_wordpiece_v2.py`.

**Unigram (`unigram_v3`)** is the algorithm behind SentencePiece. BPE and
WordPiece grow a vocabulary by merging pairs; Unigram goes the other way,
starting from a large set of candidate substrings and repeatedly pruning
the ones whose removal hurts corpus likelihood least. Each piece has a
learned probability, and encoding picks the most probable segmentation
of each word. Like SentencePiece, whitespace is an ordinary symbol
(rewritten to `▁`), so word-initial pieces carry their own marker and
`decode()` reproduces the input text exactly, as long as every character
was seen in training; an unseen character becomes `<UNK>`.
**Try it:** `configs/toy_copy_unigram_v3.py`.

**Pretrained (`pretrained_v4`)** loads an already-trained tokenizer
(GPT-2's byte-level BPE, BERT's WordPiece, ...) instead of training one
on our corpus, which is ignored. Set `source` in the tokenizer
`parameters` to a Hub id (downloaded on first use, so it needs network)
or a path to a local `tokenizer.json`; the file's own normalizer,
pre-tokenizer, model and decoder are used unchanged. The pipeline needs
`<PAD>/<UNK>/<BOS>/<EOS>` ids that pretrained vocabularies often lack
(GPT-2 only has `<|endoftext|>`, and byte-level BPEs have no unknown
token), so missing ones are appended as new ids, making `vocab_size`
slightly larger than the original. Pass `pad_token`/`unk_token`/
`bos_token`/`eos_token` to reuse an existing token instead (e.g.
`eos_token="<|endoftext|>"`, or `unk_token="[UNK]"` for BERT); a token
that isn't in the vocabulary raises an error. The pretrained
post-processor's own markers (BERT's `[CLS]`/`[SEP]`) are never added,
so BOS/EOS only appear via `add_bos`/`add_eos`. Tests use a tiny locally
saved tokenizer, since the Hub isn't reachable from CI-like sandboxes.
**Try it:** `configs/toy_copy_pretrained_v4.py` (needs network).

All planned tokenizers are now implemented.

---

## v2 - making it fast and able to handle long sequences

**Code:** `src/models/v2/` (`modules_v2.py`, `encoder_v2.py`, `decoder_v2.py`, `transformer_v2.py`)
**Try it:** `configs/toy_copy_v2.py` (GQA only), `configs/toy_copy_efficient_v2.py` (everything below)

v2's theme is *efficiency*: everything so far has been about making the
model learn *better*; v2 is about making it *affordable to run*,
especially at generation time and with long sequences. `transformer_v2`
is a drop-in replacement for v1 (same `forward()` signature) and reuses
v1's RMSNorm, RoPE and SwiGLU modules unchanged.

| Piece | v1 | v2 | Why change |
|---|---|---|---|
| Attention | Every head has its own Q, K and V projection | **GQA**: one K/V head shared by a group of query heads (`num_kv_heads`) | Shrinks the K/V projections now and, later, the KV cache |
| Attention span | Every token attends to every (unmasked) token | **Local attention**: optional sliding window (`window_size`) on self-attention, optionally computed over the band only (`banded`) | Limits how far each token looks, and the banded kernel makes that cheaper |
| Generation | Recomputes the whole prefix every step | **KV cache** | Per-step cost O(n) instead of O(n²) |
| Batches | Padded to the longest example | **Sequence packing**, length bucketing, streaming, prefetch | Fewer wasted `<PAD>` tokens, no idle GPU |
| Training memory/speed | fp32, all activations stored | **bf16/fp16 mixed precision**, **gradient checkpointing**, **torch.compile** | Less memory, faster matmuls |

**GQA (Grouped-Query Attention)** keeps one query projection per head
but shares each key/value head across `num_heads / num_kv_heads` query
heads. `num_kv_heads == num_heads` is ordinary multi-head attention
(exactly v1's attention); `num_kv_heads == 1` is multi-query attention.
Applied to both self- and cross-attention. Set `num_kv_heads` in the
encoder/decoder `parameters` of the config (omitting it means no
sharing). The saving at `D=256, H=8, num_kv_heads=2` is 98,304 attention
parameters per module (262,144 -> 163,840), but the real payoff is
inference memory: a KV cache stores K and V per past token, so its size
scales with `num_kv_heads`, which is why the KV cache below is the
natural next step. See `GroupedQueryAttentionV2` in `modules_v2.py`;
properties are checked in `tests/test_gqa_v2.py`.

**Local attention (sliding window)** restricts self-attention so each
token only sees positions less than `window_size` away from itself
(itself plus `window_size - 1` neighbours per side; in the decoder the
causal mask leaves just the previous ones). Set `window_size` in the
encoder/decoder `parameters`; omitting it means full attention, and a
window at least as long as the sequence is identical to full attention.
Cross-attention is never windowed (decoder and encoder positions aren't
in the same frame), and passing a window to it raises an error. Stacked
layers still carry information further than one window: after L layers a
token is influenced by tokens up to `L * (window_size - 1)` positions
away. Honest limitation: the window is applied as a mask over the full
score matrix, so it changes *what* is attended to but not the O(n²)
compute or memory - a banded kernel would be needed for the speedup. One
gotcha found while building it: a `<PAD>` query whose entire window is
also padding has every key masked, which makes softmax return NaN; those
rows are zeroed instead. See `_local_window_mask` in `modules_v2.py`;
properties are checked in `tests/test_local_attention_v2.py`.
**Try it:** `configs/toy_copy_local_v2.py`.

**KV cache** (`init_cache`/`encode`/`decode_step` on `TransformerV2`,
used automatically by `Generator`, `use_cache=False` to disable). The
decoder's keys and values for tokens already generated never change, so
each step projects only the newest token, appends its (RoPE-rotated,
`num_kv_heads`-sized) K/V to a per-layer cache and attends over it;
cross-attention caches the projected encoder output once. Generation goes
from re-running the whole prefix at every step (O(n²) work per step) to
O(n) per step. With a cache, self-attention
is causal by construction (positions decide what is visible), and with a
`window_size` the cache is trimmed to the `W-1` keys still in reach, so
its size stops growing. Properties checked in
`tests/test_model_efficiency_v2.py` (stepwise logits equal the full
forward pass, with and without a window) and
`tests/test_data_pipeline_v2.py` (cached and uncached greedy generation
agree).

**Banded attention kernel** (`banded: True` next to `window_size` in the
encoder/decoder `parameters`). Instead of masking a full S×S score matrix,
each query gathers just the `2W-1` keys inside its window and attends
over that band: O(S·W) compute and memory instead of O(S²), same result
as the masked path (tested for several windows, with causal and padding
masks, and with packing). Honest limitation: it is plain PyTorch
(gather + einsum), not a fused GPU kernel, so the gather copies K/V
`2W-1` times and the speedup only shows once `S` is much larger than
`W`; a custom/FlashAttention-style kernel would be the next step.

**Sequence packing** (`data.extra["pack_sequences"] = True`). Several
short examples share one row, filled up to `max_seq_len` on both the
source and target side, so batches carry far fewer `<PAD>` tokens.
Examples must not see each other, so each row carries segment ids
(1, 2, ... per example, 0 = padding) and the model builds a block-
diagonal mask from them in the encoder, the decoder and cross-attention
(RoPE needs no per-example reset - it only encodes relative distance).
The trainer ignores the one label per boundary that would otherwise ask
the model to predict the next example's `<BOS>`. Packing is static
(rows fixed once, shuffled as whole rows), and validation is never
packed so its metrics stay comparable. `PackedSeq2SeqDataset` in
`src/data/seq2seq.py`; `test_packed_row_matches_separate_examples`
verifies a packed row gives the same logits as the examples alone.

**Mixed precision** (`training.mixed_precision=True`, `precision="bf16"`
or `"fp16"`). The `Trainer` already ran autocast and a `GradScaler`
(fp16 only; CPU falls back to bf16); v2 makes it complete by also
running validation under autocast with the loss computed in fp32, and by
keeping the attention masks/softmax safe in low precision (masked rows
are zeroed, not NaN). Exercised on CPU in the bf16 efficient config; the
fp16 + `GradScaler` path needs a GPU and has not been run here.

**Gradient checkpointing** (`training.gradient_checkpointing=True`).
Each encoder/decoder layer is wrapped in `torch.utils.checkpoint`, so
only layer inputs are kept during the forward pass and activations are
recomputed in backward: less memory for roughly one extra forward pass of
compute. Only active in training mode; a test confirms gradients equal
the un-checkpointed ones. Models without `set_gradient_checkpointing`
(v0, v1) raise a clear error if it is switched on.

**torch.compile** (`runtime.compile_model=True`). Was already an opt-in
flag; v2 makes it work end to end - checkpoints now unwrap the compiled
model, so saved `state_dict` keys have no `_orig_mod.` prefix and load
into an ordinary model. Smoke-tested on CPU (needs a C++ compiler; the
test skips without one). Speedups only show on a GPU with real sizes, and
variable batch shapes can trigger recompilation.

**Efficient data pipeline** (`data.extra` / `runtime`):
- `streaming: True` (+ `shuffle_buffer`) - `StreamingSeq2SeqDataset`
  tokenizes lazily instead of encoding the whole corpus up front, shards
  across DataLoader workers and shuffles with a buffer.
- `bucket_by_length: True` - `BucketBatchSampler` groups similar-length
  examples so batches pad less (shuffled chunks, sorted inside each).
- `runtime.num_workers` with `runtime.extra["prefetch_factor"]` - worker
  processes prepare upcoming batches while the model computes; host-to-GPU
  copies are `non_blocking` from pinned memory on CUDA.
Streaming can't be combined with packing or bucketing (both need all
lengths up front); the options live in `extra` dicts so existing
experiment hashes don't change.

## v3 - scaling up and getting smarter about compute

**Code:** `src/models/v3/` (`modules_v3.py`, `encoder_v3.py`, `decoder_v3.py`, `transformer_v3.py`), `src/generator/inference_compute.py`, `src/quantization/quantize.py`, `src/evaluation/advanced.py`
**Try it:** `configs/toy_copy_v3.py`, `scripts/quantize.py`

v3's theme is *doing more without doing everything, every time*.
`transformer_v3` extends v2 (all of v2's features keep working with it)
and with its new options off it computes exactly what v2 does. The
last three items below don't change the model at all - they are tools
around it.

| Piece | v2 | v3 | Why change |
|---|---|---|---|
| Feed-forward | One dense SwiGLU per layer | **MoE**: several expert FFNs, **top-k routing**, **load-balancing loss** | More parameters without more compute per token |
| Depth | Every token uses every layer | **Dynamic depth**: a per-token router may skip a layer | "Easy" tokens can use less of the stack |
| Generation | One greedy/sampled answer | **Inference-time compute**: best-of-N, majority vote, verification | Better output from more compute, same model |
| Weights | fp32 | **Quantization**: int8/int4 weight-only | Several times smaller model |
| Evaluation | Loss, perplexity, token accuracy | BLEU, chrF, exact match, CER, judge hook, benchmark suites, calibration | Scores what the model actually generates |

**MoE and top-k routing** (`MoEFeedForward`). Each token is scored by a
small linear router over `num_experts` experts; the `top_k` best run and
their outputs are summed, weighted by the renormalised router
probabilities (which is also what gives the router a gradient). Total
parameters grow about `num_experts`-fold while per-token compute stays at
`top_k` experts. Set `num_experts`, `top_k` and `moe_every` (convert
every N-th layer; default every layer) in the encoder/decoder
`parameters`. Honest limitations: it is *dropless* (no per-expert capacity
limit) and experts run in a Python loop over their tokens - correct and
simple, not the fused grouped matmul a real system uses.

**Load balancing.** Left alone, routers collapse onto a few favourite
experts. The Switch-Transformer auxiliary loss `E * sum_e f_e * P_e`
(`f_e`: share of routing slots sent to expert `e`; `P_e`: mean router
probability for `e`) equals 1 when balanced and grows when collapsed; its
gradient flows through `P_e`. Padding tokens are excluded.
`moe_aux_weight` scales it; the model returns the weighted sum of all
auxiliary losses as `output.aux_loss` and `Trainer` adds it to the
cross-entropy. `model.set_usage_tracking(True)` + `routing_usage()` show
how tokens were actually routed.

**Conditional computation / dynamic depth** (`dynamic_depth: True`,
`depth_capacity`). A `DepthRouter` per layer decides per token whether
the layer's update is applied (`sigmoid(score) > 0.5`); a skipping token
keeps its residual stream unchanged. A per-token threshold, rather than
"top 50% of the sequence" as in Mixture-of-Depths, is deliberate: it
needs no look at future tokens, so training and KV-cached generation
agree. Gradients use a straight-through estimator, and a budget loss
pulls the average usage toward `depth_capacity`. The router starts open
(almost every token uses every layer). Honest limitation: skipped
tokens' updates are *masked out*, not avoided - attention and the FFN
still run on the full batch - so this trains and demonstrates the
routing but does not yet save compute; that needs gathering selected
tokens into a smaller tensor, which interacts with attention, RoPE
positions and the KV cache.

**Inference-time compute** (`src/generator/inference_compute.py`).
`best_of_n` samples N candidates and keeps the best by the model's own
length-normalised log-probability, or by any `verifier(source_ids,
candidate_ids) -> score` you supply; `majority_vote` returns the answer
most candidates agree on (self-consistency); `verify_and_vote` filters by
a verifier first. Quality improves with N only as far as the scorer
prefers good outputs, and cost grows linearly with N. Works with any
model version through `Generator`.

**Quantization** (`quantize_model` in `src/quantization/quantize.py`).
Post-training, weight-only, symmetric, per-group: each group of
`group_size` weights in a row shares a float scale and is stored as int8
(4x smaller) or int4 (packed two per byte, ~8x smaller before scales).
Weights are dequantized on the fly, so activations stay float - this
saves storage, not arithmetic. The tied embedding/output matrix and the
small routers are kept in float. Int4 usually wants `group_size` 32-128.
`python -m scripts.quantize --config ... --bits 8` evaluates a trained
experiment before and after.

**Advanced evaluation** (`src/evaluation/advanced.py`).
`evaluate_generation` generates for held-out pairs and scores exact
match, character error rate, BLEU, chrF and length ratio, plus an
optional `judge(source, reference, hypothesis) -> [0, 1]` callable for
qualitative scoring (no judge model is bundled - plug in your own);
`BenchmarkSuite`/`run_benchmarks` score named held-out sets separately;
`calibration_error` measures whether the model's confidence matches its
accuracy. BLEU/chrF reward surface overlap, not meaning, so treat them as
proxies.

---

## Trying any of this yourself

```bash
# fast synthetic smoke test (seconds, validates the pipeline)
uv run python -m scripts.train --config configs/toy_copy_v1.py   # or _v0.py

# real training run (minutes on GPU, hours on CPU)
uv run python -m scripts.train --config configs/translation_v1.py   # or _v0.py

# generate text from a trained experiment
uv run python -m scripts.generate --config configs/translation_v1.py --text "hello"
```

See [README.md](README.md) for the full pipeline walkthrough (how configs
map to experiments, how to add a new data source, Colab/Kaggle setup via
`bootstrap_colab.sh`, etc).
