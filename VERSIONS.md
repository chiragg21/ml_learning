# Model Versions: v0 → v1 → v2 → v3

This project implements a transformer from scratch, one version at a
time, each version adding a handful of real architectural ideas used in
modern LLMs. This file explains what each version contains, why each
piece exists, and where to find it in code. It assumes you know roughly
what a neural network is, but nothing about transformers specifically.

If you just want to *run* something (train a model, generate text), see
[README.md](README.md) instead - this file is about what the models
themselves are made of.

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

Still not implemented: Unigram/SentencePiece, and pretrained tokenizers.

---

## Future: v2 - making it fast and able to handle long sequences

Not yet implemented - this is the plan. v2's theme is *efficiency*:
everything so far has been about making the model learn *better*; v2 is
about making it *affordable to run*, especially at generation time and
with long sequences.

- **GQA (Grouped-Query Attention):** instead of every attention head
  having its own key/value projections, share one set of keys/values
  across a small group of query heads. Cuts the size of the KV cache
  (below) and memory bandwidth at inference, with little quality loss -
  used by LLaMA 2 70B onward, Mistral, and most current models.
- **KV cache:** during autoregressive generation, the decoder's
  key/value vectors for already-generated tokens never change - caching
  them instead of recomputing the whole sequence's attention at every
  new token turns generation from O(n²) into O(n) per step. This is the
  single biggest practical speedup for text generation and every
  real inference server relies on it.
- **Local/sparse attention:** full self-attention costs O(n²) in
  sequence length. Restricting each position to attend only to a local
  window (or another sparse pattern) makes long sequences tractable.
- **Sequence packing:** instead of padding every batch to its longest
  sequence (wasting compute on `<PAD>` tokens, as the current pipeline
  does), concatenate multiple short sequences into one row up to the
  max length. More total useful tokens processed per batch.
- **BF16/FP16 mixed precision:** training in a16-bit floating point
  format instead of 32-bit, for roughly 2x less memory and (on a GPU
  with tensor cores) significantly faster matmuls. (A first version of
  this is already wired into `Trainer` - see `training.mixed_precision`
  in `src/datamodels.py` - v2 is where it becomes load-bearing rather
  than optional.)
- **Gradient checkpointing:** trade compute for memory by not storing
  every layer's activations during the forward pass, recomputing them
  during backprop instead - lets you train bigger models/longer
  sequences on the same GPU memory, at some training-time cost.
- **torch.compile:** JIT-compiles the model's computation graph for
  faster execution (also already wired as an opt-in flag -
  `runtime.compile_model` - v2 is where it's expected to matter).
- **Efficient data pipeline:** streaming data loading, overlapping data
  loading with compute, and other plumbing to make sure the GPU is never
  sitting idle waiting for the next batch.

## Future: v3 - scaling up and getting smarter about compute

v3's theme is *doing more without doing everything, every time*:

- **MoE (Mixture of Experts):** replace the single dense feed-forward
  network with several "expert" FFNs, where each token is routed to
  only a few of them instead of all of them. Lets total parameter count
  grow much larger while the compute cost *per token* stays small -
  this is how models like Mixtral and GPT-4 (reportedly) scale.
- **Top-k routing:** the specific mechanism MoE needs - a learned
  router picks the top-k experts for each token.
- **Load balancing:** without an explicit nudge, routers tend to send
  every token to the same few experts, wasting the rest. An auxiliary
  loss term encourages tokens to spread out across experts evenly.
- **Conditional computation / dynamic depth:** let "easy" tokens skip
  some layers and "hard" tokens use the full stack, instead of every
  token always doing the same fixed amount of work.
- **Inference-time compute:** spending extra compute *at generation
  time* (e.g. multiple reasoning passes, self-verification, sampling
  many candidates and picking the best) to improve output quality
  without growing the model itself - the idea behind "reasoning models."
- **Quantization concepts:** compressing trained weights to lower
  precision (int8, int4) after training, trading a little accuracy for
  much cheaper storage/inference.
- **Advanced evaluation:** moving past loss/perplexity/token-accuracy
  (what `src/evaluation/evaluator.py` measures today) toward held-out
  benchmark suites and qualitative/LLM-judged generation quality -
  the kind of evaluation a model needs once "does it produce a lower
  number" stops being a good enough question.

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
