# Learning Guide: a deep dive into every component

[VERSIONS.md](VERSIONS.md) tells you *what* each version adds and gives
you the one-paragraph *why*. This file is the companion for when that
isn't enough - it walks through the actual code, component by
component: what shape of tensor goes in, what comes out, what every
line in between is doing, and why it's written that way rather than
some other way. It also answers the cross-cutting "why can't I just..."
questions that only make sense once you've seen how the pieces actually
fit together (tokenizers, models, configs, experiment identity).

Read this alongside the source files it names - it quotes real
variable and class names on purpose, so you can go find them.

---

## v0 - the vanilla transformer, line by line

**Code:** `src/models/v0/encoder_v0.py`, `decoder_v0.py`, `transformer_v0.py`, `datamodels_v0.py`

### `EncoderLayerV0`

Input shape `(B, S, D)` - batch, sequence length, `d_model`.

```python
attention_output, _ = self.self_attention(x, x, x, key_padding_mask=key_padding_mask)
x = self.norm1(x + self.dropout(attention_output))
ffn_output = self.ffn(x)
x = self.norm2(x + self.dropout(ffn_output))
```

`self.self_attention` is `nn.MultiheadAttention` - PyTorch's built-in,
not custom code. Query, key and value are all `x` (self-attention), so
internally it projects `x` through three `D×D` matrices, splits into
`num_heads` heads of size `D/num_heads`, computes
`softmax(QKᵗ/√d_head)·V` per head, concatenates the heads back to `D`,
and projects once more through a fourth `D×D` matrix. Shape in and out
is the same: `(B, S, D)`. `key_padding_mask` is `(B, S)` boolean, `True`
at `<PAD>` positions - `nn.MultiheadAttention` broadcasts it to block
every query from attending to those key positions.

The residual add (`x + dropout(attention_output)`) then LayerNorm is
**Post-Norm**: the sublayer sees the *raw* `x`, and normalization only
happens after the result is folded back in. `self.ffn` is
`Linear(D, F) -> GELU -> Linear(F, D)` - a plain two-matrix MLP with a
smooth nonlinearity. No surprises here; this is the file to read if
you've never seen a transformer layer's internals at all.

### `DecoderLayerV0`

Same shape of layer, with two attention calls instead of one:

```python
self_attention_output, _ = self.self_attention(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
x = self.norm1(x + dropout(self_attention_output))
cross_attention_output, _ = self.cross_attention(x, memory, memory, key_padding_mask=memory_key_padding_mask)
x = self.norm2(x + dropout(cross_attention_output))
ffn_output = self.ffn(x)
x = self.norm3(x + dropout(ffn_output))
```

`tgt_mask` is the causal mask: an `(S, S)` boolean upper-triangular
matrix (`torch.triu(..., diagonal=1)`), `True` above the diagonal. Query
position `i` can only see key positions `≤ i` - this is what makes
training on a whole target sequence at once equivalent to generating it
left-to-right one token at a time (teacher forcing works because the
model is mathematically prevented from peeking ahead).

Cross-attention's query is the decoder's own `x` (shape `(B, T, D)`,
`T` = target length), but its key/value come from `memory` - the
*encoder's* output, shape `(B, S, D)`, `S` = source length. These two
sequences can have completely different lengths; that's fine, because
attention only requires the query and key to share `d_model`, not
length. This is literally how "look at the source sentence while
writing the translation" is implemented: at every decoder position, the
query (decoder's current state) scores against every encoder output
position, and pools them weighted by relevance.

### `EncoderV0.forward` / `DecoderV0.forward`

```python
x = self.token_embedding(tokens)          # (B, S) -> (B, S, D)
positions = torch.arange(seq_len, device=tokens.device)
x = x + self.position_embedding(positions)  # broadcasts (S, D) over the batch
```

`token_embedding` is a lookup table, `vocab_size × d_model` - row `i` is
the learned vector for token id `i`. `position_embedding` is a *second*
lookup table, `max_seq_len × d_model` - row `i` is a learned vector for
"this token is at position `i`". Adding them means the network can
never fully separate "what word is this" from "where is it" - both are
baked into one vector before layer 1 even runs. This table has a hard
`max_seq_len` ceiling: there is no row for position `max_seq_len`, so a
longer sequence raises `ValueError("Sequence Length exceeds max_seq_len")`
at the top of `forward` - this is the exact error you'd hit if you fed
this architecture a sequence longer than it was configured for (and
the error that originally motivated `TokenizedSeq2SeqDataset`'s
tokenized-length filtering - see the tokenizer FAQ below).

### `TransformerV0`

```python
self.llm_head = nn.Linear(self.decoder.d_model, self.decoder.target_vocab_size)
...
logits = self.llm_head(decoder_output)
```

One more `D × V` matrix turns each decoder position's `D`-dim vector
into a score per vocabulary entry. This is a *separate*, independently
learned matrix from the input embedding - v1 changes that (weight
tying, below). `forward()`'s signature -
`(source_ids, target_ids, source_key_padding_mask=None, target_key_padding_mask=None) -> TransformerOutput(logits=...)`
- is the contract every later version keeps, which is *why* swapping
model versions is a config change rather than a rewrite (see the FAQ
"why models swap more freely than tokenizers").

### Parameter accounting (why it matters beyond trivia)

`encoder_v0.py`'s docstring derives `4*D*D + 2*D*F + 9*D + F` per layer:
`4*D*D` is attention's four `D×D` matrices (Q, K, V, O - all full-size
in v0, unlike GQA later), `2*D*F` is the FFN's two matrices, `9*D + F`
is the biases every one of those six matrices carries (v0 is the only
version with biases - v1 onward drops them, see below) plus the two
LayerNorms' `gamma, beta` (`2*D` each... actually `2*D` total for both
params together per norm, `D` and `D`). Working through this formula
once is the fastest way to internalize where a transformer's parameters
actually live: attention's four square matrices and the FFN's two
rectangular ones dominate everything else (the embedding tables scale
with vocabulary, not depth).

---

## v1 - Pre-Norm, RMSNorm, RoPE, SwiGLU, weight tying

**Code:** `src/models/v1/modules_v1.py`, `encoder_v1.py`, `decoder_v1.py`, `transformer_v1.py`

### Why a new `modules_v1.py` file at all

v0 built everything from `nn.MultiheadAttention` and `nn.Sequential`
off-the-shelf. v1 can't do that, because RoPE (below) has to modify Q
and K *between* their linear projection and the attention dot product -
a hook `nn.MultiheadAttention` simply doesn't expose. So v1 writes its
own `MultiHeadAttentionV1`, and once attention is custom, RMSNorm and
SwiGLU get their own small classes too, for symmetry and so every later
version (v2, v3) can import and reuse them unchanged.

### `RMSNorm`

```python
variance = x.pow(2).mean(dim=-1, keepdim=True)
x = x * torch.rsqrt(variance + self.eps)
return x * self.weight
```

Shape in/out: `(B, S, D) -> (B, S, D)`, `self.weight` is `(D,)`.
Compare to LayerNorm (`(x - mean) / sqrt(var) + eps) * gamma + beta`):
RMSNorm skips subtracting the mean and has no learned `beta` - it only
rescales by the root-mean-square magnitude. One fewer reduction
(`mean(x)` is never computed, only `mean(x²)`), fewer parameters (`D`
vs `2*D`), and empirically no quality loss in practice - which is why
every current open LLM (LLaMA, Mistral, Gemma) uses it over LayerNorm.

### `RotaryEmbedding` and `apply_rotary_pos_emb` - the actual mechanism

This is worth tracing fully, because "RoPE rotates Q and K" is true but
not illuminating until you see the shapes.

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim/2,)
positions = torch.arange(max_seq_len).float()                                  # (max_seq_len,)
angles = torch.outer(positions, inv_freq)                                      # (max_seq_len, head_dim/2)
angles = torch.cat([angles, angles], dim=-1)                                   # (max_seq_len, head_dim)
self.register_buffer("cos_cached", angles.cos(), persistent=False)
self.register_buffer("sin_cached", angles.sin(), persistent=False)
```

`inv_freq[i] = theta^(-2i/head_dim)` - a geometric sequence from 1 down
to `1/theta`, so low indices rotate fast (high frequency) and high
indices rotate slowly (low frequency), the same spirit as the original
sinusoidal position encoding. `angles[m, i] = m * inv_freq[i]` is "how
far position `m` has rotated along frequency `i`". The table is
precomputed once for every position up to `max_seq_len` and sliced per
call - zero per-step cost beyond a lookup.

`_rotate_half` and `apply_rotary_pos_emb`:

```python
def _rotate_half(x):
    half = x.shape[-1] // 2
    first_half, second_half = x[..., :half], x[..., half:]
    return torch.cat([-second_half, first_half], dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return x * cos + _rotate_half(x) * sin
```

`x` here is `(B, H, S, head_dim)` - already split into heads. This
treats the first and second halves of `head_dim` as the real and
imaginary parts of `head_dim/2` complex numbers, and multiplies each by
`e^(i·angle)` using the real-arithmetic identity
`(a+bi)(cosθ+i sinθ) = (a cosθ - b sinθ) + i(a sinθ + b cosθ)` - which
is exactly `x*cos + rotate_half(x)*sin` once you check the signs. This
"rotate-half" layout (used by GPT-NeoX/LLaMA) is a fixed relabeling of
dimensions compared to the original paper's *interleaved* pairs
(`(x0,x1), (x2,x3), ...`); the two are mathematically equivalent, just
implemented differently.

**Why this buys you relative position for free**: a rotation is a
linear, *orthogonal* map - rotating two vectors by angles `mθ` and `nθ`
and then taking their dot product gives the same result as rotating one
of them by `(m-n)θ` and leaving the other alone. So `q` at position `m`
dotted with `k` at position `n` depends only on `m - n` - the model
never sees `m` or `n` directly, only their difference, baked into the
attention score itself. That's *why* RoPE generalizes beyond
`max_seq_len`'s training range, unlike v0's lookup table, which has no
row to return for a position it never saw - the relative-distance
behavior from the geometry holds at any `m, n`, the precomputed table
is just a cache for the angles, nothing stops you from computing more.

### `MultiHeadAttentionV1`

```python
q = self._split_heads(self.w_q(query_input))   # (B, H, S, head_dim)
k = self._split_heads(self.w_k(key_value_input))
v = self._split_heads(self.w_v(key_value_input))
if self.use_rope:
    cos, sin = rope(q.shape[-2]); q = apply_rotary_pos_emb(q, cos, sin)
    cos, sin = rope(k.shape[-2]); k = apply_rotary_pos_emb(k, cos, sin)
scores = torch.matmul(q, k.transpose(-2, -1)) * self.head_dim ** -0.5   # (B, H, S_q, S_k)
...
attn_weights = torch.softmax(scores, dim=-1)
context = torch.matmul(attn_weights, v)         # (B, H, S_q, head_dim)
return self.w_o(self._merge_heads(context))     # (B, S_q, D)
```

Note RoPE is applied to Q and K *after* the head split and *before* the
score matmul - it has to be, since it rotates per-head, per-position
vectors. `use_rope=True` for self-attention (query and key share one
sequence's positions - relative distance means something), `use_rope=False`
for cross-attention (decoder query at position `m`, encoder key at
position `n` of a *different* sequence - "relative distance" between
two different indexing schemes carries no real meaning, so RoPE is
skipped there and self-attention on both sides carries all positional
signal instead - see `decoder_v1.py`'s own comment on this).

`bias=False` on every `nn.Linear` here - deliberate, and paired with
RMSNorm/Pre-Norm by convention across nearly every modern LLM: one less
set of parameters to learn, with negligible quality cost at these model
sizes.

### `SwiGLUFeedForward`

```python
hidden = self.activation(self.w1(x)) * self.w3(x)   # Swish(xW1) * (xW3), both (B,S,F)
return self.w2(self.dropout(hidden))                 # (B,S,F) -> (B,S,D)
```

Three matrices (`W1, W3: D×F`, `W2: F×D`) instead of GELU's two
(`W1: D×F`, `W2: F×D`). `W1` produces a signal, `W3` produces a parallel
*gate* of the same shape, and they're multiplied elementwise before the
down-projection - the gate lets the network learn to suppress parts of
the signal per-input rather than always passing everything through a
fixed nonlinearity. Shazeer (2020) found this consistently beats plain
activated FFNs at matched compute; because of the extra matrix, a
*fair* v1-vs-v0 comparison would shrink `d_ff` for v1 to keep total FFN
compute similar, but this repo leaves `d_ff` an explicit config choice
so the actual parameter cost of each architectural decision stays
visible (see each file's own parameter-count docstring for the exact
numbers at a given `D`, `F`).

### Pre-Norm in `EncoderLayerV1`/`DecoderLayerV1`

```python
normed = self.norm1(x)
attention_output = self.self_attention(normed, normed, rope=self.rope, ...)
x = x + self.dropout(attention_output)     # <- x itself was never normalized
```

Compare v0: `x = self.norm1(x + dropout(attention_output))` - the norm
wraps the *sum*. Here the norm only wraps the *input* to the sublayer;
what gets added back to `x` is the raw, unnormalized sublayer output.
The practical effect: `x`, the residual stream, is a running sum that
every layer reads a normalized *copy* of but never itself gets
renormalized. Gradients flowing backward from the output to any earlier
layer pass through that sum via pure addition (`d(x+f(x))/dx` includes
an identity term), with no LayerNorm's nonlinear rescaling in the way -
which is the concrete reason 50+ layer Pre-Norm transformers train
without the careful warmup/init tuning Post-Norm networks of similar
depth need. Because there's no norm after the *last* layer's residual
add either, `EncoderV1`/`DecoderV1` apply one `final_norm` after the
whole stack, so whatever reads their output (cross-attention, the
output head) sees a sanely-scaled vector.

### Weight tying - the real bug, traced

`transformer_v1.py`:

```python
self.llm_head = nn.Linear(self.decoder.d_model, self.decoder.target_vocab_size, bias=False)
self.llm_head.weight = self.decoder.token_embeddings.weight   # same Parameter object
self.apply(self._init_weights)
```

`self.llm_head.weight` and `self.decoder.token_embeddings.weight` now
*are* the same tensor (not a copy - literally the same `nn.Parameter`,
so a gradient update to one updates both, which is the entire point:
one matrix, two jobs, `V*D` fewer parameters than learning both).

The bug this caused: `nn.Embedding`'s PyTorch default init is
`Normal(mean=0, std=1)`. That's reasonable *for a lookup table* (each
row just needs to start somewhere distinguishable), but once that same
matrix is reused as a `Linear` layer's weight, PyTorch's own default for
`Linear` would have initialized it far smaller
(`~1/sqrt(in_features)`, i.e. `~1/sqrt(256) ≈ 0.0625` for a `d_model=256`
model, not `std=1`). With the embedding's `std=1` init feeding the
output projection, `logits = decoder_output @ W.T` where `W` has
`std=1` rows produces logits with `std ≈ sqrt(d_model) ≈ 16` - and the
actual initial training loss for this came out in the 50s, instead of
the `~log(vocab_size) ≈ 3-6` a freshly initialized model should show
(a random model should be roughly as confused as uniform guessing over
the vocabulary, and cross-entropy against a uniform distribution is
`log(V)`). This was found by *actually training it* and noticing the
number didn't match the sane theoretical floor - not by code review.

The fix, `_init_weights`:

```python
if isinstance(module, (nn.Linear, nn.Embedding)):
    nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

`self.apply(self._init_weights)` walks every submodule and re-inits
every `Linear`/`Embedding` weight to the same small, fixed `std=0.02` -
the GPT-2/nanoGPT convention - so the one tied matrix is a sane scale
for *both* of its roles simultaneously, rather than being torn between
two different "correct" defaults that disagree. v0 never hits this
because its output head is a separate `nn.Linear`, default-initialized
for that one role alone, never double-booked.

---

## v2 - efficiency, line by line

**Code:** `src/models/v2/modules_v2.py`, `encoder_v2.py`, `decoder_v2.py`, `transformer_v2.py`

v2 keeps v1's RMSNorm/RoPE/SwiGLU/weight-tying/Pre-Norm exactly - only
attention changes, so only `modules_v2.py` is new model code; the rest
of v2's files mostly just swap `MultiHeadAttentionV1` for
`GroupedQueryAttentionV2`.

### GQA - the actual tensor-shape story

```python
kv_dim = num_kv_heads * self.head_dim
self.w_q = nn.Linear(d_model, d_model, bias=False)       # D -> D   (num_heads heads)
self.w_k = nn.Linear(d_model, kv_dim, bias=False)         # D -> kv_dim  (num_kv_heads heads)
self.w_v = nn.Linear(d_model, kv_dim, bias=False)
```

With `num_heads=8, num_kv_heads=2`: `q` splits into 8 heads of
`head_dim` each (`_split_heads(..., self.num_heads)`), but `k`/`v`
split into only 2 heads (`_split_heads(..., self.num_kv_heads)`) - half
as many parameters spent on `W_k, W_v` right away (`D×kv_dim` instead
of `D×D`). RoPE is applied to `k` *at this smaller head count*
(`cos, sin = rope(k.shape[-2])`) - rotating 2 heads instead of 8 is
less work, and it's mathematically fine because rotation is applied
per-head independently either way.

Before the actual attention matmul:

```python
k = k.repeat_interleave(self.group_size, dim=1)   # (B, num_kv_heads, S, hd) -> (B, num_heads, S, hd)
v = v.repeat_interleave(self.group_size, dim=1)
```

`group_size = num_heads // num_kv_heads = 4`. `repeat_interleave`
duplicates each KV head 4 times *consecutively*, so query heads
`0,1,2,3` (which came from `q`'s first 4 slots) all read the
*duplicated* copy of KV head 0, and query heads `4,5,6,7` all read KV
head 1's duplicated copy. This is why the module computes RoPE before
expanding, not after: expanding first would mean rotating the same
angles 4 times redundantly; expanding after rotation gives identical
results for 1/4 the rotary work. `num_kv_heads == num_heads` collapses
`group_size` to 1 and `repeat_interleave` becomes a no-op - exactly
v1's attention, bit-for-bit. `num_kv_heads == 1` is the other named
extreme (multi-query attention): every query head shares one single K/V
head.

### Local attention mask - and the NaN it caused

```python
def _local_window_mask(self, query_len, key_len, device, query_start=0, key_start=0):
    query_positions = query_start + torch.arange(query_len, device=device)
    key_positions = key_start + torch.arange(key_len, device=device)
    distance = query_positions[:, None] - key_positions[None, :]
    return distance.abs() >= self.window_size
```

Produces an `(S_q, S_k)` boolean mask, `True` where the key is
`window_size` or more positions away from the query (in *either*
direction - `.abs()` is deliberate so one formula covers both the
encoder, where a token can see neighbours on both sides, and the
decoder, where the causal mask applied separately already zeroes out
the future half). This gets `masked_fill`'d into the score matrix
alongside the causal/padding masks - purely additive restriction, same
`(B,H,S,S)` score tensor shape as without it.

The bug: consider a fully-padded row at the end of a batch (shorter
sequences padded up to the batch's max length). Its query position's
window covers *only other padding positions*, which are themselves
masked out by `key_padding_mask`. Every entry in that query's score row
becomes `-inf`, and `softmax` over an all-`-inf` row is mathematically
`0/0` = `NaN`. Full attention (no window) never hits this, because a
padding query can always still attend to *itself* at minimum (nothing
restricts self-to-self visibility without a window) - the window is
specifically what can exclude even that. The fix:

```python
if self.window_size is not None or segment_mask is not None:
    attn_weights = torch.nan_to_num(attn_weights)
```

turning NaN rows into 0 after softmax. Those rows' output is then
`0 · V = 0`, contributing nothing - which is correct, since the row
belongs to a padding position whose output nobody reads anyway (the
loss masks it out via `ignore_index=pad_token_id`), but a raw `NaN`
would have "poisoned" the *gradient* through that batch element even
though the forward value was going to be discarded regardless
(`NaN * 0` is still `NaN`, not `0`, during backprop through other
terms that touch the same batch). This is a genuinely subtle class of
bug: the forward pass can look fine (the padding row's value is thrown
away) while the gradient computation silently breaks, because
autograd doesn't know a value is "unused" just because you don't read
it in `forward()` - it only knows what mathematical operations touched
which tensors.

### Banded attention - computing only what you need

```python
offsets = torch.arange(-reach, reach + 1, device=q.device)              # (2W-1,)
key_index = torch.arange(seq_len, device=q.device)[:, None] + offsets[None, :]   # (S, 2W-1)
...
k_band = k[:, :, key_index]   # (B, kv_heads, S, 2W-1, head_dim) - gathered, not masked
scores = torch.einsum("bkgid,bkitd->bkgit", q, k_band)   # (B, kv_heads, group, S, 2W-1)
```

The masked local-attention path above still computes a full `(S, S)`
score matrix and *then* throws away everything outside the window -
`O(S²)` work regardless of how small the window is. The banded path
instead builds `key_index[i]` = the `2W-1` key positions query `i` can
actually see, and *gathers* exactly those keys via fancy indexing
(`k[:, :, key_index]`) before ever computing a dot product - so the
score tensor is `(S, 2W-1)` instead of `(S, S)`: `O(S·W)` instead of
`O(S²)`. For `S` much larger than `W` this is a real asymptotic win;
the honest caveat (stated directly in the code's docstring) is that
this specific implementation is plain PyTorch gather + einsum, which
*copies* each key `2W-1` times into the gathered tensor - a real fused
kernel (FlashAttention-style) would read each key once and stream the
computation, avoiding that memory blow-up. This version demonstrates
the *algorithmic* idea (sparse attention pattern -> less compute) without
yet delivering the *systems* optimization (fused kernel -> less memory
traffic) that production implementations add on top.

### KV cache - what actually gets cached and why causality changes

```python
past_len = cache.get("seen", 0) if use_cache else 0
...
if use_cache:
    if "k" in cache:
        k = torch.cat([cache["k"], k], dim=-2)
        v = torch.cat([cache["v"], v], dim=-2)
    seen = past_len + query_len
    if self.window_size is not None:
        keep = self.window_size - 1 + query_len
        k, v = k[..., -keep:, :], v[..., -keep:, :]
    cache["k"], cache["v"], cache["seen"] = k, v, seen
```

Without a cache, generating token `t+1` means re-running the decoder on
tokens `0..t` from scratch - the keys/values for tokens `0..t-1` get
*recomputed* even though they're mathematically identical to what they
were on the previous step (nothing about an already-generated token's
key/value depends on what gets generated after it). The cache instead
keeps the running `k, v` tensors across calls and `decode_step` passes
in *only the newest token*, so each step projects one token instead of
re-projecting the whole growing prefix - `cache["k"]` grows by one
position per step instead of being rebuilt. `past_len` + the new
query's own length tells RoPE which absolute positions to rotate by
(`cos[past_len:]`), since the cached keys were already rotated
correctly when they were first added.

With a `window_size`, the cache is trimmed (`k[..., -keep:, :]`) to
only the `window_size - 1` most recent positions plus the new ones -
anything older is provably outside every future query's window, so
keeping it would waste memory for no benefit; this is also what keeps
a windowed cache's memory *bounded* rather than growing with sequence
length, which is the whole efficiency point of combining local
attention with caching.

Causality with a cache works differently from the training-time
`tgt_mask`:

```python
if use_cache:
    query_positions = past_len + torch.arange(query_len, device=scores.device)
    key_positions = key_start + torch.arange(k.shape[-2], device=scores.device)
    scores = scores.masked_fill(key_positions[None, :] > query_positions[:, None], float("-inf"))
```

There's no explicit `(S,S)` triangular mask here, because during
generation the decoder genuinely only ever *has* past and present
tokens in its input - there's no future to accidentally see. But once a
cache mixes "old cached keys" with "one freshly-projected new key" in
one tensor, the code still needs *some* rule for "can query `i` see key
`j`", and that rule is simply `key_positions ≤ query_positions`, derived
from each side's actual absolute position rather than a precomputed
mask shape. Cross-attention's cache (`cache["cross"]`) is simpler: the
encoder's output never changes once computed, so it's projected to K/V
exactly once (`if "k" in cache: use it; else: compute and store`) and
reused unchanged at every decode step.

### Sequence packing and the block-diagonal mask

`PackedSeq2SeqDataset._pack` concatenates several examples'
`source_ids`/`target_ids` into one row and builds a parallel
`segment_ids` tensor (`1,1,1,2,2,0,0...` - example 1's tokens, example
2's tokens, then padding as `0`). `TransformerV2._segment_masks` turns
that into boolean masks:

```python
source_segment_ids[:, :, None] != source_segment_ids[:, None, :]
```

Broadcasting `(B, S, 1) != (B, 1, S)` gives `(B, S, S)` - `True`
wherever two positions belong to *different* segments (including where
either is padding, segment 0, since `0 != 0` is `False`... wait, two
padding positions compare equal, `0 == 0`, so that cell is *not*
blocked by this mask alone - padding exclusion still comes from the
ordinary `key_padding_mask`, applied separately and combined via
`masked_fill` on top). This mask gets `masked_fill`'d into the score
matrix exactly like the causal/window masks, blocking any query in
segment 1 from attending to any key in segment 2 and vice versa - the
model computes attention over the *whole padded row* as always, it's
just mathematically prevented from letting unrelated packed examples
influence each other. RoPE needs no change for packing, because it only
ever encodes *relative* distance within whatever positions it's given -
it doesn't know or care that position 47 and position 52 happen to
belong to different logical examples, and nothing in the attention
computation lets segment-2 tokens leak into segment-1's output, so the
relative-position math stays correct per-segment automatically.

The one place packing *does* need explicit handling outside the model:
`Trainer._train_step` masks out the label at every segment boundary
(`labels.masked_fill(target_segment_ids[:, :-1] != target_segment_ids[:, 1:], pad_token_id)`)
- because without it, the very last real token of example 1 would be
asked to predict example 2's `<BOS>` as "the next token" (since they're
adjacent in the packed row), which is nonsense the model shouldn't be
trained on.

### Gradient checkpointing

```python
if self.gradient_checkpointing and self.training:
    x = checkpoint(layer, x, key_padding_mask, segment_mask, use_reentrant=False)
```

Normally, the forward pass of every layer keeps its intermediate
activations in memory, because backward needs them to compute
gradients. `torch.utils.checkpoint.checkpoint` instead runs the
wrapped function *without* saving those activations, and when backward
reaches that point, re-runs the forward pass *again* just for that one
layer to regenerate what it needs, then immediately discards it. Net
effect: memory drops from "every layer's activations, kept" to "every
layer's *inputs* only, kept" at the cost of roughly one extra forward
pass of total compute. `use_reentrant=False` is the modern, non-legacy
checkpoint implementation (works correctly with RNG state, keyword
arguments). Gated behind `self.training` because there's no backward
pass during evaluation/generation, so recomputing anything would be
pure waste with no memory to save in return.

### Why checkpoints needed to change for `torch.compile`

`torch.compile(self.model)` wraps the model in an `OptimizedModule`
whose `state_dict()` keys are prefixed `_orig_mod.<original key>` - a
compiled model's saved weights literally don't match an uncompiled
model's key names. `CheckpointManager.save`/`load` now explicitly do
`getattr(model, "_orig_mod", model).state_dict()` - reach through the
wrapper if it's there, fall back to the model itself if it isn't - so a
checkpoint saved from a compiled run loads cleanly into either a
compiled or uncompiled model later, instead of raising a key-mismatch
error the first time someone tried to resume or export a compiled run.

---

## v3 - scaling without scaling compute, line by line

**Code:** `src/models/v3/modules_v3.py`, `encoder_v3.py`, `decoder_v3.py`, `transformer_v3.py`

### Why `EncoderV3`/`DecoderV3` *subclass* v2 instead of reimplementing

```python
class EncoderV3(EncoderV2):
    def __init__(self, config):
        super().__init__(config)             # builds v2's embedding, RoPE, final_norm
        self.layers = nn.ModuleList([...])    # then replaces just the layer stack
```

Everything about v3's encoder that *isn't* the layer internals - the
token embedding, the RoPE table, the final norm, the `max_seq_len`
check - is exactly v2's, so `EncoderV3` calls `EncoderV2.__init__` to
get all of that for free and only overwrites `self.layers` afterward.
This is possible specifically *because* v2's constructor builds those
shared pieces as plain attributes a subclass can see and replace,
not because of anything magic - it's the same technique
`TransformerV3(TransformerV2)` uses (see below), and it's a direct,
visible instance of "architecture versions are interchangeable
implementations of one interface," which the FAQ section below makes
more general.

### `MoEFeedForward` - the routing mechanism, traced

```python
probs = torch.softmax(self.router(flat).float(), dim=-1)      # (B*S, num_experts)
top_probs, top_experts = probs.topk(self.top_k, dim=-1)        # (B*S, top_k) each
top_weights = top_probs / top_probs.sum(dim=-1, keepdim=True)  # renormalize to sum to 1
output = flat.new_zeros(flat.shape)
for expert_id, expert in enumerate(self.experts):
    token_index, slot = (top_experts == expert_id).nonzero(as_tuple=True)
    if token_index.numel() == 0:
        continue
    expert_output = expert(flat[token_index])
    weight = top_weights[token_index, slot, None]
    output.index_add_(0, token_index, expert_output * weight)
```

`flat` is every token in the batch flattened to one axis: `(B,S,D) ->
(B*S, D)`, since routing is a per-token decision with no notion of
"batch" or "sequence position" involved. The router is one `D×E`
matrix - tiny compared to an expert's `3*D*F` - producing one score per
expert per token; `softmax` turns scores into a probability
distribution; `topk` picks the `top_k` highest. The loop then asks,
*for each expert in turn*, "which tokens, anywhere in the flattened
batch, chose me?" (`(top_experts == expert_id).nonzero()`), runs that
expert *only* on those tokens (`flat[token_index]`, a much smaller
tensor than the full batch whenever `top_k < num_experts`), and
scatter-adds the weighted result back into the right rows of `output`
via `index_add_`. This is "dropless" - every token that selects an
expert gets routed to it, no capacity limit, no dropped tokens - and
it's a Python loop over experts rather than one fused grouped matmul,
which is simple and correct but not what a production MoE kernel does
(a real system would batch all experts' matmuls together on the GPU
instead of looping in Python, since a loop like this serializes work
that's embarrassingly parallel across experts).

Why `top_weights` is a renormalized *subset* and not the raw `probs`:
if expert A's raw probability was 0.6 and expert B's was 0.3 (with
`top_k=2`, both chosen, some other expert got the remaining 0.1), using
raw `0.6`/`0.3` as the mixing weights would mean the two chosen outputs
don't average up to the token's full representation - renormalizing to
`0.6/0.9=0.667` and `0.3/0.9=0.333` ensures the weights the token
actually uses sum to 1, exactly as if those were the *only* two experts
that existed for this token. And because `output` is built by
multiplying each expert's *output* by `top_weights` (not by using
`top_weights` as an input gate), the router sits downstream of a
differentiable path - gradient flows from the loss, through the
weighted sum, back into `top_weights`, back into `probs`, back into the
router's own `D×E` matrix. That's the entire mechanism by which the
router "learns" anything: it's not special-cased, it's just an ordinary
parameter sitting in an ordinary multiplication that happens to decide
*discretely* who gets a nonzero gradient contribution (`topk` is the
non-differentiable part; the weighting of whichever experts *are*
chosen is fully differentiable).

### The load-balancing loss - why it's shaped the way it is

```python
assignments = torch.zeros_like(probs).scatter(1, top_experts, 1.0)      # (B*S, E), 1 where chosen
fraction = (assignments * valid[:, None]).sum(0) / (num_valid * self.top_k)   # f_e
mean_prob = (probs * valid[:, None]).sum(0) / num_valid                        # P_e
balance_loss = self.num_experts * (fraction.detach() * mean_prob).sum()
```

`fraction[e]` ("`f_e`" in the docstring's notation) is literally "what
fraction of all routing *slots* (tokens × top_k) went to expert `e`" -
not differentiable (`assignments` comes from `topk`'s hard choice, and
it's also explicitly `.detach()`'d when used in the loss, so no
gradient flows through it even if it could). `mean_prob[e]` ("`P_e`")
is the *average router probability* for expert `e` across all tokens,
*whether or not* that token actually picked expert `e` - and this one
*is* differentiable, because `probs` came straight from a `softmax`.
If every expert got exactly `1/E` of the slots and the router's average
probability for every expert was also exactly `1/E`, then
`sum_e f_e * P_e = sum_e (1/E)*(1/E) = E*(1/E²) = 1/E`, and multiplying
by `E` gives exactly `1` - the loss's minimum. If instead the router
collapses onto one favourite expert (`f_favourite ≈ 1`, everyone else
`≈ 0`, and `P_favourite` also large since that's the expert everyone's
being scored toward), the product `f_favourite * P_favourite` is large,
and the loss grows well above `1`. Minimizing this loss therefore
*specifically* pushes `P_e` down for whichever experts are already
overloaded (since that's the only differentiable lever the loss has)
and up for underused ones - nudging the router's own probabilities
toward uniform, without ever needing `fraction` itself to be
differentiable. Padding tokens are excluded from both sums via `valid`
(the token mask), so a batch with lots of padding doesn't skew which
experts look "overloaded."

### `DepthRouter` - straight-through estimation, concretely

```python
soft = torch.sigmoid(self.score(self.norm(x)).float().squeeze(-1))   # (B, S), differentiable
hard = (soft > 0.5).float()                                           # (B, S), the actual 0/1 decision
gate = hard + soft - soft.detach()                                    # forward = hard, backward = d(soft)
```

This is the standard straight-through-estimator trick, worth
understanding mechanically: in the *forward* pass, `gate`'s numeric
value is `hard + soft - soft` = `hard` exactly (since `soft` and
`soft.detach()` have the same *value*, just different gradient
tracking) - so the model genuinely applies a hard, binary decision
("use this layer or don't"), not some soft blend. But when `.backward()`
walks back through `gate`, autograd differentiates the *expression*
`hard + soft - soft.detach()` with respect to its inputs: `hard` is a
comparison (`>`), whose gradient is zero almost everywhere (it's a step
function) so that term contributes nothing; `soft.detach()` is
explicitly marked to block gradient entirely; `soft` is an ordinary
sigmoid output, fully differentiable. So the gradient that reaches
`self.score`'s weights is exactly `d(soft)/d(weights)` - as if the hard
threshold had never been applied at all. This is how you can train a
genuinely discrete, interpretable routing decision (useful for
analysis - "did this token use this layer, yes or no") while still
getting a gradient signal that would otherwise be completely blocked by
the non-differentiable `>` comparison.

```python
return (mean_soft - self.capacity) ** 2
```

A plain squared-error budget loss pulling the *average* soft
probability toward `capacity` (default 0.5, "half of tokens should use
this layer on average") - note it regularizes the differentiable `soft`
average, not the hard decision's average, so it actually has a useful
gradient (the hard average, like `fraction` above, would be locally flat
almost everywhere).

```python
def initialize(self):
    nn.init.zeros_(self.score.weight)
    nn.init.constant_(self.score.bias, 2.0)
```

`sigmoid(2.0) ≈ 0.88` - training starts with the router strongly
favouring "use this layer" for nearly every token (since the weight is
zero, every token gets the *same* initial score regardless of its
content), and the budget loss is what gradually teaches it which tokens
can afford to skip. Starting the other way around (mostly skipping)
would mean most of the network's depth goes unused from step one, with
nothing to tell the router which layers are even worth learning to use.
This call happens *after* `self.apply(self._init_weights)` in
`TransformerV3.__init__` specifically because that generic init would
otherwise zero the bias too (it zeros every `Linear`'s bias
indiscriminately) - `DepthRouter.initialize()` has to run second to
overwrite that default with its own deliberate non-zero starting bias.

### Why dynamic depth doesn't actually save compute yet (and what would)

```python
gate, budget_loss = self.depth_router(layer_input, token_mask)
return layer_input + gate * (layer_output - layer_input), ...
```

Both `layer_input` and `layer_output` were computed for *every* token
in the batch - attention and the FFN ran on the full tensor regardless
of what the gate says. The gate is applied *after the fact*, blending
each token's own before/after state based on its own 0/1 decision. This
genuinely trains a real, interpretable routing signal and demonstrates
the idea end-to-end (and the per-token-threshold design specifically
means training and generation agree, since a threshold decision needs
no knowledge of other tokens' decisions, unlike a sequence-wide top-k
cutoff which would need to see the whole sequence - a real problem for
one-token-at-a-time generation). But it does not yet save any wall-clock
compute, because nothing *skips* running the expensive parts. Making it
actually save compute would require *gathering* the subset of tokens
that pass the gate into a smaller tensor before running attention/FFN
on them - which immediately raises real complications this repo's
comment flags honestly: gathered tokens need their RoPE positions kept
track of correctly (since gathering reorders/shrinks the sequence), and
a KV cache would need to know how many "real" tokens are in each
layer's cache versus how many were skipped-and-passed-through
unchanged. This is explicitly left as the next real step, not
pretended to be solved.

### Why `TransformerV3.__init__` bypasses `TransformerV2.__init__`

```python
class TransformerV3(TransformerV2):
    def __init__(self, config):
        BaseTransformer.__init__(self, config)   # NOT super().__init__(config)
        self.encoder = EncoderV3(config.encoder_config)
        self.decoder = DecoderV3(config.decoder_config)
        ...
```

If `TransformerV3` called `super().__init__(config)` normally, that
would run `TransformerV2.__init__`, which builds `EncoderV2`/`DecoderV2`
- the wrong classes - only for `TransformerV3.__init__` to immediately
overwrite `self.encoder`/`self.decoder` with `EncoderV3`/`DecoderV3`
right after. That would work, but it would also construct (and then
immediately discard) a whole throwaway `EncoderV2`/`DecoderV2`, wasting
time and triggering `EncoderV2`'s own weight initialization for nothing.
Calling `BaseTransformer.__init__` directly - the grandparent, two
levels up - skips straight past `TransformerV2`'s model-building logic
while still running whatever trivial setup `BaseTransformer` itself
does, and `TransformerV3` then builds exactly the v3 pieces it actually
wants, once.

### `aux_loss` plumbing, end to end

`MoEFeedForward.forward` and `DepthRouter.forward` each return a scalar
loss alongside their main output. `V3LayerMixin._feed_forward`/
`_apply_depth` multiply those by `moe_aux_weight`/`depth_aux_weight`
respectively (so a layer's two contributions can be weighted
independently) and the layer's own `forward` sums them:
`return x, aux_loss + depth_loss`. `EncoderV3.forward`/`DecoderV3.forward`
accumulate every layer's `aux_loss` into one running scalar across the
whole stack, and `TransformerV3.forward` adds the encoder's and
decoder's totals into `TransformerOutput.aux_loss`. Finally,
`Trainer._train_step` does:

```python
aux_loss = getattr(output, "aux_loss", None)
if aux_loss is not None:
    loss = loss + aux_loss
```

`getattr(..., None)` rather than `output.aux_loss` directly, because
`TransformerOutput` for v0/v1/v2 has no `aux_loss` field at all (only
v3's dataclass defines it) - this one line is what lets the *exact
same* `Trainer` class train every architecture version without an
`if model_version == ...` branch anywhere in the training loop. With
neither MoE nor dynamic depth turned on, every `aux_loss` contribution
is `.new_zeros(())` all the way up, so `TransformerV3` with its new
features switched off adds a literal `0.0` to the loss - behaviorally
identical to `TransformerV2`, which is exactly the "off switches
compute exactly what v2 does" claim in the code's own docstring.

---

## Tokenizers, line by line

**Code:** `src/tokenizers/v0/tokenizer_v0.py` through `v4/tokenizer_v4.py`

All five share one interface by convention (not an abstract base class
- just consistent naming): `PAD_TOKEN`/`UNK_TOKEN`/`BOS_TOKEN`/`EOS_TOKEN`
class constants, `vocab_size`/`pad_token_id`/`unk_token_id`/
`bos_token_id`/`eos_token_id` properties, and `encode(text, add_bos=,
add_eos=) -> list[int]` / `decode(ids, skip_special_tokens=) -> str`
methods. This shared shape is *why* `TokenizerFactory` can treat every
tokenizer identically (build it, pickle it, call `.encode`/`.decode`)
without knowing which algorithm it wraps.

### `CharTokenizer` (v0)

```python
characters = sorted(set(text))
tokens = self.special_tokens + characters
token_to_id = {token: idx for idx, token in enumerate(tokens)}
```

The entire vocabulary is "every distinct character that appears in
`text`" - nothing more, nothing less. `encode` looks up each character
one at a time, falling back to `unk_token_id` for anything not in that
exact set (which can only happen for characters the training corpus
genuinely never contained, since the vocabulary was built *from* that
same corpus - this is the clearest possible illustration of "a
tokenizer's vocabulary is a direct function of its training corpus,"
elaborated in the FAQ below). `vocab_size` is simply
`len(token_to_id)` - for this repo's OPUS-100 en-fr corpus, 341 (4
special tokens + however many distinct Unicode characters, including
accented French letters, show up in that particular dataset).

### `BPETokenizerV1`, `WordPieceTokenizerV2`, `UnigramTokenizerV3`

All three wrap the `tokenizers` library (the same Rust-backed library
Hugging Face's own tokenizers use) rather than implementing the
algorithms from scratch, and all three share the same training call
shape:

```python
tokenizer = Tokenizer(<Model>(...))
tokenizer.pre_tokenizer = <PreTokenizer>()
tokenizer.decoder = <Decoder>()
trainer = <Trainer>(vocab_size=vocab_size, special_tokens=self.special_tokens, ...)
lines = [line for line in text.split("\n") if line]
tokenizer.train_from_iterator(lines, trainer)
```

The three differ in which `Model`/`PreTokenizer`/`Decoder`/`Trainer`
classes they plug in, and that difference *is* the whole algorithmic
distinction:

- **BPE** (`BPE` model, `ByteLevel` pre-tokenizer/decoder, `BpeTrainer`):
  starts from the 256 possible byte values as a *fixed* base alphabet -
  not characters, bytes - and repeatedly merges whichever adjacent pair
  is *most frequent* in the corpus into a new token, up to
  `vocab_size`. Because the base alphabet is bytes rather than
  characters, there is no character the tokenizer can ever fail to
  represent: anything that doesn't match a learned merge just falls
  back toward its raw UTF-8 bytes, which is why there's genuinely no
  "unknown" failure mode for BPE here (an `UNK_TOKEN` constant exists
  for interface consistency with the others, but byte-level BPE never
  actually needs it). The catch this project hit in practice: because
  the base unit is *bytes*, not characters, a multi-byte UTF-8 character
  (e.g. an accented French letter, which is 2 bytes) can occasionally
  tokenize into *more* tokens than it has characters, if the specific
  byte-pair for that character never got merged into the trained
  vocabulary - meaning **character count is not a safe proxy for
  post-tokenization token count** under this tokenizer. This is exactly
  what crashed a real training run (`ValueError: Sequence Length
  exceeds max_seq_len`, discovered mid-run), fixed by having
  `TokenizedSeq2SeqDataset` filter by *actual tokenized length*, not a
  character-count pre-filter.

- **WordPiece** (`WordPiece` model, `BertPreTokenizer`, `WordPieceTrainer`):
  also grows from single units by merging, but scores a candidate merge
  by how much it increases the *corpus likelihood* (roughly: how often
  the pair occurs together divided by how often its two parts occur
  separately) rather than raw frequency - so it favours merging parts
  that rarely show up apart, not just parts that happen to be common.
  `BertPreTokenizer` splits on whitespace and punctuation *before*
  training/encoding, working on Unicode characters, not bytes - so (1)
  a character genuinely never seen during training has no fallback and
  becomes `<UNK>`, a real failure mode BPE doesn't have, and (2)
  because pre-tokenization discards the original whitespace structure,
  `decode()` can only approximately reconstruct it (rejoining with
  single spaces, continuation pieces marked `##` so the decoder knows
  `"play", "##ing"` glue into `"playing"` with no space between them) -
  it will not byte-for-byte reproduce unusual original spacing.

- **Unigram** (`Unigram` model, `Metaspace` pre-tokenizer/decoder,
  `UnigramTrainer`): the only one of the three that *shrinks* a
  vocabulary instead of growing one - it starts from a large candidate
  set of substrings and repeatedly removes whichever pieces hurt the
  corpus's modeled likelihood least, until reaching `vocab_size`. Each
  surviving piece carries a learned probability, and encoding uses the
  Viterbi algorithm to find the single highest-probability way to split
  a word into pieces - a genuinely different search than BPE's "replay
  merges in training order" or WordPiece's "greedily match the longest
  piece from the left." `Metaspace` rewrites spaces to `▁` and treats
  that as an ordinary vocabulary symbol rather than a separator, so a
  word-initial piece carries its own `▁` marker and `decode()` can
  reconstruct the *exact* original text (unlike WordPiece) just by
  concatenating pieces and turning `▁` back into a space - as long as
  every character in the input was present during training; an unseen
  character still becomes `<UNK>`, same limitation as WordPiece, since
  this is also character-based, not byte-based.

### `PretrainedTokenizerV4`

```python
@staticmethod
def _load(source: str) -> Tokenizer:
    if Path(source).is_file():
        return Tokenizer.from_file(source)
    return Tokenizer.from_pretrained(source)
```

The `text`/corpus argument the factory passes every other tokenizer is
accepted but silently ignored here - nothing is trained, an existing
`tokenizer.json` (local file or Hugging Face Hub id) is loaded as-is,
with whatever algorithm, normalizer and pre-tokenizer its original
authors chose.

```python
def _resolve(self, requested, default):
    token = requested or default
    if self._tokenizer.token_to_id(token) is not None:
        return token
    if requested is not None:
        raise ValueError(f"Token '{requested}' is not in the vocabulary of '{self.source}'.")
    self._tokenizer.add_special_tokens([token])
    return token
```

This is solving a real interoperability problem: this codebase's
pipeline (`TokenizedSeq2SeqDataset`, `Trainer`, `Generator`) all assume
a tokenizer has working `pad_token_id`/`bos_token_id`/`eos_token_id`/
`unk_token_id`, but a pretrained vocabulary often doesn't define all
four by those names (GPT-2's vocabulary has only `<|endoftext|>`, used
for both BOS-like and EOS-like purposes; raw byte-level BPE vocabularies
often have no explicit "unknown" token at all, since byte-level BPE
never needs one). `_resolve` tries the caller's own choice first (e.g.
`eos_token="<|endoftext|>"`, reusing what the pretrained vocabulary
already has), and if the caller didn't specify anything and the default
(`"<PAD>"` etc.) isn't present either, it *adds* a new token to the
vocabulary rather than failing - which is exactly why `vocab_size` for
a pretrained tokenizer can come out slightly larger than the original
file's own vocabulary count. If the caller *did* specify a token
explicitly and it genuinely isn't in the vocabulary, that's treated as
a config typo worth a hard error, not something to paper over.

---

## The "Why" FAQ

Cross-cutting questions that only make sense once you've seen the
mechanics above, answered from the code paths that actually enforce
each answer.

### Why is a tokenizer tied to the dataset it was trained on?

Every trained tokenizer here (`CharTokenizer`, `BPETokenizerV1`,
`WordPieceTokenizerV2`, `UnigramTokenizerV3`) builds its vocabulary by
running a statistical procedure *over the specific corpus string it's
given* - `CharTokenizer` literally takes `sorted(set(text))`; the other
three call `tokenizer.train_from_iterator(lines, trainer)`, where
`lines` comes straight from that same corpus. The resulting vocabulary
- which characters/subwords exist as tokens at all, and for BPE/WordPiece
specifically *in what merge order* - is a direct function of that
corpus's statistics: which pairs/substrings were frequent, which were
rare, which never appeared. Train the exact same algorithm on a
different corpus (say, legal documents instead of casual translation
sentence pairs) and you'd get a *different* vocabulary, because the
frequency statistics driving every merge/prune decision would differ.
Using a tokenizer trained on one domain against text from a very
different domain doesn't break anything outright - every tokenizer
here can still emit *some* token sequence for any input text, through
fallback (BPE's byte-level base alphabet, WordPiece/Unigram's `<UNK>`)
- but it segments badly: words or subwords common in the *new* domain
that weren't common in training have no single-token representation,
so they fragment into many small pieces instead, inflating sequence
length and asking the model to spend attention/parameters reconstructing
word-level meaning from fragments it would have gotten as a single
learned unit under a matched tokenizer.

### Why can't you freely swap tokenizers on an already-trained model?

Two mechanical facts make this not just inadvisable but literally
incoherent: first, `ModelFactory`'s builder functions
(`_build_transformer_v0_config` through `_v3`) do
`encoder_params.setdefault("vocab_size", tokenizer.vocab_size)` -
the embedding table (`nn.Embedding(vocab_size, d_model)`) is sized
*exactly* to whatever tokenizer built the model, at model-construction
time. A different tokenizer almost certainly has a different
`vocab_size`, so the embedding matrix's very shape wouldn't even match
- you couldn't load the old weights into it at all without an explicit,
separate reshaping/remapping step this codebase doesn't provide.
Second, and more fundamentally: even in the contrived case where two
tokenizers happened to share the exact same `vocab_size`, token id `47`
under tokenizer A and token id `47` under tokenizer B almost certainly
mean two *completely different* pieces of text (different training
corpus, different algorithm, different merge order - id assignment is
essentially incidental bookkeeping, not a shared standard). Every row
of a trained embedding table is a learned vector *for a specific
token's meaning as that tokenizer defines it*; swapping which tokenizer
maps text to ids doesn't touch those learned vectors at all, so row 47
would suddenly be fed whatever *new* concept id 47 now represents - the
model's entire learned mapping from "token id" to "vector" is
systematically scrambled, not approximately right. This is exactly why
`src/utils/hashing.py`'s `get_config_hash` includes the full `config`
(which always includes `TokenizerConfig`) in what it hashes: a
tokenizer change always produces a different experiment hash, hence a
different `artifacts/experiments/<hash>/` directory, hence a brand new
model trained from scratch - there is no code path that lets you keep
old weights while changing the tokenizer underneath them.

### Why does changing *any* config field change the experiment hash?

`get_config_hash` calls `config_to_json(config)`, which recursively
`asdict()`s the *entire* `ExperimentConfig` - model, tokenizer, data,
training, evaluation, runtime, everything - sorts every dict's keys for
determinism, and SHA256-hashes the resulting JSON string. There is no
field excluded from this; changing `training.batch_size`,
`runtime.seed`, or adding a brand-new field like `save_total_limit`
(which happened in this repo's own history - see `CLAUDE.md`'s note
about having to rename an experiment directory after that change) all
produce a different hash, hence a different `artifacts/experiments/`
directory. This is deliberate, not an inconvenience to work around: the
hash is meant to *be* the claim "this exact configuration produced this
exact set of weights." If two different configs could somehow map to
the same experiment directory, you could end up loading a checkpoint
trained under settings that no longer match what you think you're
running - silently wrong reproducibility, the worst kind, because
nothing would ever error to tell you. A hash that changes on *any*
difference, no matter how seemingly minor, is what makes "find the
experiment with this exact config" and "this config has never been run
before, train it fresh" both unambiguous operations.

### Why can model architecture be swapped more freely than the tokenizer?

Every `TransformerVN.forward()` - v0 through v3 - has the identical
call signature: `(source_ids, target_ids, source_key_padding_mask=,
target_key_padding_mask=, ...) -> TransformerOutput(logits=...)` (v3
adds an optional `aux_loss` field to the *output*, defaulted so nothing
else needs to check for it, and optional `segment_ids` *inputs* v0/v1
simply never receive since nothing in those architectures' `forward()`
asks for them). `Trainer`, `Evaluator`, `Generator`, `Predictor` - every
piece of code that *uses* a model - only ever calls through this one
shape. That's what "architecture versions are interchangeable
implementations of the same input/output contract" concretely means:
given token ids in, you get logits out, and the training/inference code
around the model is written against that contract, never against any
particular version's internals. A tokenizer is not an implementation of
some downstream contract in the same sense - it's what *defines* what a
token id means in the first place, upstream of everything else. The
model's `forward()` never asks "what algorithm produced these ids" -
it just consumes integers and returns logits over however many ids
exist in that integer space. Change *which* integers mean *what* (a
different tokenizer), and every downstream consumer of those integers -
most importantly, the model's own learned embedding table - is now
being handed a different question than it learned to answer, even
though `forward()`'s literal code path still "works" in the sense of
not crashing. Changing *how* the model answers (a different
architecture version) while the integers' meaning stays fixed is a
much smaller, well-contained kind of change - which is exactly why
`EncoderV3(EncoderV2)` and `TransformerV3(TransformerV2)` can be built
as *subclasses* that inherit and selectively override, while no
tokenizer here is built as a subclass of another: there's no shared
base behavior to inherit when the entire point of a tokenizer is to
define its own vocabulary from scratch.

### Why is `vocab_size` the thing that actually ties tokenizer and model together?

Concretely, in `src/models/factory.py`:

```python
encoder_params.setdefault("vocab_size", tokenizer.vocab_size)
decoder_params.setdefault("target_vocab_size", tokenizer.vocab_size)
```

`TokenizerConfig` and `ModelConfig` are two entirely separate dataclass
fields on `ExperimentConfig` - nothing in the config schema itself
forces them to agree. The connection is made at *build time*, in code:
`ModelFactory.create(config, tokenizer=tokenizer)` is always called
*after* the tokenizer has already been built (see `scripts/train.py`'s
`build_tokenizer` running before `ModelFactory.create`), and the model's
builder function reads `tokenizer.vocab_size` directly off the already-
built tokenizer object and injects it into the model's config dict
(`setdefault` only sets it if the config didn't already specify
`vocab_size` explicitly, which no example config in this repo does -
every config leaves vocab_size for the factory to fill in). This is why
an `EncoderConfigV0`/`V1`/`V2`/`V3` dataclass *has* a `vocab_size`
field at all despite no config file ever setting it directly - it's a
required constructor argument for the dataclass, but a derived value in
practice, sourced from whichever tokenizer the same experiment's config
names. It's "load-bearing" in the sense that if this wiring were ever
removed or bypassed, you could build a model whose embedding table is
simply the wrong size for the tokenizer it will actually receive token
ids from - either silently truncating valid ids as Python indexing
errors, or (if the vocab_size happened to be larger) wasting embedding
rows nothing would ever address.

### Why can't sequence packing and streaming data loading be combined?

`build_datasets` in `src/data/seq2seq.py` raises outright if both
`streaming` and `pack_sequences` are requested together. The reason is
in each one's own implementation: `PackedSeq2SeqDataset.__init__` walks
every example's *already-tokenized* length up front, deciding how many
examples fit into each packed row by running a greedy bin-packing loop
over the *entire* dataset before training starts - it needs to see
every example's length before it can decide a single row's contents,
because deciding where one row ends depends on how much "budget" is
left after every example packed into it so far. `StreamingSeq2SeqDataset`
exists specifically to *avoid* that: its entire value proposition is
tokenizing lazily, one example at a time, as the training loop asks for
the next one, specifically so you never need the whole corpus
tokenized and held in memory before training can begin. Packing's
"see every length up front" requirement and streaming's "never see
more than one example at a time" design goal are direct opposites -
there's no way to pack rows from a stream without first materializing
enough of the stream to know what to pack, at which point you're no
longer meaningfully streaming. `BucketBatchSampler` (length bucketing)
has this exact same conflict with streaming for the same reason (it
also needs every example's length up front to sort by), which is why
`Trainer.__init__` separately raises if `bucket_by_length` is requested
on an `IterableDataset`.

### Why did mixed precision need the NaN-masking fix specifically in v2?

`v0`/`v1` have no local attention and no sequence packing, so there was
never a code path where a padding query's *entire* set of visible keys
could be masked out - full attention always leaves at least "attend to
myself" available, since nothing restricts self-to-self visibility
without a window or a segment boundary. `GroupedQueryAttentionV2`
introduced both `window_size` (which *can* exclude even a token's own
position if its whole window lands on padding) and `segment_mask`
(packing, which can do the same across a segment boundary) - so v2 is
specifically where "every key for this query is masked" first became
possible, and therefore where `softmax` producing `NaN` for that row
first became a real failure mode worth guarding against
(`torch.nan_to_num(attn_weights)`, applied only when `window_size` or
`segment_mask` is in play - not unconditionally, since it would be
pure wasted work to guard against a failure mode that can't occur
under full, unwindowed, unpacked attention). This has nothing
specifically to do with *precision* (fp16/bf16 vs fp32) despite the
docstring mentioning mixed precision in the same breath - it's an
attention-masking correctness fix that happens to matter *more* under
low precision, where `NaN` propagation and the narrower dynamic range
of fp16 make silent corruption easier to trigger and harder to notice
than under fp32's wider range, but the underlying bug (an all-`-inf`
softmax row) is exactly as real at fp32 as at fp16/bf16.

---

## How the pieces compose: two concrete configs traced end to end

### `configs/toy_copy_v1.py`

```python
model.transformer = ComponentConfig(name="transformer", version="v1")
model.encoder = ComponentConfig(name="encoder", version="v1", parameters={
    "max_seq_len": 64, "d_model": 64, "num_heads": 4, "d_ff": 256, "num_layers": 2, ...
})
tokenizer = TokenizerConfig(name="char", version="v0")
data = DataConfig(name="toy_copy", version="v0", parameters={"vocab": "abcdefghijklmnopqrstuvwxyz", ...})
```

Tracing `scripts/train.py`'s flow for this config:

1. `DataFactory.build(config)` looks up `"toy_copy_v0"` in its registry
   and generates synthetic (source, target) text pairs from the 26-letter
   `vocab` string (this is a "copy" task - the target is the source,
   possibly transformed; the exact transform lives in the toy_copy data
   builder, not shown above).
2. `TokenizerFactory.build(config, corpus)` looks up
   `f"{config.tokenizer.name}_{config.tokenizer.version}"` = `"char_v0"`,
   finds `CharTokenizer`, and calls `CharTokenizer(corpus)` - building a
   vocabulary of whatever distinct characters appear in the generated
   copy-task text (here, exactly the 26 lowercase letters plus 4 special
   tokens = 30).
3. `ModelFactory.create(config, tokenizer=tokenizer)` looks up
   `f"{config.model.transformer.name}_{config.model.transformer.version}"`
   = `"transformer_v1"`, finds `(TransformerV1, _build_transformer_v1_config)`,
   and `_build_transformer_v1_config` builds an `EncoderConfigV1(vocab_size=30,
   max_seq_len=64, d_model=64, num_heads=4, d_ff=256, num_layers=2, ...)`
   (the `30` is injected from the tokenizer just built, via `setdefault`)
   and the matching `DecoderConfigV1`, then constructs `TransformerV1`.
4. Inside `TransformerV1.__init__`: `EncoderV1(encoder_config)` builds a
   `token_embedding` of shape `(30, 64)`, a `RotaryEmbedding(head_dim=16,
   max_seq_len=64)` (`head_dim = d_model/num_heads = 64/4 = 16`), and 2
   `EncoderLayerV1`s each holding one `MultiHeadAttentionV1(d_model=64,
   num_heads=4)` (four `64×64` matrices) and one
   `SwiGLUFeedForward(d_model=64, d_ff=256)` (two `64×256` matrices, one
   `256×64`). `DecoderV1` mirrors this with its own embedding/layers plus
   cross-attention per layer. `self.llm_head.weight =
   self.decoder.token_embeddings.weight` ties the `(30, 64)` embedding
   to double as the `64 -> 30` output projection. `self.apply(self._init_weights)`
   then re-initializes every one of those matrices to `std=0.02`.
5. At training time, a batch of source text (shape `(B, S_src)` after
   padding to the batch's longest example) and target text (`(B, S_tgt)`)
   flows through `TransformerV1.forward`: the encoder turns `(B, S_src)`
   token ids into `(B, S_src, 64)` memory; the decoder (fed
   `target_ids[:, :-1]` for teacher forcing) turns `(B, S_tgt-1)` ids into
   `(B, S_tgt-1, 64)` hidden states, attending to that `64`-dim memory via
   cross-attention; `llm_head` projects to `(B, S_tgt-1, 30)` logits,
   compared against `target_ids[:, 1:]` via cross-entropy.

### `configs/toy_copy_wordpiece_v2.py`

Per its own comment: "Same toy_copy task and model as toy_copy_v1.py,
with the tokenizer swapped from char_v0 to wordpiece_v2." Concretely,
only `model.transformer.version` stays `"v1"` or becomes `"v2"`
(depending on the exact file - check it directly) and `tokenizer =
TokenizerConfig(name="wordpiece", version="v2")` changes step 2 above:
`TokenizerFactory.build` now resolves `"wordpiece_v2"` instead of
`"char_v0"`, builds a `WordPieceTokenizerV2` by running the WordPiece
training algorithm over the same corpus, and ends up with some
different `vocab_size` (WordPiece's target `vocab_size` parameter,
capped by however many distinct pieces the small toy corpus actually
supports) - everything downstream in `ModelFactory.create` proceeds
identically, just with that different `vocab_size` flowing into the
embedding tables' shape instead. This is the concrete proof, in a real
config pair from this repo, that tokenizer and model vary along
genuinely independent axes: one file changes exactly one
`ComponentConfig`/`TokenizerConfig` field and nothing else, and the
entire pipeline (`DataFactory` -> `TokenizerFactory` -> `ModelFactory` ->
`Trainer`) adapts automatically, because every stage reads its inputs
from the generic `ExperimentConfig` rather than hardcoding which
version it expects.
