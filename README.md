# ml-learning

A from-scratch, versioned implementation of a transformer encoder-decoder model,
built to learn transformer internals by implementing and evolving each
architectural piece (encoder, decoder, tokenizer) as its own version, while a
generic, config-driven pipeline handles training, validation, inference, and
experiment tracking around it.

## Project layout

```
src/
  models/
    template.py        # Base classes (BaseEncoder/BaseDecoder/BaseTransformer)
    factory.py          # ModelFactory: config -> model instance (safe to edit)
    v0/                 # Versioned model implementation (do not touch structure)
    v1/, v2/, ...        # Future versions go here
  tokenizers/
    factory.py          # TokenizerFactory: config -> tokenizer instance
    v0/                 # Versioned tokenizer implementation
  data/
    factory.py          # DataFactory: config -> (train_pairs, val_pairs, corpus)
    seq2seq.py           # Tokenizing dataset wrapper + padding collate_fn
    v0/toy_copy.py       # Placeholder synthetic dataset (pipeline validation only)
  training/
    trainer.py           # Trainer: full training loop
    checkpoint.py         # CheckpointManager: save/load optimizer+model state
  evaluation/
    evaluator.py          # Evaluator: loss / perplexity / token accuracy
  inference/
    base.py               # LoadedExperiment: shared "load tokenizer+model" logic
    predictor.py           # Predictor: run the model, compute eval metrics
  generator/
    generator.py           # Generator: greedy decoding, text-in/text-out
  tracking/
    tracking.py             # setup_mlflow()
    mlflowlogger.py          # MLflowLogger wrapper
    system.py                # Environment info (python/torch/cuda/device)
  datamodels.py              # ExperimentConfig and all its nested configs
  utils/hashing.py            # Deterministic config -> experiment hash

artifacts/
  manager.py                  # ArtifactManager / ExperimentManager (paths, persistence)
  experiments/<hash>/          # Generated per-experiment output (gitignored)

configs/                        # Python files, each defining a CONFIG = ExperimentConfig(...)
scripts/
  train.py                      # Config in -> trained + logged experiment
  generate.py                    # Config + text in -> generated text out
```

## How the config system works

Everything about one experiment — which model version, which tokenizer
version, which data source, which hyperparameters — lives in a single
`ExperimentConfig` (see `src/datamodels.py`), authored as a plain Python file
under `configs/` that sets a module-level `CONFIG` variable (see
`configs/toy_copy_v0.py`).

That config is hashed (`src/utils/hashing.py`) into a short experiment ID.
Everything for that experiment — trained weights, tokenizer, checkpoints,
metrics, the config itself — is written under:

```
artifacts/experiments/<config_hash>/
  config.json
  metadata.json          # status + final metrics once training finishes
  model/model.pt
  tokenizer/tokenizer.pkl
  checkpoints/
  evaluation/results.json
```

Running the same config twice does **not** retrain — `scripts/train.py`
checks `metadata.json` for `status: completed` and skips unless you pass
`--force`. Change any field in the config and you get a different hash, i.e.
a fresh, independent experiment directory.

## Setup

```
uv sync
```

Training and inference log to a local MLflow tracking server. Start it once
per session, in its own terminal, and leave it running:

```
uv run mlflow server --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` in a browser to watch runs, params, and metrics
as they land.

## Running the pipeline

### Train an experiment

```
uv run python -m scripts.train --config configs/toy_copy_v0.py
```

This will, in order:

1. Load the config from the given file.
2. Compute the experiment hash and check `artifacts/experiments/<hash>/metadata.json` for a prior completed run (skips if found, unless `--force` is passed).
3. Build (or load, if already saved) the tokenizer via `TokenizerFactory`, using the corpus returned by `DataFactory`.
4. Build the train/validation datasets, tokenized and padded via `TokenizedSeq2SeqDataset` + a pad-aware `collate_fn`.
5. Build the model via `ModelFactory` (encoder/decoder/transformer versions and hyperparameters come from the config; vocab sizes are injected from the tokenizer automatically).
6. Run `Trainer.train()`: teacher-forced training with gradient accumulation, gradient clipping, optional LR scheduling, periodic checkpointing (`training.save_every_n_steps`), per-epoch validation, and per-step/per-epoch MLflow logging.
7. Save the trained model, and mark the experiment `completed` in `metadata.json` with final metrics.

Force a retrain of an already-completed experiment:

```
uv run python -m scripts.train --config configs/toy_copy_v0.py --force
```

### Generate text from a trained experiment

```
uv run python -m scripts.generate --config configs/toy_copy_v0.py --text "hello"
```

Optional flags: `--max-new-tokens` (default 128), `--temperature` (default 1.0).

This loads the experiment's saved tokenizer and weights (via
`LoadedExperiment`), tokenizes your input text, greedily decodes with the
trained model, and prints the decoded output text. The config you pass must
match (i.e. hash to) the experiment you already trained — this is exactly
how you pick which saved model to load.

### Evaluate a trained model on a dataset (no dedicated script yet — use directly)

```python
from src.inference.predictor import Predictor
from src.data.factory import DataFactory
from src.data.seq2seq import TokenizedSeq2SeqDataset
from configs.toy_copy_v0 import CONFIG

predictor = Predictor(CONFIG)
_, val_pairs, _ = DataFactory.build(CONFIG)

metrics = predictor.evaluate(
    TokenizedSeq2SeqDataset(val_pairs, predictor.tokenizer)
)
# {"loss": ..., "perplexity": ..., "token_accuracy": ...}
# also written to artifacts/experiments/<hash>/evaluation/results.json
```

## Running your own experiment

1. Copy `configs/toy_copy_v0.py` to a new file, e.g. `configs/my_experiment.py`.
2. Edit any of:
   - `model.transformer` / `model.encoder` / `model.decoder` — name, version, hyperparameters (`d_model`, `num_heads`, `d_ff`, `num_layers`, `dropout_param`, `max_seq_len`, ...). Do **not** set `vocab_size`/`target_vocab_size` — those are injected automatically from the tokenizer at build time.
   - `tokenizer` — name/version/parameters.
   - `data` — name/version/parameters.
   - `training` / `evaluation` / `runtime` — epochs, batch size, optimizer, learning rate, scheduler, seed, device, etc.
3. `uv run python -m scripts.train --config configs/my_experiment.py`

Because the config's hash *is* the experiment identity, this automatically
gets its own artifact directory and never collides with or overwrites
another experiment's results.

## Adding a new version (this is the part you'll do most often)

The factories are the only place that needs to know a new version exists —
nothing in `Trainer`, `Evaluator`, `Predictor`, or `Generator` changes.

**New model architecture version:**
1. Add `src/models/vN/` with your encoder/decoder/transformer implementation, following the `BaseEncoder`/`BaseDecoder`/`BaseTransformer` contract in `src/models/template.py`.
2. In `src/models/factory.py`, write a small builder function that turns the generic `config.model.encoder.parameters` / `config.model.decoder.parameters` dicts into your version's own config dataclass (mirror `_build_transformer_v0_config`).
3. Register it: `ModelFactory.register("transformer_vN", TransformerVN, build_transformer_vN_config)`, or add the entry directly to `ModelFactory._registry`.

**New tokenizer version:**
1. Add `src/tokenizers/vN/` with your tokenizer class (needs `.vocab_size`, `.pad_token_id`, `.bos_token_id`, `.eos_token_id`, `.encode()`, `.decode()` to work with the rest of the pipeline).
2. Register it in `src/tokenizers/factory.py`'s `TokenizerFactory._registry` (or via `TokenizerFactory.register(...)`). Save/load is automatic (pickled generically) — no extra code needed.

**New data source:**
1. Write a builder function `(parameters: dict) -> (train_pairs, val_pairs, corpus)` where `pairs` are lists of `(source_text, target_text)` tuples and `corpus` is the text used to build the tokenizer's vocabulary.
2. Register it in `src/data/factory.py`'s `DataFactory._registry`.

Note: `src/data/v0/toy_copy.py` is a synthetic placeholder (source ->
reversed source) used only to validate the pipeline end-to-end. The real,
configurable multi-step data-preparation pipeline is a planned future
iteration, not part of this one.

## Known limitations / follow-ups

- Mixed precision, gradient checkpointing, and `torch.compile` are declared in `RuntimeConfig`/`TrainingConfig` but not yet wired into `Trainer` — currently inert fields reserved for later.
- `Generator.generate()` doesn't yet handle batched inputs with padding masks on the decoder side (fine for single-sequence `generate_text`, would need extending for batched generation).
- No dedicated `scripts/evaluate.py` yet — use `Predictor` directly (see above) until one is added.
