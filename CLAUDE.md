# Working on this repo

This project implements a transformer from scratch across versions
v0→v1→v2→v3 (see [VERSIONS.md](VERSIONS.md) for what each version is
and what's still unimplemented - check it before starting work to see
which version/feature is next). v0 and v1 are done, tested on real
training runs, and pushed. v2, v3, and additional tokenizers (listed in
VERSIONS.md / discussed in prior sessions: WordPiece, Unigram/
SentencePiece, pretrained tokenizers) are not yet implemented.

## Workflow for implementing new versions/features

1. **Write a lite smoke test alongside the code, not after.** A smoke
   test here means: tiny synthetic data, a handful of steps/epochs,
   CPU is fine - the goal is "does this run end-to-end and produce
   sane shapes/losses," not real training quality. Use
   `configs/toy_copy_v0.py` as the pattern for a fast synthetic config.
   Put these under `tests/`, run via `pytest` (already configured in
   `pyproject.toml`). The point of writing it alongside the code is to
   catch integration bugs (shape mismatches, wrong tensor dtypes,
   config wiring) immediately, instead of discovering them hours into
   a real Colab training run.
2. **Finish one version/feature, smoke-test it, then commit and push
   before starting the next one.** Do not batch multiple
   versions/features into one push at the end of a session. Each push
   should leave `master` in a working, tested state. If a session gets
   interrupted (e.g. hits a usage quota) mid-version, whatever was
   already finished and tested is already safe on `origin/master`.
3. Update [VERSIONS.md](VERSIONS.md) in the same commit when a version
   moves from "Future" to implemented, so the next session (or a
   scheduled one resuming after a quota reset) can tell what's left
   just by reading it.

## Conventions already established

- Config-driven: new model/tokenizer/data variants go through the
  `ModelFactory`/`TokenizerFactory`/`DataFactory` registries
  (`src/*/factory.py`), keyed `"{name}_{version}"` - see v1's
  registration for the pattern.
- Experiment identity is a SHA256 hash of the *entire* config
  (`src/utils/hashing.py`). Adding a new config field changes every
  experiment's hash - expected, not a bug, but means any locally
  stored experiment directories need renaming to match after such a
  change (this happened when `save_total_limit` was added).
- `artifacts/experiments/`, `data/`, `mlruns/`, `mlflow.db`,
  `mlartifacts/`, and `export_bundle*` are gitignored - never try to
  commit trained models, datasets, or MLflow logs.
