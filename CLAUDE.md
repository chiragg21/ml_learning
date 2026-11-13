# Working on this repo

This project implements a transformer from scratch across versions
v0→v1→v2→v3 (see [VERSIONS.md](VERSIONS.md) for what each version is
and what's still unimplemented - check it before starting work to see
which version/feature is next - VERSIONS.md is the source of truth on
progress, not this file, since it's updated every time a version
lands).

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
2. **Finish one entire version/tokenizer - every piece it's defined to
   have, per VERSIONS.md - before committing and pushing.** The unit
   of "done" is the whole version, not one sub-feature inside it: e.g.
   v2 is GQA + local attention + KV cache + a banded kernel + sequence
   packing + mixed precision + gradient checkpointing + torch.compile
   + an efficient data pipeline, all together, not one push per
   sub-feature. Smoke-test each piece as it's written, but hold the
   commit until the entire version's checklist in VERSIONS.md is
   implemented and every test passes - only then commit and push,
   once, for that whole version. Do not batch *multiple different*
   versions into one push either - one version per push, but that
   version complete, not partial. If a run can't fit the whole version
   in its budget, it should push nothing rather than push a fragment;
   a later run starts over on that version from a clean `master`.
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
