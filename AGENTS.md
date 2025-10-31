# Repository Guidelines

## Project Structure & Module Organization
- `fabfile.py` is the Fabric entry point; collections in `fab_tasks/` forward to modules in `scripts/`.
- `configs/` holds the canonical YAML for datasets, training, evaluation, and packaging; update them before touching code.
- Data artifacts live in `data/raw/` and `data/processed/`, checkpoints in `models/`, and reports in `reports/`.

## Build, Test, and Development Commands
- Run `./scripts/setup_env.sh` to create `.venv`, install dependencies, and scaffold `.env` from the template.
- Dataset flow: manually `scp` the DuckDB file into `data/raw/` (table `examples` with `id`, `task`, `prompt_text`, `response_strong`, `metadata`, optional `slot_specs`) and then run `fab dataset.clean|split|stats` for normalization, stratification, and coverage summaries.
- Training flow: `fab train.prepare`, `fab train.run --config configs/training.yaml`, and `fab train.resume` scaffold, launch, and restart fine-tuning.
- Evaluation and ops: `fab eval.generate|judge|report`, `fab package.export`, and `fab ops.project_tokens` cover judging, packaging, and usage tracking.
- 128k context: `configs/training.yaml` ships with `max_seq_length: 131072` and rope scaling (`factor: 4.0`). Adjust downward only if a new task cannot fit within 128k tokens.

## Coding Style & Naming Conventions
- Target Python 3.10+, four-space indentation, and PEP 8 throughout `scripts/` and `fab_tasks/`. Favor type hints and lightweight dataclasses for schema enforcement.
- Name Fabric tasks with dot-accessible verbs (`dataset.clean`), modules and functions in `snake_case`, and exported classes in `PascalCase`.
- Keep orchestration inside Fabric tasks thin; push transformative or stateful logic into reusable helpers under `scripts/io_utils.py` and sibling modules.

## Testing & Validation Guidelines
- Lean on runtime validation: `fab dataset.clean` enforces schemas, while `fab dataset.stats` and `fab eval.judge` serve as regression gates. Run them before proposing training or evaluation updates.
- Add unit tests under `tests/` with `pytest` (`test_<module>.py`) as logic stabilizes, supplying fixtures for exemplar and output samples.
- Capture evaluation evidence in `reports/` (including judge transcripts) and link relevant files when requesting reviews.

## Commit & Pull Request Guidelines
- Commit subjects stay imperative and under 72 chars; describe dataset or config changes in the body.
- PRs must note touched Fabric tasks, config updates, and environment requirements; link the latest `reports/*.md` plus rerun instructions.
- Reference issue IDs where applicable and flag follow-up work (TODOs, future automation) in the PR body instead of burying them in code comments.

## Dataset Schema Notes
- `task` drives default prompts (`navpage_classifier` → YES/NO classification, `productpage_extractor` → JSON extraction). Introduce new task slugs as you add workflows.
- `prompt_text` should mirror the real prompts (`NavPage.analyze` classification question or `ProductPage.analyze` extraction instructions) so fine-tuning sees the same mutilated HTML snippets the production LLM receives.
- `response_strong` may be JSON or plain text; the cleaner serializes both into training/eval targets while preserving raw structures for judging.
- Default rope scaling keeps Gemma responsive to long prompts; update `model.rope_scaling` and `training.max_seq_length` together when introducing new context lengths.

## Security & Configuration Tips
- Copy `.env.example` to `.env` and populate dataset paths (`DATASET_DB_PATH`), LiteLLM credentials, and registry tokens. The file is gitignored and automatically loaded by `scripts/config.py` via python-dotenv.
- Rotate secrets regularly and prefer short-lived credentials; reference them in runbooks or PRs without pasting raw tokens.
