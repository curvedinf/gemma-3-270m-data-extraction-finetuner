# Gemma 3 270M Data Extraction Fine-Tuning Pipeline

This repository houses Fabric automation, configuration, and supporting scripts for building a fine-tuned Gemma 3 270M model that replaces the `navpage.analyze` and `productpage.analyze` strong-model calls in `../blazed_deals`. The pipeline covers dataset creation, training, evaluation with LLM-as-judge, and packaging artifacts suitable for ROCm vLLM runtime.

## Prerequisites
- Linux environment with ROCm-capable GPUs (for training/inference stages).
- Python 3.10+ recommended.
- Access to strong-model logs that provide *correct* JSON outputs to supervise Gemma.
- LiteLLM configured with the desired judge backend (environment variables or `litellm` config file). By default we point at `openrouter/qwen/qwen3-coder-30b-a3b-instruct`; export `LITELLM_API_BASE=https://openrouter.ai/api/v1` and `LITELLM_API_KEY=<your key>` before running judge tasks.
- `git`, `make` (optional), and Fabric 3.x (installed via project requirements).
- Hardware capable of 128k-context training (e.g., 80GB+ ROCm GPUs) — the pipeline applies 4× rope scaling to extend Gemma 3 270M from 32k to 128k tokens.
- Hugging Face tooling (`transformers`, `optimum[amd]`) installs via `requirements.txt`; AMD's *Running models from Hugging Face* guide covers optional Flash Attention 2, GPTQ, and ONNX workflows.

## Quickstart
1. **Clone & enter the repository**
   ```bash
   git clone <repo-url>
   cd gemma-3-270M-data-extraction-finetuner
   ```

2. **Bootstrap the environment**
   ```bash
   ./scripts/setup_env.sh
   ```
   (Override the Python executable via `PYTHON=/opt/conda/bin/python ./scripts/setup_env.sh` if needed. The script installs the ROCm nightly `torch/vision/audio` wheels from AMD's index.)

3. **Configure environment secrets**
   ```bash
   # edit .env and populate dataset paths, LiteLLM keys, and registry tokens
   ```
   Sensitive values (e.g., `LITELLM_API_KEY`, `HUGGINGFACE_TOKEN`) stay outside Git and are loaded automatically via `scripts/config.py`. Adjust the ROCm environment variables (`ROCM_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, etc.) in `.env` to match your GPU topology before launching training or inference.

4. **Explore Fabric task groups**
   ```bash
   fab -l
   fab dataset.clean --schema-version v1  # requires data/raw/dataset.duckdb
   fab train.prepare
   ```
   Each collection (`env`, `dataset`, `train`, `eval`, `package`, `ops`) aligns with the sections in `PLAN.md`.

## Training Workflow
1. **Copy the dataset DuckDB**
   ```bash
   scp user@server:/path/to/dataset.duckdb data/raw/dataset.duckdb
   ```
   Keep the file name consistent with `DATASET_DB_PATH` in `.env` (default: `data/raw/dataset.duckdb`) and ensure it exposes an `examples` table with `id`, `task`, `prompt_text`, `response_strong`, `metadata`, and `slot_specs` columns (use slugs such as `navpage_classifier` / `productpage_extractor`).

2. **Clean and normalize data**
   ```bash
   fab dataset.clean --schema-version v1
   ```
   Produces validated task-specific files under `data/processed/`, preserving the full prompts (instructions + mutated HTML snippets) and attaching the schema version.

3. **Create stratified splits + sanity stats**
   ```bash
   fab dataset.split --config configs/datasets.yaml
   fab dataset.stats --config configs/datasets.yaml  # optional but recommended
   ```
   Verify the generated manifest in `data/processed/` before training.

4. **Stage the training run**
   ```bash
   fab train.prepare --config configs/training.yaml
   ```
   Ensures datasets, prompt templates, and output directories exist; snapshots the config next to the checkpoints. Confirm `configs/training.yaml` retains `max_seq_length: 131072` and the rope-scaling factor aligned with your hardware.

5. **Launch fine-tuning**
   ```bash
   fab train.run --config configs/training.yaml
   ```
   Use `--resume path/to/checkpoint` to continue from an interruption.

6. **Evaluate with LiteLLM judge**
   ```bash
   fab eval.generate --split validation --config configs/eval.yaml
   fab eval.judge --split validation --config configs/judge_slots.yaml
   fab eval.report --run-id validation_YYYYmmddTHHMMSSZ --output reports/latest_eval.md
   ```
   `fab eval.generate` loads the checkpoint with ROCm Torch/PEFT, applies grammar-based decoding to guarantee JSON-valid completions, and writes `reports/model_outputs/{split}_candidates.jsonl`; confirm pass rates in `reports/judge/` and share the Markdown summary in PRs.

7. **Package deployable artifacts**
   ```bash
   fab package.export --config configs/package.yaml --output models/export
   ```
   Produces ROCm-friendly bundles for integration with the `blazed_deals` runtime.

## Dataset DuckDB Columns
- `id` – unique example identifier used for deduplication, manifest accounting, and evaluation joins.
- `task` – scenario label (e.g., `navpage_classifier`, `productpage_extractor`). Determines default system prompts and groups outputs for splitting.
- `prompt_text` – the entire user-facing payload sent to the model. For nav pages this should mirror the `NavPage.analyze` question (classification instructions plus `last_scrape_keyword_sections` snippets); for product pages it should mirror `ProductPage.analyze`, including the key/description catalog and the mutilated snippet body.
- `response_strong` – the ground-truth answer. Nav pages use a `YES`/`NO` string, while product pages retain the structured JSON returned by the strong model.
- `metadata` – optional JSON with extra context such as domain hints, schema versions, or overrides like `system_prompt`.
- `system_prompt` (optional) – allows per-example control over the assistant instructions; defaults derive from the `task` slug if omitted.
- `slot_specs` – optional JSON describing per-slot scoring logic for evaluation (used by product-page judging, ignored for classification tasks).

The cleaning step infers sensible defaults (e.g., system prompts for nav/product datasets) yet remains generic: add new task labels, prompts, and responses to repurpose the pipeline for future extraction tasks.

## Repository Layout
- `fabfile.py` – orchestrates Fabric task collections.
- `fab_tasks/` – thin wrappers delegating work to script modules.
- `scripts/` – placeholder implementations for dataset, training, evaluation, packaging, and operational helpers. Expand these with real logic as data and infra become available.
- `configs/` – YAML/TXT configs for datasets, training hyperparameters, judge behavior, evaluation prompts, packaging, and token usage projections.
- `data/` – empty directories reserved for raw (`data/raw`) and processed (`data/processed`) datasets.
  - Copy `dataset.duckdb` into `data/raw/` to provide training exemplars.
- `models/` – checkpoints, adapters, and packaged artifacts.
- `reports/` – generated evaluation summaries and judge outputs.
- `PLAN.md` – high-level roadmap and best practices.

## Pipeline Highlights
- **Dataset tooling** (`scripts/dataset_pipeline.py`) ingests DuckDB-sourced exemplars, validates them via Pydantic, normalizes prompts/targets for nav-page classification and product-page extraction, infers domains for stratification, and stratifies splits per `configs/datasets.yaml`. `fab dataset.clean` and `fab dataset.split` execute end-to-end once the `.duckdb` file is in `data/raw/`.
- **128k context training** (`configs/training.yaml`) applies rope scaling (`factor: 4.0`) and sets `max_seq_length: 131072`, ensuring the fine-tuned Gemma model can process the long prompt payloads generated by `NavPage.analyze` and `ProductPage.analyze`.
- **Training loop with Unsloth** (`scripts/training_pipeline.py`) loads configs from `configs/training.yaml`, instantiates Gemma 3 270M using `FastLanguageModel`, applies LoRA adapters, and runs SFT fine-tuning through TRL. `fab train.run` accepts `--config` overrides and optional `--resume`.
- **Evaluation with LiteLLM judge** (`scripts/evaluation_pipeline.py`) performs ROCm Torch+PEFT inference (optionally enabling Flash Attention 2 via `model.use_flash_attention_2`) to materialize candidates, compares them with references using LiteLLM, and emits Markdown summaries via `fab eval.judge` + `fab eval.report`.

## Next Implementation Steps
1. **Dataset acquisition**
   - Automate the creation of `data/raw/dataset.duckdb` on the source server and ensure it mirrors the expected schema.
   - Feed real paths + weights into `configs/datasets.yaml` and regenerate splits.

2. **Inference integration**
   - Extend `scripts/evaluation_pipeline.generate_outputs` with quantization / mixed precision settings once you decide on a deployment target (e.g., FP16 vs. AWQ). For now the Python path loads the merged PEFT checkpoint directly on ROCm Torch.
   - Follow AMD’s Hugging Face guidance if you enable Flash Attention 2 or GPTQ; mirror those toggles in `configs/training.yaml` / `configs/eval.yaml` when you flip them on.

3. **Hyperparameter tuning**
   - Adjust `configs/training.yaml` for batch size, learning rate, precision, and logging once hardware specs are finalized.

4. **Packaging pipeline**
   - Extend `scripts/packaging_pipeline.py` to merge LoRA weights, run ROCm compatibility checks, and emit manifest files ready for deployment consumers.

5. **Operational automation**
   - Flesh out token projections and backfill orchestration under `scripts/ops_pipeline.py` to close the loop with production monitoring.

## Best Practices
- Keep curated DuckDB exports (`data/raw/dataset.duckdb`) and processed `.jsonl` snapshots with rich metadata so they can be regenerated/revalidated, including the full prompt text used in NavPage/ProductPage analyses.
- Snapshot configs and dataset checksums for each training/eval run to guarantee reproducibility.
- Monitor token usage versus the 10M/day budget via `fab ops.project_tokens`.
- Record judge transcripts and maintain a manual review loop for regression failures.
- Store API tokens and environment-specific paths in `.env` (based on `.env.example`) so secrets never land in Git history.
- Validate ROCm settings (`ROCM_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, allocator tuning) on each host before launching 128k-context jobs; mismatched settings are the most common cause of OOMs or kernel faults.
- Enable Flash Attention 2 by setting `model.use_flash_attention_2: true` in `configs/training.yaml` / `configs/eval.yaml` after verifying the kernels are available (per AMD's Hugging Face guidance).

Refer to `PLAN.md` for deeper context, open questions, and long-term guidance.
