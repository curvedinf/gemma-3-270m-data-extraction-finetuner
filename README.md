# Gemma 3 270M Data Extraction Fine-Tuning Pipeline

This repository houses Fabric automation, configuration, and supporting scripts for building a fine-tuned Gemma 3 270M model that replaces the `navpage.analyze` and `productpage.analyze` strong-model calls in `../blazed_deals`. The pipeline covers dataset creation, training, evaluation with LLM-as-judge, and packaging artifacts suitable for ROCm vLLM runtime.

## Prerequisites
- Linux environment with ROCm-capable GPUs (for training/inference stages).
- Python 3.10+ recommended.
- Access to strong-model logs that provide *correct* JSON outputs to supervise Gemma.
- LiteLLM configured with the desired judge backend (environment variables or `litellm` config file).
- `git`, `make` (optional), and Fabric 3.x (installed via project requirements).

## Quickstart
1. **Clone & enter the repository**
   ```bash
   git clone <repo-url>
   cd gemma-3-270M-data-extraction-finetuner
   ```

2. **Create the virtual environment** (already scaffolded as `.venv/`)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   (Alternatively run `fab env.bootstrap` after cloning.)

3. **Explore Fabric task groups**
   ```bash
   fab -l
   fab dataset.pull
   fab train.prepare
   ```
   Each collection (`env`, `dataset`, `train`, `eval`, `package`, `ops`) aligns with the sections in `PLAN.md`.

## Repository Layout
- `fabfile.py` – orchestrates Fabric task collections.
- `fab_tasks/` – thin wrappers delegating work to script modules.
- `scripts/` – placeholder implementations for dataset, training, evaluation, packaging, and operational helpers. Expand these with real logic as data and infra become available.
- `configs/` – YAML/TXT configs for datasets, training hyperparameters, judge behavior, evaluation prompts, packaging, and token usage projections.
- `data/` – empty directories reserved for raw (`data/raw`) and processed (`data/processed`) datasets.
- `models/` – checkpoints, adapters, and packaged artifacts.
- `reports/` – generated evaluation summaries and judge outputs.
- `PLAN.md` – high-level roadmap and best practices.

## Pipeline Highlights
- **Dataset tooling** (`scripts/dataset_pipeline.py`) now validates raw exemplars via Pydantic, infers domains for stratification, performs deduplication, writes page-type snapshots, and stratifies splits per `configs/datasets.yaml`. `fab dataset.clean` and `fab dataset.split` execute end-to-end.
- **Training loop with Unsloth** (`scripts/training_pipeline.py`) loads configs from `configs/training.yaml`, instantiates Gemma 3 270M using `FastLanguageModel`, applies LoRA adapters, and runs SFT fine-tuning through TRL. `fab train.run` accepts `--config` overrides and optional `--resume`.
- **Evaluation with LiteLLM judge** (`scripts/evaluation_pipeline.py`) expects candidate outputs in `reports/model_outputs/<split>_candidates.jsonl`, compares them with references using LiteLLM-routed judge models, and generates Markdown summaries via `fab eval.judge` + `fab eval.report`.

## Next Implementation Steps
1. **Dataset acquisition**
   - Wire `fab dataset.pull` to your production export source (S3, BigQuery, etc.).
   - Feed real paths + weights into `configs/datasets.yaml` and regenerate splits.

2. **Inference integration**
   - Implement ROCm vLLM generation inside `scripts/evaluation_pipeline.generate_outputs`, producing candidate JSON for each split.

3. **Hyperparameter tuning**
   - Adjust `configs/training.yaml` for batch size, learning rate, precision, and logging once hardware specs are finalized.

4. **Packaging pipeline**
   - Extend `scripts/packaging_pipeline.py` to merge LoRA weights, run ROCm compatibility checks, and emit manifest files ready for deployment consumers.

5. **Operational automation**
   - Flesh out token projections and backfill orchestration under `scripts/ops_pipeline.py` to close the loop with production monitoring.

## Best Practices
- Keep datasets file-based (`.jsonl`) with rich metadata so they can be regenerated/revalidated.
- Snapshot configs and dataset checksums for each training/eval run to guarantee reproducibility.
- Monitor token usage versus the 10M/day budget via `fab ops.project_tokens`.
- Record judge transcripts and maintain a manual review loop for regression failures.

Refer to `PLAN.md` for deeper context, open questions, and long-term guidance.
