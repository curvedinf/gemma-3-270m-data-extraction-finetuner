## Goal
- Replace `navpage.analyze` and `productpage.analyze` calls in `../blazed_deals` with a fine-tuned Gemma 3 270M model specialized in extracting structured data from HTML while staying within a 10M tokens/day budget.
- Provide Fabric tasks that orchestrate dataset creation, training, evaluation, packaging, and deployment so the pipeline is fully automatable and reproducible.

## Guiding Principles & Best Practices
- **Data hygiene:** capture raw HTML, prompts, strong-model responses, and metadata; track source, timestamp, and extraction schema version to enable auditing and re-generation.
- **File-based datasets:** persist curated examples as newline-delimited JSON (`.jsonl`) with fields for prompt, reference response, metadata, and per-slot scoring rules; keep immutable snapshots for each training/eval run.
- **Reproducibility:** lock Python dependencies (e.g., `uv` + `requirements.lock`), pin base model revision, store configs that capture hyperparameters, seeds, and Fabric task inputs.
- **Token budget discipline:** benchmark prompt+completion lengths during evaluation, add Fabric tasks to report daily usage projections, and rate-limit batch inference jobs.
- **Model safety & drift:** retain holdout data, run LLM-as-judge regression tests before promotion, archive judge transcripts, and alert on score regressions.

## Repository Layout (proposed)
- `fabfile.py`: Fabric entrypoint defining grouped tasks (dataset, train, eval, deploy, utils).
- `configs/`: YAML/JSON configs for dataset splits, judge parameters, training hyperparams, deployment targets.
- `data/raw/`: unprocessed `.jsonl` dumps generated from production logging or ad-hoc pulls.
- `data/processed/`: cleaned and schema-validated `.jsonl` ready for training.
- `models/`: checkpoints, adapters, tokenizer artifacts, quantized builds.
- `reports/`: evaluation summaries, token-usage reports, judge transcripts.
- `scripts/`: helper Python scripts invoked by Fabric (training loop, evaluation runner, conversion utilities).

## Fabric Task Matrix
| Group | Task | Responsibility |
| --- | --- | --- |
| `env` | `./scripts/setup_env.sh` | Create Python env, install deps, scaffold dotenv file. |
| `dataset` | `scp user@server:/path/dataset.duckdb data/raw/dataset.duckdb` | Copy curated exemplars into the repo. |
|  | `fab dataset.clean` | Normalize prompts/targets, strip PII, enforce schema, dedupe. |
|  | `fab dataset.split` | Stratified split into train/val/test (config-driven). |
|  | `fab dataset.stats` | Emit summary stats (token counts, slot coverage). |
| `train` | `fab train.prepare` | Materialize training config, resolve paths, seed directories. |
|  | `fab train.run` | Launch fine-tuning (PEFT/LoRA or full) via chosen trainer. |
|  | `fab train.resume` | Resume from last checkpoint with updated config. |
| `eval` | `fab eval.generate` | Run fine-tuned model to produce outputs on eval set. |
|  | `fab eval.judge` | Invoke LLM-as-judge to compare model vs reference. |
|  | `fab eval.report` | Aggregate metrics, generate markdown/JSON reports. |
| `package` | `fab package.export` | Merge LoRA, quantize/export format needed by inference stack. |
| `ops` | `fab ops.project_tokens` | Estimate daily token burn vs 10M budget. |
|  | `fab ops.backfill` | Enqueue re-processing jobs when schema changes. |

## Data Creation Pipeline
1. **Capture strong-model references**  
   - Log HTML page source, prompt template (nav vs product), and strong-model extraction JSON.  
   - Append rows into a DuckDB database (default `data/raw/dataset.duckdb`) with table `examples(id TEXT PRIMARY KEY, task TEXT, prompt_text TEXT, response_strong JSON/TEXT, metadata JSON, slot_specs JSON NULLABLE, system_prompt TEXT NULLABLE)`.
2. **Curation & Augmentation**  
   - Run `dataset.clean` to strip tracking scripts, normalize whitespace, and serialize prompts/targets for each task (nav-page classification, product-page extraction, or future workflows).  
   - Optionally augment with adversarial HTML variants (DOM shuffles, malformed tags) to harden model.  
   - Attach `slot_specs` describing each extraction field, target type, tolerances.
3. **Schema Validation**  
   - Use `pydantic` or `jsonschema` in cleaning step; reject malformed entries, log issues to `reports/dataset_validation.md`.  
   - Maintain schema versioning to coordinate with downstream consumers.
4. **Splitting Strategy**  
   - Stratify by `page_type`, domain, language; ensure new domains only in validation/test to assess generalization.  
   - Snapshot splits (`data/processed/train.v1.jsonl` etc.) and record checksums in `configs/datasets.yaml`.
5. **Metadata & Governance**  
   - Track consent/compliance flags, ensure removal pipeline for takedown requests.  
   - Add Fabric command to diff dataset revisions before training runs.

## Training Strategy
- **Framework**: Use Unsloth (`FastLanguageModel` + `trl` SFTTrainer) on top of `transformers` + LoRA adapters for efficient fine-tuning; keep option for full fine-tune if GPU memory permits.
- **Base model**: Gemma 3 270M instruct checkpoint (confirm exact HF ID, e.g., `google/gemma-3-0.27b-it`), pin commit hash in config.
- **Input construction**:  
  ```
  <start_of_turn>system: {context instructions}\n<end_of_turn>
  <start_of_turn>user: {HTML + prompt framing}\n<end_of_turn>
  <start_of_turn>assistant: {reference JSON}\n<end_of_turn>
  ```
  Compress HTML via DOM summarization when needed; store tokenization heuristics in config.
- **Hyperparameters (initial defaults)**: learning rate 2e-4 (LoRA), batch size 64 tokens effective, 3 epochs, cosine decay, gradient accumulation, sequence length tuned per HTML size.  
- **Regularization**: mix-in synthetic negative examples (incorrect JSON) to encourage strict schema conformance; apply JSON formatting loss weighting.  
- **Monitoring**: log training metrics (loss, perplexity, slot-level F1) to `wandb`/`mlflow`; store config + git commit hash.

## Evaluation & LLM-as-Judge
- **Judge Model**: Use LiteLLM router to call strong judge models (e.g., gpt-4o-mini) with caching and rate controls.  
- **Process**:  
  1. `fab eval.generate` captures Gemma outputs on validation/test sets.  
  2. `fab eval.judge` feeds `(prompt, reference, candidate, slot_specs)` to judge prompt, requesting JSON verdict per slot.  
  3. Enforce deterministic formatting, retry on invalid JSON.  
  4. Store judge outputs in `reports/judge_run_{timestamp}.jsonl`.
- **Slot Similarity Levels** (configurable in `configs/judge_slots.yaml`):  
  | Level | Description | Example Fields | Scoring Rule | Pass Threshold |
  | --- | --- | --- | --- | --- |
  | `exact` | JSON value must exactly match reference | IDs, booleans | `candidate == reference` | 1.0 |
  | `substring` | Candidate must contain reference string | short descriptions | `reference in candidate` | 0.8 |
  | `numeric_tol` | Numeric within tolerance | prices, ratings | `abs(diff) <= tolerance` | 0.9 |
  | `set_match` | Order-invariant set equality | category lists | `set(candidate) == set(reference)` | 0.95 |
  | `date_norm` | Normalize to ISO date before compare | availability dates | parse + compare | 0.9 |
  | `llm_similarity` | Free-text closeness judged via semantic similarity | long descriptions | ask judge for 0-1 score | 0.85 |
- **Metrics**: overall slot accuracy, per-field pass rate, JSON validity rate, latency, token usage.  
- **Promotion Criteria**: establish minimum thresholds (e.g., 99% JSON validity, >= strong model on critical slots, <= 5% drop on non-critical slots).  
- **Human Spot-check**: sample failures for manual review, feed corrections back into dataset.

## Packaging Considerations
- Merge LoRA weights when needed and export `.safetensors`, tokenizer files, and inference config via `fab package.export`.  
- Target ROCm-compatible vLLM runtime: emit artifacts in FP16/bfloat16, confirm tokenizer + config align with vLLM loader requirements, and include ROCm build notes.  
- Optionally include AWQ/RTN quantized variants if ROCm vLLM pipeline supports them; document tested configs.  
- Emit manifest capturing model revision, dataset checksums, training config, and judge scores to accompany artifacts.  
- Run integrity checks (hashes, JSON schema validation of manifest) before marking package ready for integration.

## Validation & Continuous Improvement
- Schedule nightly eval runs; alert on degradation.  
- Maintain backlog of hard examples; auto-tag via judge feedback.  
- Periodically refresh training data with new production captures; re-run full fine-tuning or adapter updates.  
- Archive every training run (dataset checksums, configs, metrics) for reproducibility.

## Outstanding Questions / Follow-ups
- Confirm exact Hugging Face model ID for Gemma 3 270M and licensing constraints.  
- Decide judge model provider and cost controls (prompt caching, shared context).  
- Determine acceptable latency for nav/product extraction to size hardware appropriately.  
- Clarify data retention policies, especially if HTML contains user-sensitive content.  
- Align with `../blazed_deals` deployment process (CI/CD, container images, feature flags).
