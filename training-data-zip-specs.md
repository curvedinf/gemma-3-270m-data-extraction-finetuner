# LLM Training Dataset Zip Workflow

## Overview
- Live LLM call captures append to `backups/llm_training_dataset.duckdb`.
- `main_app.management.commands.archive_training_data` copies the live DuckDB under the same lock the writer uses, optimises the copy, compresses it, and resets the live DB so capture can continue without growth.
- Each archive run appends a new `llm_training_chunk_<timestamp>.duckdb` file (plus a JSON summary) into `backups/llm_training_dataset.zip`. The ZIP is the only artifact you need to ship to `../gemma-3-270m-data-extraction-finetuner`.

## Operational Flow
1. Capture pipeline (`call_llm(training_metadata=...)`) writes to the live DuckDB.
2. Run the archive command on a schedule (cron/Celery beat already configured to trigger daily at 5:00 AM CT):
   ```bash
   DATABASE_TYPE=sqlite ./venv/bin/python manage.py archive_training_data
   ```
   - Optional flags:
     * `--db-path` to target a different live DuckDB file.
     * `--zip-path` to change the destination archive.
     * `--compresslevel` (1-9) to tune ZIP compression (default=9).
3. The command:
   - Locks the live DB (`llm_training_dataset.duckdb.lock`).
   - Copies live data to a temp file, counts rows, and clears the live DB (fresh schema).
   - If no new rows exist, the job exits.
   - Otherwise, vacuums/checkpoints the temp copy, appends it to the ZIP as `llm_training_chunk_<timestamp>.duckdb`, and adds a JSON sidecar (`llm_training_chunk_<timestamp>.json`) with row counts and timestamp.
   - Deletes temporary files.
   - Result: the ZIP accumulates multiple chunks; each chunk covers the interval between archive runs.
4. Transfer `backups/llm_training_dataset.zip` to the training host.

## Training Repo Expectations
- **Source location:** Copy the ZIP into `../gemma-3-270m-data-extraction-finetuner/data/raw/`.
- **Consumption:** Update the finetuner dataset loader to detect `.zip` input. Iterate over `llm_training_chunk_*.duckdb` entries, extract each to a temp location, and append their `examples` tables into the processing pipeline (DuckDB `ATTACH` + `INSERT`, or equivalent). The JSON sidecars provide row counts/timestamps for logging.
- **Schema:** Matches the existing DuckDB table used by `scripts/dataset_pipeline.py.clean`:
  ```
  examples(
      id TEXT PRIMARY KEY,
      task TEXT,
      prompt_text TEXT,
      system_prompt TEXT,
      response_strong TEXT,
      metadata JSON,
      slot_specs JSON,
      llm_signature TEXT,
      provider TEXT,
      model_name TEXT,
      input_tokens BIGINT,
      output_tokens BIGINT,
      latency_ms BIGINT,
      created_at TIMESTAMP,
      prompt_hash TEXT
  )
  ```
- **Retention:** Because the live DB is cleared after each archive run *after* the chunk is appended, the ZIP holds the full history of captured examples. Rotate or rename the ZIP if you need point-in-time snapshots.

## Automation Recommendations
- Celery beat already issues `archive_training_data` daily at 5:00 AM CT; adjust the schedule in `blazed_deals/celery.py` if a different cadence is needed.
- Monitor archive size and timestamp; alert if the ZIP fails to update.
- Version the ZIP (e.g., `llm_training_dataset-YYYYmmdd.zip`) before transfer if you need multiple snapshots on the training side.
