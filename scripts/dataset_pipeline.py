"""Dataset-related helper functions."""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Optional, Tuple

import duckdb
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.table import Table

from . import LOGGER
from .config import SETTINGS
from .io_utils import load_yaml, read_jsonl, write_jsonl

PROCESSED_DIR = Path("data/processed")
console = Console()


class RawExample(BaseModel):
    """Schema for raw exemplars sourced from DuckDB."""

    id: str
    task: str = Field(default="generic")
    prompt_text: str
    response_strong: Any
    metadata: Dict = Field(default_factory=dict)
    system_prompt: Optional[str] = None
    slot_specs: Optional[Dict] = None


def _load_raw_examples(db_path: Path, table: str) -> List[Dict]:
    connection = duckdb.connect(str(db_path))
    try:
        cursor = connection.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
    finally:
        connection.close()

    records: List[Dict] = []
    for row in rows:
        records.append({column: value for column, value in zip(columns, row)})
    return records


def _normalize_raw_entry(entry: Dict) -> Dict:
    normalized = dict(entry)
    normalized["metadata"] = _ensure_dict(normalized.get("metadata"))

    slot_specs = normalized.get("slot_specs")
    normalized["slot_specs"] = _ensure_dict(slot_specs) if slot_specs is not None else None
    if normalized["slot_specs"] == {}:
        normalized["slot_specs"] = None

    prompt_candidates = [
        normalized.get("prompt_text"),
        normalized.get("prompt"),
        normalized.get("input_text"),
        normalized.get("user_prompt"),
    ]
    prompt_text = _first_non_empty(prompt_candidates)
    if not prompt_text:
        raise ValueError("Missing prompt text for exemplar.")
    normalized["prompt_text"] = prompt_text.strip()

    task_candidates = [
        normalized.get("task"),
        normalized["metadata"].get("task"),
        normalized.get("page_type"),
        normalized["metadata"].get("page_type"),
    ]
    task = _first_non_empty(task_candidates, default="generic")
    normalized["task"] = str(task).strip()

    response_candidate = normalized.get("response_strong")
    if response_candidate is None:
        for key in ("response_payload", "response", "target", "expected", "answer"):
            if key in normalized:
                response_candidate = normalized[key]
                break
    normalized["response_strong"] = _coerce_response(response_candidate)

    system_prompt = normalized.get("system_prompt") or normalized["metadata"].get("system_prompt")
    normalized["system_prompt"] = system_prompt or _default_system_prompt(normalized["task"])

    return normalized


def _ensure_dict(value) -> Dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _first_non_empty(values, default=None):
    for value in values:
        if isinstance(value, str):
            if value and value.strip():
                return value
        elif value is not None:
            return value
    return default


def _coerce_response(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            parsed = json.loads(stripped)
            return parsed
        except json.JSONDecodeError:
            return stripped
    return value


def _render_response_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, (dict, list)):
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
    return str(response).strip()


def _default_system_prompt(task: str) -> str:
    task_lower = (task or "").lower()
    if "nav" in task_lower:
        return (
            "You classify mutated navigation-page snippets to decide if they describe a single product page. "
            "Answer strictly with 'YES' or 'NO'."
        )
    if "product" in task_lower:
        return (
            "You extract structured data from mutated product-page snippets. Return only valid JSON for the requested keys."
        )
    return "You are a precise web-analysis assistant. Follow the instructions and honour the requested output format."


def _slugify(task: str) -> str:
    if not task:
        return "generic"
    slug = re.sub(r"[^a-z0-9]+", "_", task.lower())
    slug = slug.strip("_")
    return slug or "generic"


def clean(schema_version: str, output_dir: Optional[Path] = None) -> None:
    """
    Normalize DuckDB exemplars that already bundle instructions plus mutated HTML snippets.

    Produces per-task JSONL files in `data/processed/` with names `{task}.{schema_version}.jsonl`
    and corresponding `_latest` copies.
    """
    destination = output_dir or PROCESSED_DIR
    destination.mkdir(parents=True, exist_ok=True)

    db_path = Path(SETTINGS.dataset_db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Dataset DuckDB not found at {db_path}. Copy it into place via scp before cleaning."
        )

    raw_records = _load_raw_examples(db_path, SETTINGS.dataset_db_table)
    if not raw_records:
        LOGGER.warning(
            "No rows returned from %s.%s; ensure the DuckDB file contains training data.",
            db_path,
            SETTINGS.dataset_db_table,
        )
        return

    seen_ids = set()
    grouped_by_task: Dict[str, List[Dict]] = defaultdict(list)

    for entry in raw_records:
        try:
            normalized = _normalize_raw_entry(entry)
            example = RawExample(**normalized)
        except ValidationError as exc:
            LOGGER.warning("Skipping invalid example from DuckDB row %s: %s", entry.get("id"), exc)
            continue
        except ValueError as exc:
            LOGGER.warning("Skipping row %s due to normalization error: %s", entry.get("id"), exc)
            continue

        if example.id in seen_ids:
            continue
        seen_ids.add(example.id)

        cleaned_record = example.model_dump()
        cleaned_record["schema_version"] = schema_version

        metadata = cleaned_record.setdefault("metadata", {})
        metadata.setdefault("task", example.task)
        metadata.setdefault("system_prompt", example.system_prompt)
        if "domain" not in metadata:
            metadata["domain"] = _infer_domain(metadata, cleaned_record.get("prompt_text", ""))

        cleaned_record["assistant_response"] = _render_response_text(example.response_strong)
        cleaned_record["prompt_text"] = example.prompt_text.strip()
        cleaned_record["system_prompt"] = example.system_prompt
        cleaned_record["page_type"] = metadata.get("page_type") or example.task

        grouped_by_task[example.task].append(cleaned_record)

    if not grouped_by_task:
        LOGGER.warning("No valid examples discovered during cleaning.")
        return

    manifest = {
        "schema_version": schema_version,
        "input_source": str(db_path),
        "total_examples": sum(len(items) for items in grouped_by_task.values()),
        "task_counts": {task: len(items) for task, items in grouped_by_task.items()},
    }
    (destination / f"manifest.{schema_version}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for task, examples in grouped_by_task.items():
        slug = _slugify(task)
        versioned_path = destination / f"{slug}.{schema_version}.jsonl"
        latest_path = destination / f"{slug}_latest.jsonl"
        LOGGER.info("Writing %d examples to %s", len(examples), versioned_path)
        write_jsonl(versioned_path, examples)
        shutil.copy2(versioned_path, latest_path)


def split(config_path: str, input_file: Optional[Path] = None) -> None:
    """
    Split the processed dataset according to the provided config.
    """
    config = load_yaml(Path(config_path))
    defaults = config.get("defaults", {})
    stratify_keys = defaults.get("stratify", [])
    splits: Dict[str, float] = defaults.get(
        "splits", {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    seed = defaults.get("seed", 42)
    schema_version = config.get("metadata", {}).get("schema_version", "v1")

    source_examples: List[Dict] = []
    sources = config.get("sources", {})
    for label, source_cfg in sources.items():
        path = Path(source_cfg["path"])
        if not path.exists():
            LOGGER.warning("Source %s for %s not found; skipping.", path, label)
            continue
        source_examples.extend(read_jsonl(path))

    if not source_examples:
        raise RuntimeError("No processed examples available. Run `fab dataset.clean` first.")

    grouped = _stratify_examples(source_examples, stratify_keys)
    random_state = Random(seed)
    split_buckets: Dict[str, List[Dict]] = {name: [] for name in splits}

    for _, bucket_examples in grouped.items():
        random_state.shuffle(bucket_examples)
        allocations = _compute_split_allocations(len(bucket_examples), splits)
        cursor = 0
        for split_name, count in allocations.items():
            if count == 0:
                continue
            split_buckets[split_name].extend(bucket_examples[cursor : cursor + count])
            cursor += count

    for split_name, examples in split_buckets.items():
        output_name = f"{split_name}.{schema_version}.jsonl"
        output_path = PROCESSED_DIR / output_name
        LOGGER.info("Writing %d examples to %s", len(examples), output_path)
        write_jsonl(output_path, examples)


def report_stats(config_path: str) -> None:
    """
    Emit dataset descriptive statistics for the current snapshot.
    """
    config = load_yaml(Path(config_path))
    schema_version = config.get("metadata", {}).get("schema_version", "v1")
    split_paths = {
        name: PROCESSED_DIR / f"{name}.{schema_version}.jsonl"
        for name in config.get("defaults", {}).get("splits", {"train": 0.8, "validation": 0.1, "test": 0.1})
    }

    table = Table(title=f"Dataset Stats (schema={schema_version})")
    table.add_column("Split")
    table.add_column("Examples", justify="right")
    table.add_column("Page Types")
    table.add_column("Domains")

    for split_name, path in split_paths.items():
        if not path.exists():
            table.add_row(split_name, "0", "-", "-")
            continue

        page_counter: Counter = Counter()
        domain_counter: Counter = Counter()
        examples = list(read_jsonl(path))
        for example in examples:
            page_counter.update([example.get("page_type", "unknown")])
            metadata = example.get("metadata", {})
            domain = metadata.get("domain", "unknown")
            domain_counter.update([domain])

        table.add_row(
            split_name,
            str(len(examples)),
            ", ".join(f"{k}:{v}" for k, v in sorted(page_counter.items())),
            ", ".join(f"{k}:{v}" for k, v in sorted(domain_counter.items())),
        )

    console.print(table)


def _infer_domain(metadata: Dict, prompt_text: str) -> str:
    """Infer a domain or site label for stratification."""

    for candidate in ("domain", "site", "source_domain"):
        value = metadata.get(candidate) if metadata else None
        if value:
            return str(value)

    prompt = prompt_text or ""
    if "http" in prompt:
        start = prompt.find("http")
        url = prompt[start:].split()[0]
        if "/" in url:
            host = url.split("/")[2]
        else:
            host = url
        return host.lower()
    return "unknown"


def _stratify_examples(examples: List[Dict], stratify_keys: List[str]) -> Dict[Tuple, List[Dict]]:
    """
    Group examples by the provided stratification keys.
    """
    buckets: Dict[Tuple, List[Dict]] = defaultdict(list)
    for example in examples:
        key = []
        for strat in stratify_keys:
            if strat in example:
                key.append(example[strat])
            else:
                key.append(example.get("metadata", {}).get(strat, "unknown"))
        buckets[tuple(key)].append(example)
    return buckets or {("all",): examples}


def _compute_split_allocations(total: int, splits: Dict[str, float]) -> Dict[str, int]:
    """
    Determine how many examples from a bucket should go to each split while
    ensuring allocations sum to `total`.
    """
    if total == 0:
        return {name: 0 for name in splits}

    raw_counts = {name: total * ratio for name, ratio in splits.items()}
    floored = {name: int(math.floor(count)) for name, count in raw_counts.items()}
    remainder = total - sum(floored.values())

    fractional = sorted(
        ((name, raw_counts[name] - floored[name]) for name in splits),
        key=lambda item: item[1],
        reverse=True,
    )

    for idx in range(remainder):
        name, _ = fractional[idx % len(fractional)]
        floored[name] += 1

    return floored
