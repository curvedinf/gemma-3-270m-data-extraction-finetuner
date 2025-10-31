"""Dataset-related helper functions."""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.table import Table

from . import LOGGER
from .io_utils import load_yaml, read_jsonl, write_jsonl

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
console = Console()


class RawExample(BaseModel):
    """Schema for raw strong-model exemplars."""

    id: str
    page_type: str
    html: str
    prompt: str
    response_strong: Dict
    metadata: Dict = Field(default_factory=dict)
    slot_specs: Optional[Dict] = None


def pull(source: str, output_dir: Optional[Path] = None) -> None:
    """
    Ingest reference examples from the strong model into raw storage.

    `source` may be a file or directory containing JSONL exports. Files are
    copied into `data/raw/` for downstream processing.
    """
    destination = output_dir or RAW_DIR
    destination.mkdir(parents=True, exist_ok=True)

    source_path = Path(source)
    if not source_path.exists():
        LOGGER.warning("Source %s not found. Ensure the upstream export exists.", source_path)
        return

    copied_files = 0
    if source_path.is_file():
        target = destination / source_path.name
        shutil.copy2(source_path, target)
        copied_files = 1
    elif source_path.is_dir():
        for jsonl_file in source_path.glob("*.jsonl"):
            target = destination / jsonl_file.name
            shutil.copy2(jsonl_file, target)
            copied_files += 1
    else:
        LOGGER.warning("Unsupported source path: %s", source_path)
        return

    LOGGER.info("Imported %d file(s) from %s into %s", copied_files, source_path, destination)


def clean(schema_version: str, input_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> None:
    """
    Validate and normalize raw HTML extraction examples.

    Produces per-page-type JSONL files in `data/processed/` with names
    `{page_type}.{schema_version}.jsonl` and corresponding `_latest` copies.
    """
    source_dir = input_dir or RAW_DIR
    destination = output_dir or PROCESSED_DIR
    destination.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(source_dir.glob("*.jsonl"))
    if not raw_files:
        LOGGER.warning("No raw JSONL files found under %s. Run `fab dataset.pull` first.", source_dir)
        return

    seen_ids = set()
    cleaned_by_page: Dict[str, List[Dict]] = defaultdict(list)

    for raw_file in raw_files:
        for record in read_jsonl(raw_file):
            try:
                example = RawExample(**record)
            except ValidationError as exc:
                LOGGER.warning("Skipping invalid example in %s: %s", raw_file, exc)
                continue

            if example.id in seen_ids:
                continue
            seen_ids.add(example.id)

            cleaned_record = example.model_dump()
            cleaned_record["html"] = example.html.strip()
            cleaned_record["prompt"] = example.prompt.strip()
            cleaned_record["schema_version"] = schema_version

            # Surface domain metadata for stratified splits if available.
            metadata = cleaned_record.setdefault("metadata", {})
            if "domain" not in metadata:
                metadata["domain"] = _infer_domain(example)

            cleaned_by_page[example.page_type].append(cleaned_record)

    if not cleaned_by_page:
        LOGGER.warning("No valid examples discovered during cleaning.")
        return

    manifest = {
        "schema_version": schema_version,
        "input_files": [str(path) for path in raw_files],
        "total_examples": sum(len(items) for items in cleaned_by_page.values()),
        "page_type_counts": {page: len(items) for page, items in cleaned_by_page.items()},
    }
    (destination / f"manifest.{schema_version}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for page_type, examples in cleaned_by_page.items():
        versioned_path = destination / f"{page_type}.{schema_version}.jsonl"
        latest_path = destination / f"{page_type}_latest.jsonl"
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


def _infer_domain(example: RawExample) -> str:
    """
    Attempt to infer the domain for stratification from metadata or prompt.
    """
    metadata = example.metadata or {}
    for candidate in ("domain", "site", "source_domain"):
        value = metadata.get(candidate)
        if value:
            return str(value)

    # Fall back to a naive heuristic from the prompt text.
    prompt = example.prompt
    if "http" in prompt:
        start = prompt.find("http")
        url = prompt[start:].split()[0]
        return url.split("/")[2] if "/" in url else url
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
