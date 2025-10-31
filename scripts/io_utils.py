"""Utility helpers for reading and writing project data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import yaml


def read_jsonl(path: Path) -> Iterator[Dict]:
    """
    Yield parsed JSON objects from a newline-delimited JSON file.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} in {path}") from exc


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    """
    Persist dictionaries to disk as newline-delimited JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def load_yaml(path: Path) -> Dict:
    """
    Load a YAML document into a Python dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: Dict) -> None:
    """
    Write a Python dictionary to YAML.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
