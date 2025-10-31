"""Centralized configuration loading backed by dotenv."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _load_dotenv() -> None:
    """
    Load the repository-level `.env` file if present.

    The load is idempotent and keeps any already-exported environment variables
    untouched so CI/CD can inject secrets securely.
    """
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    else:
        load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    """Project configuration sourced from environment variables."""

    dataset_db_path: str
    dataset_db_table: str
    litellm_judge_model: Optional[str]
    litellm_api_key: Optional[str]
    huggingface_token: Optional[str]
    wandb_api_key: Optional[str]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton settings object.

    Defaults match the previous hard-coded values so existing flows continue to
    work without additional configuration.
    """
    _load_dotenv()
    return Settings(
        dataset_db_path=os.getenv("DATASET_DB_PATH", "data/raw/dataset.duckdb"),
        dataset_db_table=os.getenv("DATASET_DB_TABLE", "examples"),
        litellm_judge_model=os.getenv("LITELLM_JUDGE_MODEL"),
        litellm_api_key=os.getenv("LITELLM_API_KEY"),
        huggingface_token=os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN"),
        wandb_api_key=os.getenv("WANDB_API_KEY"),
    )


SETTINGS = get_settings()
