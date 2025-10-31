"""Packaging helpers for producing ROCm vLLM compatible artifacts."""

from pathlib import Path

from . import LOGGER


EXPORT_ROOT = Path("models/export")


def export_artifacts(config_path: str, output_path: str) -> None:
    """
    Prepare packaged model assets suitable for ROCm vLLM.
    """
    destination = Path(output_path)
    destination.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Placeholder: package model using config=%s into %s (ROCm vLLM compatibility checks pending)",
        config_path,
        destination,
    )
