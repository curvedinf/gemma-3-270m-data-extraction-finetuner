"""Operational reporting helpers."""

from . import LOGGER


def project_token_usage(config_path: str) -> None:
    """
    Estimate daily token consumption to ensure we remain under budget.
    """
    LOGGER.info("Placeholder: project token usage leveraging config=%s", config_path)


def plan_backfill(since: str) -> None:
    """
    Generate a backfill plan when schema or model versions change.
    """
    LOGGER.info("Placeholder: plan backfill for data newer than %s", since)
