"""Operational utility tasks."""

from fabric import task

from scripts import ops_pipeline


@task(help={"config": "Path to usage projection configuration file."})
def project_tokens(c, config="configs/usage.yaml"):
    """
    Estimate daily token consumption vs the 10M token budget.
    """
    ops_pipeline.project_token_usage(config_path=config)


@task(help={"since": "ISO timestamp to start the backfill from."})
def backfill(c, since):
    """
    Schedule or enqueue backfill jobs when the extraction schema changes.
    """
    ops_pipeline.plan_backfill(since=since)
