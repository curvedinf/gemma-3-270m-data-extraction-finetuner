"""Dataset preparation tasks."""

from fabric import task

from scripts import dataset_pipeline


@task(help={"schema_version": "Semantic version for the extraction schema."})
def clean(c, schema_version="v1"):
    """
    Normalize, redact, and validate raw examples before training.
    """
    dataset_pipeline.clean(schema_version=schema_version)


@task(help={"config": "Path to dataset split configuration file."})
def split(c, config="configs/datasets.yaml"):
    """
    Produce stratified train/val/test splits and record metadata.
    """
    dataset_pipeline.split(config_path=config)


@task(help={"config": "Path to dataset stats configuration file."})
def stats(c, config="configs/datasets.yaml"):
    """
    Generate coverage and token-count summaries for curated datasets.
    """
    dataset_pipeline.report_stats(config_path=config)
