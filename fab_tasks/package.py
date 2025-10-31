"""Packaging tasks for ROCm vLLM deployment artifacts."""

from fabric import task

from scripts import packaging_pipeline


@task(help={"config": "Packaging configuration path.", "output": "Destination directory for artifacts."})
def export(c, config="configs/package.yaml", output="models/export"):
    """
    Merge adapters, produce ROCm-friendly artifacts, and create manifest.
    """
    packaging_pipeline.export_artifacts(config_path=config, output_path=output)
