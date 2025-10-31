"""Training tasks for Gemma fine-tuning."""

from fabric import task

from scripts import training_pipeline


@task(help={"config": "Training configuration file path."})
def prepare(c, config="configs/training.yaml"):
    """
    Materialize training run directories, configs, and sanity checks.
    """
    training_pipeline.prepare_run(config_path=config)


@task(help={"config": "Training configuration file path.", "resume": "Checkpoint to resume from."})
def run(c, config="configs/training.yaml", resume=None):
    """
    Launch the fine-tuning job. Placeholder orchestrating call into training script.
    """
    training_pipeline.run_training(config_path=config, resume_from=resume)


@task(help={"checkpoint": "Checkpoint directory or file to resume from.", "config": "Training configuration path."})
def resume(c, checkpoint, config="configs/training.yaml"):
    """
    Resume training from an existing checkpoint.
    """
    training_pipeline.resume_training(checkpoint_path=checkpoint, config_path=config)
