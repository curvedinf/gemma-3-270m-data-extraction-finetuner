"""Evaluation and LLM-as-judge tasks."""

from fabric import task

from scripts import evaluation_pipeline


@task(help={
    "split": "Dataset split to evaluate (e.g., validation, test).",
    "config": "Evaluation config path.",
    "run_id": "Optional run identifier to reuse across stages.",
})
def generate(c, split="validation", config="configs/eval.yaml", run_id=None):
    """
    Run the fine-tuned model to produce outputs on the specified split.
    """
    evaluation_pipeline.generate_outputs(split=split, config_path=config, run_id=run_id)


@task(help={
    "split": "Dataset split whose outputs should be judged.",
    "config": "Judge configuration path.",
    "run_id": "Optional run identifier to reuse across stages.",
})
def judge(c, split="validation", config="configs/judge_slots.yaml", run_id=None):
    """
    Compare candidate outputs against reference responses using LLM-as-judge.
    """
    evaluation_pipeline.run_judging(split=split, config_path=config, run_id=run_id)


@task(help={"run_id": "Identifier for the evaluation run.", "output": "Path for the summary report."})
def report(c, run_id, output="reports/latest_eval.md"):
    """
    Aggregate judge results and emit a human-readable report.
    """
    evaluation_pipeline.write_report(run_id=run_id, output_path=output)
