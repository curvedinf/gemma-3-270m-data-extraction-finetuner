"""Training loop orchestration helpers leveraging Unsloth + TRL."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

from . import LOGGER
from .io_utils import load_yaml

CHECKPOINT_DIR = Path("models/checkpoints")


def prepare_run(config_path: str) -> None:
    """
    Perform pre-flight checks before launching a training run.
    """
    config = load_yaml(Path(config_path))
    run_cfg = config.get("run", {})
    data_cfg = config.get("data", {})

    output_dir = Path(run_cfg.get("output_dir", CHECKPOINT_DIR / "run_latest"))
    output_dir.mkdir(parents=True, exist_ok=True)

    _assert_exists(Path(data_cfg["train_file"]), "Training dataset")
    if eval_file := data_cfg.get("eval_file"):
        _assert_exists(Path(eval_file), "Evaluation dataset")
    _assert_exists(Path(data_cfg["prompt_template"]), "Prompt template")

    snapshot_path = output_dir / f"training_config.snapshot.{datetime.utcnow().isoformat()}.yaml"
    shutil.copy2(config_path, snapshot_path)
    LOGGER.info("Configuration snapshot written to %s", snapshot_path)


def run_training(config_path: str, resume_from: Optional[str] = None) -> None:
    """
    Execute the fine-tuning job using the provided configuration.
    """
    config = load_yaml(Path(config_path))
    run_cfg = config.get("run", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    evaluation_cfg = config.get("evaluation", {})
    peft_cfg = config.get("peft", {})

    output_dir = Path(run_cfg.get("output_dir", CHECKPOINT_DIR / "run_latest"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = _load_datasets(data_cfg)
    prompt_template = Path(data_cfg["prompt_template"]).read_text(encoding="utf-8")

    model, tokenizer = _load_model(config, training_cfg, peft_cfg)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatting_fn = lambda example: _format_example(example, prompt_template)  # noqa: E731

    training_args = _build_training_arguments(run_cfg, training_cfg, evaluation_cfg, output_dir)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("validation"),
        args=training_args,
        formatting_func=formatting_fn,
        max_seq_length=training_cfg.get("max_seq_length", 4096),
        packing=data_cfg.get("packing", False),
    )

    LOGGER.info("Starting training run (resume=%s)", resume_from)
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    LOGGER.info("Training complete. Artifacts saved to %s", output_dir)


def resume_training(checkpoint_path: str, config_path: str) -> None:
    """
    Resume a prior training run from the given checkpoint.
    """
    run_training(config_path=config_path, resume_from=checkpoint_path)


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")


def _load_datasets(data_cfg: Dict) -> Dict[str, Dict]:
    data_files = {"train": str(data_cfg["train_file"])}
    if eval_file := data_cfg.get("eval_file"):
        data_files["validation"] = str(eval_file)

    dataset_dict = load_dataset("json", data_files=data_files)
    return dataset_dict


def _load_model(config: Dict, training_cfg: Dict, peft_cfg: Dict):
    model_cfg = config.get("model", {})
    max_seq_length = training_cfg.get("max_seq_length", 4096)
    load_in_4bit = peft_cfg.get("load_in_4bit", True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model"],
        max_seq_length=max_seq_length,
        dtype=None,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        load_in_4bit=load_in_4bit,
        revision=model_cfg.get("revision"),
    )

    if peft_cfg.get("enabled", True):
        model = FastLanguageModel.get_peft_model(
            model,
            r=peft_cfg.get("r", 8),
            lora_alpha=peft_cfg.get("lora_alpha", 16),
            lora_dropout=peft_cfg.get("lora_dropout", 0.05),
            target_modules=peft_cfg.get("target_modules"),
            bias=peft_cfg.get("bias", "none"),
            modules_to_save=peft_cfg.get("modules_to_save"),
        )

    return model, tokenizer


def _build_training_arguments(run_cfg: Dict, training_cfg: Dict, evaluation_cfg: Dict, output_dir: Path) -> TrainingArguments:
    precision = (run_cfg.get("mixed_precision") or "").lower()
    bf16 = precision == "bf16"
    fp16 = precision == "fp16"

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg.get("epochs", 3),
        max_steps=training_cfg.get("max_steps"),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=training_cfg.get("lr", 2e-4),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
        logging_steps=run_cfg.get("log_steps", 20),
        save_steps=run_cfg.get("save_steps", 500),
        save_total_limit=evaluation_cfg.get("save_total_limit", 3),
        evaluation_strategy=evaluation_cfg.get("eval_strategy", "steps"),
        eval_steps=evaluation_cfg.get("eval_steps", 500),
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        report_to=training_cfg.get("report_to", []),
    )
    return args


def _format_example(example: Dict, template: str):
    html_context = example.get("html", "")
    metadata = example.get("metadata", {}) or {}
    instructions = metadata.get("task_instructions") or example.get("prompt", "")

    rendered_prompt = (
        template.replace("{{html_context}}", html_context.strip())
        .replace("{{task_instructions}}", instructions.strip())
        .rstrip()
    )

    response = json.dumps(example.get("response_strong", {}), ensure_ascii=False, sort_keys=True)
    conversation = f"{rendered_prompt}\n<start_of_turn>assistant:\n{response}\n<end_of_turn>"
    return [conversation]
