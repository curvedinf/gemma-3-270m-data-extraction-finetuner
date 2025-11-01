"""Training loop orchestration helpers leveraging Unsloth + TRL."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .config import SETTINGS  # noqa: F401 - ensure .env is loaded before ROCm checks

try:
    from unsloth import FastLanguageModel  # Import before transformers/trl for optimizations
except (NotImplementedError, ImportError) as exc:  # pragma: no cover - GPU guard
    FastLanguageModel = None  # type: ignore
    UNSLOTH_IMPORT_ERROR = exc
else:
    UNSLOTH_IMPORT_ERROR = None

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

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
    if FastLanguageModel is None:
        raise RuntimeError(
            "Unsloth could not initialize a ROCm device. "
            "Ensure torch detects your GPU and retry. "
            f"Original error: {UNSLOTH_IMPORT_ERROR}"
        )

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

    max_seq_length = training_cfg.get("max_seq_length", 131072)
    model, tokenizer = _load_model(config, training_cfg, peft_cfg, max_seq_length=max_seq_length)
    tokenizer.model_max_length = max_seq_length
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
        max_seq_length=max_seq_length,
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


def _load_model(config: Dict, training_cfg: Dict, peft_cfg: Dict, *, max_seq_length: int):
    model_cfg = config.get("model", {})
    load_in_4bit = peft_cfg.get("load_in_4bit", True)
    rope_scaling = model_cfg.get("rope_scaling")
    use_fa2 = model_cfg.get("use_flash_attention_2")

    model_kwargs = dict(
        model_name=model_cfg["base_model"],
        max_seq_length=max_seq_length,
        dtype=None,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        load_in_4bit=load_in_4bit,
        revision=model_cfg.get("revision"),
    )
    if rope_scaling:
        model_kwargs["rope_scaling"] = rope_scaling
    if use_fa2 is True:
        model_kwargs["use_flash_attention_2"] = use_fa2

    model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
    if rope_scaling:
        model.config.rope_scaling = rope_scaling
    model.config.max_position_embeddings = max_seq_length

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


def _build_training_arguments(
    run_cfg: Dict, training_cfg: Dict, evaluation_cfg: Dict, output_dir: Path
) -> "TrainingArguments":
    from transformers import TrainingArguments
    precision = (run_cfg.get("mixed_precision") or "").lower()
    bf16 = precision == "bf16"
    fp16 = precision == "fp16"

    eval_strategy = evaluation_cfg.get("eval_strategy") or evaluation_cfg.get("evaluation_strategy")

    max_steps = training_cfg.get("max_steps")
    if max_steps is None:
        max_steps = -1
    epochs = training_cfg.get("epochs", 3)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=training_cfg.get("lr", 2e-4),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
        logging_steps=run_cfg.get("log_steps", 20),
        save_steps=run_cfg.get("save_steps", 500),
        save_total_limit=evaluation_cfg.get("save_total_limit", 3),
        eval_strategy=eval_strategy or "steps",
        eval_steps=evaluation_cfg.get("eval_steps", 500),
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        report_to=training_cfg.get("report_to", []),
    )
    return args


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_coerce_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _format_example(example: Dict, template: str):
    metadata = example.get("metadata", {}) or {}
    system_prompt = _coerce_text(
        example.get("system_prompt")
        or metadata.get("system_prompt")
        or "You are a precise assistant."
    ).strip()
    user_prompt = _coerce_text(
        example.get("prompt_text")
        or example.get("prompt")
        or metadata.get("user_prompt")
        or ""
    ).strip()

    rendered_prompt = (
        template.replace("{{system_prompt}}", system_prompt)
        .replace("{{user_prompt}}", user_prompt)
        .rstrip()
    )

    if assistant_response := example.get("assistant_response"):
        response_text = _coerce_text(assistant_response).strip()
    else:
        payload = example.get("response_strong")
        if isinstance(payload, (dict, list)):
            response_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        elif payload is None:
            response_text = ""
        else:
            response_text = str(payload).strip()

    conversation = f"{rendered_prompt}\n<start_of_turn>assistant:\n{response_text}\n<end_of_turn>"
    return [conversation]
