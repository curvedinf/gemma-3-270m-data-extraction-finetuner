"""Evaluation helpers for model outputs and judge runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from litellm import completion

from . import LOGGER
from .io_utils import load_yaml, read_jsonl, write_jsonl

EVAL_OUTPUT_DIR = Path("reports/model_outputs")
JUDGE_REPORT_DIR = Path("reports/judge")
EVAL_CONFIG_PATH = Path("configs/eval.yaml")


def generate_outputs(split: str, config_path: str) -> None:
    """
    Run inference with the fine-tuned model to collect outputs for a dataset split.

    This function currently serves as a hook for integrating the ROCm vLLM
    inference pipeline. Populate it with vLLM calls that generate candidate JSON
    responses and persist to `reports/model_outputs/{split}_candidates.jsonl`.
    """
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Generation stub invoked for split=%s using config=%s. "
        "Integrate ROCm vLLM inference here.",
        split,
        config_path,
    )


def run_judging(split: str, config_path: str) -> None:
    """
    Invoke LiteLLM-backed judge to compare candidate vs reference outputs.
    """
    judge_config = load_yaml(Path(config_path))
    eval_config = load_yaml(EVAL_CONFIG_PATH)

    dataset_path = Path(eval_config.get("datasets", {}).get(split, ""))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Reference dataset for split '{split}' not found at {dataset_path}")

    candidate_file = EVAL_OUTPUT_DIR / f"{split}_candidates.jsonl"
    if not candidate_file.exists():
        raise FileNotFoundError(
            f"Candidate outputs for split '{split}' missing. Expected {candidate_file}. "
            "Run `fab eval.generate` after implementing inference."
        )

    references = {record["id"]: record for record in read_jsonl(dataset_path)}
    candidates = {record["id"]: record for record in read_jsonl(candidate_file)}

    defaults = judge_config.get("defaults", {})
    slots_config = judge_config.get("slots", {})
    levels = judge_config.get("levels", {})

    judge_model = defaults.get("judge_model")
    retry_limit = defaults.get("retry_on_invalid_json", 3)
    temperature = defaults.get("temperature", 0.0)
    max_tokens = defaults.get("max_tokens", 512)

    run_id = f"{split}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    output_path = JUDGE_REPORT_DIR / f"{run_id}_judge.jsonl"
    summary_path = JUDGE_REPORT_DIR / f"{run_id}_summary.json"
    JUDGE_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    judged_records: List[Dict] = []
    for example_id, reference in references.items():
        candidate_entry = candidates.get(example_id)
        if not candidate_entry:
            judged_records.append(
                {
                    "id": example_id,
                    "page_type": reference.get("page_type"),
                    "status": "missing_candidate",
                    "judge_output": None,
                }
            )
            continue

        judge_payload = _invoke_litellm_judge(
            judge_model=judge_model,
            reference=reference,
            candidate=candidate_entry,
            slots_config=slots_config,
            levels=levels,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_limit=retry_limit,
        )
        judged_records.append(
            {
                "id": example_id,
                "page_type": reference.get("page_type"),
                "status": "judged",
                "judge_output": judge_payload,
            }
        )

    write_jsonl(output_path, judged_records)

    summary = _summarize_judge_run(judged_records)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOGGER.info("Judge run complete. Results: %s", output_path)
    LOGGER.info("Summary stats: %s", summary_path)


def write_report(run_id: str, output_path: str) -> None:
    """
    Aggregate metrics and persist a human-readable evaluation report.
    """
    judge_file = JUDGE_REPORT_DIR / f"{run_id}_judge.jsonl"
    summary_file = JUDGE_REPORT_DIR / f"{run_id}_summary.json"

    if not judge_file.exists():
        raise FileNotFoundError(f"Judge run '{run_id}' not found at {judge_file}")

    records = list(read_jsonl(judge_file))
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    else:
        summary = _summarize_judge_run(records)

    lines = [
        f"# Evaluation Report – {run_id}",
        "",
        f"- Total examples: {summary['total_examples']}",
        f"- Judged examples: {summary['judged_examples']}",
        f"- Missing candidates: {summary['missing_candidates']}",
        f"- Pass rate (overall): {summary['overall_pass_rate']:.2%}",
    ]

    per_slot = summary.get("slot_pass_rates", {})
    if per_slot:
        lines.append("")
        lines.append("## Slot Pass Rates")
        for slot, rate in sorted(per_slot.items()):
            lines.append(f"- {slot}: {rate:.2%}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote evaluation report to %s", output_path)


def _invoke_litellm_judge(
    judge_model: str,
    reference: Dict,
    candidate: Dict,
    slots_config: Dict,
    levels: Dict,
    temperature: float,
    max_tokens: int,
    retry_limit: int,
) -> Dict:
    """
    Call LiteLLM with a structured prompt describing comparison rules.
    """
    if not judge_model:
        raise ValueError("Judge model must be specified in configs/judge_slots.yaml defaults.judge_model")

    reference_json = json.dumps(reference.get("response_strong", {}), ensure_ascii=False, sort_keys=True)
    candidate_payload = candidate.get("candidate") or candidate.get("response")
    if isinstance(candidate_payload, dict):
        candidate_json = json.dumps(candidate_payload, ensure_ascii=False, sort_keys=True)
    else:
        candidate_json = str(candidate_payload)

    prompt = _build_judge_prompt(reference_json, candidate_json, slots_config, levels)

    for attempt in range(retry_limit):
        response = completion(
            model=judge_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an impartial judge comparing structured JSON outputs field by field. "
                        "Return a JSON object describing pass/fail per slot and an overall pass verdict."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        message = response["choices"][0]["message"]["content"]
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            if attempt == retry_limit - 1:
                raise
    raise RuntimeError("Exceeded retry attempts for judge call.")


def _build_judge_prompt(reference_json: str, candidate_json: str, slots_config: Dict, levels: Dict) -> str:
    """
    Construct a human-readable prompt detailing slot-level similarity rules.
    """
    slot_rules = []
    for slot, cfg in slots_config.items():
        level = cfg.get("similarity")
        level_desc = levels.get(level, {}).get("description", "")
        extras = {k: v for k, v in cfg.items() if k != "similarity"}
        extra_str = ", ".join(f"{k}={v}" for k, v in extras.items()) if extras else "no additional constraints"
        slot_rules.append(f"- {slot}: {level} ({level_desc}); {extra_str}")

    rules_text = "\n".join(slot_rules)
    return (
        "Compare the candidate JSON to the reference JSON.\n"
        "Apply the following slot rules:\n"
        f"{rules_text}\n\n"
        "Respond with JSON using the structure:\n"
        '{"overall": {"pass": bool, "score": float, "notes": str}, '
        '"slots": {"slot_name": {"pass": bool, "score": float, "reason": str}}}\n'
        "Reference JSON:\n"
        f"{reference_json}\n\n"
        "Candidate JSON:\n"
        f"{candidate_json}\n"
    )


def _summarize_judge_run(records: List[Dict]) -> Dict:
    """
    Aggregate slot- and example-level statistics from judge output.
    """
    total = len(records)
    judged = 0
    missing = 0
    overall_pass = 0
    slot_counts: Dict[str, Dict[str, int]] = {}

    for record in records:
        if record["status"] != "judged":
            missing += 1
            continue
        judged += 1
        outcome = record.get("judge_output", {})
        if outcome.get("overall", {}).get("pass"):
            overall_pass += 1

        for slot_name, slot_result in (outcome.get("slots") or {}).items():
            slot_stats = slot_counts.setdefault(slot_name, {"pass": 0, "total": 0})
            slot_stats["total"] += 1
            if slot_result.get("pass"):
                slot_stats["pass"] += 1

    slot_pass_rates = {
        slot: (counts["pass"] / counts["total"]) if counts["total"] else 0.0
        for slot, counts in slot_counts.items()
    }

    return {
        "total_examples": total,
        "judged_examples": judged,
        "missing_candidates": missing,
        "overall_pass_rate": (overall_pass / judged) if judged else 0.0,
        "slot_pass_rates": slot_pass_rates,
    }
