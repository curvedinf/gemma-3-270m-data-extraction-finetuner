#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BIN="${REPO_ROOT}/.venv/bin"
FAB_BIN="${VENV_BIN}/fab"

# Load ROCm environment overrides if present
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/.env"
    set +a
fi

if [ ! -x "${FAB_BIN}" ]; then
    echo "[04_evaluation_pipeline] Virtual environment not found. Run ./01_setup_env.sh first." >&2
    exit 1
fi

SPLIT="${SPLIT:-validation}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval.yaml}"
JUDGE_CONFIG="${JUDGE_CONFIG:-configs/judge_slots.yaml}"

if [ -z "${RUN_ID:-}" ]; then
    RUN_ID="${SPLIT}_$(date -u +%Y%m%dT%H%M%SZ)"
    export RUN_ID
    CALCULATED_RUN_ID=true
else
    CALCULATED_RUN_ID=false
    export RUN_ID
fi

if [ ! -d "${REPO_ROOT}/models/checkpoints/run_latest" ]; then
    echo "[04_evaluation_pipeline] Expected checkpoint at models/checkpoints/run_latest is missing. Run training first." >&2
    exit 1
fi


EVAL_ARGS=("${FAB_BIN}" eval.generate --split "${SPLIT}" --config "${EVAL_CONFIG}")
if [ "${CALCULATED_RUN_ID}" = true ]; then
    EVAL_ARGS+=(--run-id "${RUN_ID}")
fi

echo "[04_evaluation_pipeline] Generating model outputs (split=${SPLIT}, config=${EVAL_CONFIG}, run_id=${RUN_ID})"
"${EVAL_ARGS[@]}"

JUDGE_ARGS=("${FAB_BIN}" eval.judge --split "${SPLIT}" --config "${JUDGE_CONFIG}")
if [ "${CALCULATED_RUN_ID}" = true ]; then
    JUDGE_ARGS+=(--run-id "${RUN_ID}")
fi

echo "[04_evaluation_pipeline] Running judge (split=${SPLIT}, config=${JUDGE_CONFIG}, run_id=${RUN_ID})"
"${JUDGE_ARGS[@]}"

echo "[04_evaluation_pipeline] Writing report (run_id=${RUN_ID})"
"${FAB_BIN}" eval.report --run-id "${RUN_ID}" --output "reports/${RUN_ID}_eval.md"

echo "[04_evaluation_pipeline] Done. Report saved to reports/${RUN_ID}_eval.md"
