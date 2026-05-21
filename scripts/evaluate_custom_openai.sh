#!/usr/bin/env bash

set -euo pipefail

source scripts/env.sh

: "${OPENAI_CHAT_MODEL:?Set OPENAI_CHAT_MODEL to your custom OpenAI-compatible chat model before running}"

BATCH_SIZE="${BATCH_SIZE:-1}"
MODEL_NAME="${MODEL_NAME:-gpt-3.5-turbo}"

extra_args=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
extra_args+=(--overwrite)
fi

if [ ${#extra_args[@]} -gt 0 ]; then
python3 task_eval/evaluate_qa.py \
--data-file "$DATA_FILE_PATH" --out-file "$OUT_DIR/$QA_OUTPUT_FILE" \
--model "$MODEL_NAME" --batch-size "$BATCH_SIZE" "${extra_args[@]}"
else
python3 task_eval/evaluate_qa.py \
--data-file "$DATA_FILE_PATH" --out-file "$OUT_DIR/$QA_OUTPUT_FILE" \
--model "$MODEL_NAME" --batch-size "$BATCH_SIZE"
fi