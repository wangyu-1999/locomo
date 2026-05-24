#!/usr/bin/env bash

set -euo pipefail

source scripts/env.sh

python3 task_eval/evaluate_qa.py \
--data-file "$DATA_FILE_PATH" \
--out-file "$OUT_DIR/$QA_OUTPUT_FILE" \
--preds-file "$OUT_DIR/$PREDS_FILE" \
--overwrite \
--model "$OPENAI_CHAT_MODEL" \
 --batch-size 1