#!/usr/bin/env bash

set -euo pipefail

source scripts/env.sh

mkdir -p "$OUT_DIR" "$EMB_DIR"

python task_eval/get_session_summaries.py \
	--data-file "$DATA_FILE_PATH" \
	--out-file "$OUT_DIR/$SESS_SUMM_OUTPUT_FILE" \
	--prompt-dir "$PROMPT_DIR" \
	--emb-dir "$EMB_DIR" \
	--overwrite \
	--use-date

python task_eval/evaluate_qa.py \
	--data-file "$DATA_FILE_PATH" \
	--out-file "$OUT_DIR/$QA_OUTPUT_FILE" \
	--emb-dir "$EMB_DIR" \
	--preds-file "$OUT_DIR/$PREDS_FILE" \
	--overwrite \
	--model "$OPENAI_CHAT_MODEL" \
	--batch-size 1 \
	--use-rag \
	--rag-mode summary \
	--top-k 5