#!/usr/bin/env bash

set -euo pipefail

source scripts/env.sh

mkdir -p "$OUT_DIR" "$EMB_DIR"

python task_eval/get_facts.py \
	--data-file "$DATA_FILE_PATH" \
	--out-file "$OUT_DIR/$OBS_OUTPUT_FILE" \
	--prompt-dir "$PROMPT_DIR" \
	--emb-dir "$EMB_DIR" \
	--use-date \
	--overwrite \
	--retriever remote

python task_eval/evaluate_qa.py \
	--data-file "$DATA_FILE_PATH" \
	--out-file "$OUT_DIR/$QA_OUTPUT_FILE" \
	--model "$OPENAI_CHAT_MODEL" \
	--batch-size 1 \
	--overwrite \
	--preds-file "$OUT_DIR/preds.jsonl" \
	--use-rag \
	--retriever remote \
	--top-k 5 \
	--emb-dir "$EMB_DIR" \
	--rag-mode observation