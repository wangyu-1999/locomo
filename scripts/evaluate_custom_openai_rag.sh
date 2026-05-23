#!/usr/bin/env bash

set -euo pipefail

source scripts/env.sh

python3 task_eval/evaluate_qa.py \
--data-file "$DATA_FILE_PATH" --out-file "$OUT_DIR/$QA_OUTPUT_FILE" \
--model "gpt-3.5-turbo" --batch-size 1 --preds-file "./outputs/preds.jsonl" \
--use-rag --retriever remote --top-k 5 \
--emb-dir $EMB_DIR --rag-mode observation