import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import argparse
from global_methods import set_openai_key
from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_aggr_acc
from task_eval.gpt_utils import get_gpt_answers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--use-rag', action="store_true")
    parser.add_argument('--use-4bit', action="store_true")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--rag-mode', type=str, default="")
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retriever', type=str, default="remote")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--preds-file', type=str, default="", help='Optional path to append per-inference predictions as JSONL')
    return parser.parse_args()


def main():

    # get arguments
    args = parse_args()

    if args.preds_file:
        Path(args.preds_file).parent.mkdir(parents=True, exist_ok=True)
    print(f"****************** Evaluating Model {args.model} ***************")
    
    set_openai_key()

    if not args.use_rag:
        model_key = args.model
    else:
        model_key = f"{args.model}_{args.rag_mode}_top_{args.top_k}"
        
    prediction_key = f"{model_key}_prediction"

    with open(args.data_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    out_samples = {}
    out_file_path = Path(args.out_file)
    if out_file_path.exists():
        with open(out_file_path, 'r', encoding='utf-8') as f:
            out_samples = {d['sample_id']: d for d in json.load(f)}

    for data in samples:
        out_data = {'sample_id': data['sample_id']}
        if data['sample_id'] in out_samples:
            out_data['qa'] = out_samples[data['sample_id']]['qa'].copy()
        else:
            out_data['qa'] = data['qa'].copy()

        answers = get_gpt_answers(data, out_data, prediction_key, args)

        # evaluate individual QA samples and save the score
        exact_matches, _, recall = eval_question_answering(answers['qa'], prediction_key)
        
        for i, qa_item in enumerate(answers['qa']):
            qa_item[f"{model_key}_f1"] = round(exact_matches[i], 3)
            if args.use_rag and len(recall) > 0:
                qa_item[f"{model_key}_recall"] = round(recall[i], 3)

        out_samples[data['sample_id']] = answers

    out_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(list(out_samples.values()), f, indent=2, ensure_ascii=False)
    
    stats_file = str(out_file_path.with_suffix('')) + '_stats.json'
    
    analyze_aggr_acc(args.data_file, args.out_file, stats_file,
                     model_key, f"{model_key}_f1", rag=args.use_rag)
    
if __name__ == '__main__':
    main()