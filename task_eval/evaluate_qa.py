import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import argparse
from global_methods import set_openai_key
from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_aggr_acc
from task_eval.gpt_utils import get_gpt_answers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--use-rag', action="store_true")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--rag-mode', type=str, default="")
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retriever', type=str, default="remote")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--preds-file', type=str, default="", help='Optional path to append per-inference predictions as JSONL')
    return parser.parse_args()

def _get_metric_keys(args: argparse.Namespace) -> Tuple[str, str]:
    """根据参数生成评测所用的 model_key 和 prediction_key"""
    if not args.use_rag:
        model_key = args.model
    else:
        model_key = f"{args.model}_{args.rag_mode}_top_{args.top_k}"
        
    return model_key, f"{model_key}_prediction"


def _load_data(data_path: Path, out_path: Path) -> Tuple[List[Dict], Dict[str, Dict]]:
    """读取原始样本数据，以及已经存在的结果数据（用于断点续跑）"""
    with open(data_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    out_samples = {}
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            out_samples = {d['sample_id']: d for d in json.load(f)}
            
    return samples, out_samples


def _prepare_output_data(data: Dict, existing_out_samples: Dict) -> Dict:
    """整合当前数据与已有缓存，构造将要送入模型的干净字典"""
    sample_id = data['sample_id']
    
    # 优先使用已缓存的 qa 数据以避免重复处理，否则使用原数据
    source_qa = existing_out_samples.get(sample_id, data)['qa']
    
    return {
        'sample_id': sample_id,
        'qa': source_qa.copy()
    }


def _evaluate_and_update_metrics(answers: Dict, prediction_key: str, model_key: str, use_rag: bool) -> None:
    """调用评测函数，并将得分直接写入到 answers 字典中"""
    exact_matches, _, recall = eval_question_answering(answers['qa'], prediction_key)
    
    for i, qa_item in enumerate(answers['qa']):
        qa_item[f"{model_key}_f1"] = round(exact_matches[i], 3)
        if use_rag and len(recall) > 0:
            qa_item[f"{model_key}_recall"] = round(recall[i], 3)

def main():
    args = parse_args()

    # 环境与路径初始化
    if args.preds_file:
        Path(args.preds_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"****************** Evaluating Model {args.model} ***************")
    set_openai_key()

    # 获取配置 Keys 与加载数据
    model_key, prediction_key = _get_metric_keys(args)
    data_file_path = Path(args.data_file)
    out_file_path = Path(args.out_file)
    
    samples, out_samples = _load_data(data_file_path, out_file_path)

    # 核心处理循环
    for data in samples:
        # 准备数据 -> 获取模型预测 -> 评测算分 -> 存入内存
        out_data = _prepare_output_data(data, out_samples)
        
        get_gpt_answers(data, out_data, prediction_key, args)
        
        _evaluate_and_update_metrics(out_data, prediction_key, model_key, args.use_rag)
        
        out_samples[data['sample_id']] = out_data

    # 保存 JSON 结果
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(list(out_samples.values()), f, indent=2, ensure_ascii=False)
    
    # 触发整体聚合统计
    stats_file = str(out_file_path.with_suffix('')) + '_stats.json'
    analyze_aggr_acc(args.data_file, args.out_file, stats_file,
                     model_key, f"{model_key}_f1", rag=args.use_rag)

if __name__ == '__main__':
    main()