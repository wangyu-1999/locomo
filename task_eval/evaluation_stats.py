import os
import json
import math
from tqdm import tqdm
from collections import defaultdict


def get_conversation_lengths(data, encoder=None):
    total_conv_length = 0
    id2length = {}
    
    for sess_num in range(1, 50):
        session_key = f'session_{sess_num}'
        session_data = data.get(session_key)
        
        if not session_data:
            continue

        for dialog in session_data:
            dialog_tokens = f"{dialog['speaker']}: {dialog['text']}\n"
            
            # 安全地获取字典键值
            if dialog.get("img_file"):
                dialog_tokens += f"[shares {dialog['blip_caption']}]\n"
                
            if encoder is not None:
                dialog_length = len(encoder.encode(dialog_tokens))
            else:
                dialog_length = len(dialog_tokens)
                
            id2length[dialog["dia_id"]] = total_conv_length + dialog_length
            total_conv_length += dialog_length
            
    return id2length


def _load_sample_data(in_file, ann_file):
    """安全读取推断输出和原始标注数据"""
    with open(in_file, 'r', encoding='utf-8') as f:
        outputs = {d['sample_id']: d for d in json.load(f)}
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = {d['sample_id']: d for d in json.load(f)}
    return outputs, data


def _parse_evidence(evidence_list):
    """清理并解析证据列表，返回 (session_id, dialog_id) 格式的元组列表"""
    parsed_ev = []
    cleaned_evidence = [q.replace('(', '').replace(')', '') for q in evidence_list if q]
    for e in cleaned_evidence:
        if e:
            parts = e.split(':')
            sess_val = int(parts[0][1:])
            dia_val = int(parts[-1])
            parsed_ev.append((sess_val, dia_val))
    return parsed_ev


def _update_memory_metrics(parsed_ev, id2length, category, metric_val, 
                           memory_counts_og, memory_counts, context_len_og, context_len_counts):
    """计算基于证据距离的内存统计信息"""
    farthest_session = min(ev[0] for ev in parsed_ev)
    farthest_dialog = min(ev[1] for ev in parsed_ev if ev[0] == farthest_session)

    farthest_length = id2length[f"D{farthest_session}:{farthest_dialog}"]
    farthest_bucket = math.ceil(farthest_length / 1000)

    memory_counts_og[category][farthest_bucket] += 1
    memory_counts[category][farthest_bucket] += metric_val

    if category == 1:
        latest_session = max(ev[0] for ev in parsed_ev)
        latest_dialog = max(ev[1] for ev in parsed_ev if ev[0] == latest_session)

        latest_length = id2length[f"D{latest_session}:{latest_dialog}"]
        context_length = latest_length - farthest_length
        context_bucket = math.ceil(context_length / 1000)
        
        context_len_og[context_bucket] += 1
        context_len_counts[context_bucket] += metric_val


def _print_evaluation_summary(total_counts, acc_counts, recall_by_category, rag):
    """打印类别准确率与召回率的汇总统计"""
    print("Total number of questions and corresponding accuracy in each category: ")
    total_k = 0
    total_v = 0
    keys = [4, 1, 2, 3, 5]
    
    for k in keys:
        v = total_counts[k]
        if v == 0:
            print(f"No questions found in category {k}")
        else:
            print(k, v, acc_counts[k], round(acc_counts[k] / v, 3))
        total_v += acc_counts[k]
        total_k += v

    overall_acc = round(total_v / total_k, 3) if total_k > 0 else 0.0
    print(f"Overall accuracy: {overall_acc}")

    if rag:
        print("Category and corresponding recall accuracy in each category: ")
        for k in keys:
            v = recall_by_category[k]
            if total_counts[k] == 0:
                print(f"No questions found in category {k}")
            else:
                print(k, round(v / total_counts[k], 3))
                
        total_recall = sum(recall_by_category.values())
        total_qs = sum(total_counts.values())
        print("Overall recall accuracy: ", round(total_recall / total_qs, 3) if total_qs > 0 else 0.0)


def _save_evaluation_results(out_file, model_name, rag, total_counts, acc_counts, 
                             recall_by_category, memory_counts_og, memory_counts, 
                             context_len_og, context_len_counts):
    """构建字典配置，并保存至输出 JSON 文件"""
    results_dict = {}
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            results_dict = json.load(f)

    results_dict[model_name] = {
        'category_counts': dict(total_counts),
        'cum_accuracy_by_category': dict(acc_counts)
    }

    if rag:
        results_dict[model_name]['recall_by_category'] = {
            k: v / total_counts[k] for k, v in recall_by_category.items() if total_counts[k] > 0
        }
    else:
        results_dict[model_name]['category_counts_by_memory'] = {k: dict(v) for k, v in memory_counts_og.items()}
        results_dict[model_name]['cum_accuracy_by_category_by_memory'] = {k: dict(v) for k, v in memory_counts.items()}
        results_dict[model_name]['context_length_counts'] = dict(context_len_og)
        results_dict[model_name]['cum_accuracy_by_context_length'] = dict(context_len_counts)

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)


def analyze_aggr_acc(ann_file, in_file, out_file, model_name, metric_key, encoder=None, rag=False):
    total_counts = defaultdict(int)
    acc_counts = defaultdict(int)
    memory_counts = defaultdict(lambda: defaultdict(int))
    memory_counts_og = defaultdict(lambda: defaultdict(int))
    context_len_counts = defaultdict(int)
    context_len_og = defaultdict(int)
    recall_by_category = defaultdict(int)

    # 加载数据
    outputs, data = _load_sample_data(in_file, ann_file)
        
    for sample_id, output in outputs.items():
        ann = data.get(sample_id)
        if not ann:
            continue
            
        id2length = get_conversation_lengths(ann['conversation'], encoder)

        # 遍历问答条目并计算统计量
        for qa in tqdm(output.get('qa', [])):
            category = qa['category']
            total_counts[category] += 1
            
            if metric_key in qa:
                metric_val = qa[metric_key]
                acc_counts[category] += metric_val
                
                # 清洗并解析证据
                parsed_ev = _parse_evidence(qa.get("evidence", []))
                
                if parsed_ev:
                    if rag:
                        recall_by_category[category] += qa.get(f"{model_name}_recall", 0)
                    else:
                        try:
                            # 计算基于对话长度与内存的评测指标
                            _update_memory_metrics(
                                parsed_ev, id2length, category, metric_val,
                                memory_counts_og, memory_counts,
                                context_len_og, context_len_counts
                            )
                        except Exception:
                            continue
            else:
                print([k for k in qa.keys() if 'mistral' in k], metric_key)

    # 打印汇总报告
    _print_evaluation_summary(total_counts, acc_counts, recall_by_category, rag)

    # 持久化分析结果
    _save_evaluation_results(
        out_file, model_name, rag,
        total_counts, acc_counts, recall_by_category,
        memory_counts_og, memory_counts,
        context_len_og, context_len_counts
    )