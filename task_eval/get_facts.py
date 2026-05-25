import sys
from pathlib import Path
import pickle as pkl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pickle
import argparse
from tqdm import tqdm

from global_methods import set_openai_key, run_json_trials
from task_eval.rag_utils import get_embeddings


CONVERSATION2FACTS_PROMPT = """
Write a concise and short list of all possible OBSERVATIONS about each speaker that can be gathered from the CONVERSATION. Each dialog in the conversation contains a dialogue id within square brackets. Each observation should contain a piece of information about the speaker, and also include the dialog id of the dialogs from which the information is taken. The OBSERVATIONS should be objective factual information about the speaker that can be used as a database about them. Avoid abstract observations about the dynamics between the two speakers such as 'speaker is supportive', 'speaker appreciates' etc. Do not leave out any information from the CONVERSATION. Important: Escape all double-quote characters within string output with backslash.\n\n
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', type=str, required=True)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--prompt-dir', type=str, default="")
    parser.add_argument('--use-date', action="store_true")
    parser.add_argument('--overwrite', action="store_true", help="set flag to overwrite existing outputs")
    parser.add_argument('--retriever', type=str, default="remote")

    return parser.parse_args()


def get_session_facts(args, agent_a, agent_b, session_idx, return_embeddings=True):
    prompt_path = Path(args.prompt_dir) / 'fact_generation_examples_new.json'
    with open(prompt_path, 'r', encoding='utf-8') as f:
        task = json.load(f)
        
    query = CONVERSATION2FACTS_PROMPT
    input_prefix = task.get('input_prefix', '')
    
    examples = [
        [f"{input_prefix}{e['input']}", json.dumps(e["output"], indent=2)] 
        for e in task.get('examples', [])
    ]

    # 构建对话内容
    session_key = f'session_{session_idx}'
    datetime_key = f'{session_key}_date_time'
    
    conv_lines = [agent_a.get(datetime_key, '')]
    
    for dialog in agent_a.get(session_key, []):
        dia_id = dialog.get("dia_id", "")
        speaker = dialog.get("speaker", "")
        
        text = dialog.get('clean_text', dialog.get('text', ''))
        
        line = f"[{dia_id}] {speaker} said, \"{text}\""
        
        if 'blip_caption' in dialog:
            line += f" and shared {dialog['blip_caption']}"
            
        conv_lines.append(line)
        
    # 用换行符拼接所有对话行
    conversation = '\n'.join(conv_lines) + '\n'
    
    # 执行推断
    input_text = f"{input_prefix}{conversation}"
    facts = run_json_trials(
        query, num_gen=1, num_tokens_request=500, 
        use_16k=False, examples=examples, input=input_text
    )

    if not return_embeddings:
        return facts


def _load_data(data_file_path: Path, out_file_path: Path):
    """安全读取输入样本和断点续跑的现有输出"""
    with open(data_file_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    out_samples = {}
    if out_file_path.exists():
        with open(out_file_path, 'r', encoding='utf-8') as f:
            out_samples = {d['sample_id']: d for d in json.load(f)}
            
    return samples, out_samples


def _save_out_samples(out_samples: dict, out_file_path: Path):
    """将预测结果持久化为 JSON 文件"""
    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(list(out_samples.values()), f, indent=2, ensure_ascii=False)


def _get_session_range(conversation: dict):
    """提取会话的最早和最晚编号。如果不存在，返回 (None, None)"""
    if not conversation:
        return None, None
        
    session_nums = [
        int(k.split('_')[-1]) 
        for k in conversation.keys() 
        if 'session' in k and 'date_time' not in k
    ]
    
    if not session_nums:
        return None, None
        
    return min(session_nums), max(session_nums)


def _process_observations(args, sample_id, data, output, min_sess, max_sess, out_samples, out_file_path):
    """遍历会话生成 facts，并返回用于 embedding 的列表集"""
    observations = []
    date_times = []
    context_ids = []

    for i in tqdm(range(min_sess, max_sess + 1), desc=f'Generating observations for {sample_id}'):
        session_obs_key = f'session_{i}_observation'
        
        # 优先使用已存在的数据
        if 'observation' in data and session_obs_key in data['observation']:
            facts = data['observation'][session_obs_key]
            output[session_obs_key] = facts
        else:
            if session_obs_key not in output or args.overwrite:
                facts = get_session_facts(
                    args, data['conversation'], data['conversation'], i, return_embeddings=False
                )
                output[session_obs_key] = facts
            else:
                facts = output[session_obs_key]

        date_time = data['conversation'].get(f'session_{i}_date_time', "")
        
        for _, v in facts.items():
            for fact, dia_id in v:
                observations.append(fact)
                context_ids.append(dia_id)
                date_times.append(date_time)

        # 边执行边保存，防止中断丢失
        out_samples[sample_id] = output
        _save_out_samples(out_samples, out_file_path)

    return observations, date_times, context_ids


def _generate_and_save_embeddings(args, sample_id, observations, date_times, context_ids, emb_dir: Path, dataset_prefix: str):
    """获取 embeddings 并将组合好的 database 存入 pickle"""
    if not observations:
        return
        
    if args.use_date:
        inputs = [f"{dt}. {obs}" for dt, obs in zip(date_times, observations)]
        embeddings = get_embeddings(args.retriever, inputs, 'context')
    else:
        embeddings = get_embeddings(args.retriever, observations, 'context')
    
    assert embeddings.shape[0] == len(observations), "Embeddings dimension mismatch!"

    database = {
        'embeddings': embeddings,
        'date_time': date_times,
        'dia_id': context_ids,
        'context': observations
    }

    pkl_path = emb_dir / f"{dataset_prefix}_observation_{sample_id}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(database, f)

def main():
    args = parse_args()
    set_openai_key()

    data_file_path = Path(args.data_file)
    out_file_path = Path(args.out_file)

    # 加载数据
    samples, out_samples = _load_data(data_file_path, out_file_path)
    dataset_prefix = data_file_path.stem

    # 确定 embedding 保存目录
    emb_dir = Path(args.emb_dir) if args.emb_dir else out_file_path.parent
    emb_dir.mkdir(parents=True, exist_ok=True)

    # 遍历处理数据集
    for data in samples:
        sample_id = data['sample_id']
        output = out_samples.get(sample_id, {'sample_id': sample_id})

        # 提取会话范围
        min_sess, max_sess = _get_session_range(data.get('conversation', {}))
        if min_sess is None:
            continue

        # 生成或读取会话 facts 
        observations, date_times, context_ids = _process_observations(
            args, sample_id, data, output, min_sess, max_sess, out_samples, out_file_path
        )

        # 生成 embeddings 并持久化为 Pickle
        _generate_and_save_embeddings(
            args, sample_id, observations, date_times, context_ids, emb_dir, dataset_prefix
        )

        out_samples[sample_id] = output
    
    # 循环结束后最后一次全量保存 JSON
    _save_out_samples(out_samples, out_file_path)


if __name__ == '__main__':
    main()