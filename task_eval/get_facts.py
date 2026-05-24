import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pickle
import argparse
from tqdm import tqdm

from generative_agents.memory_utils import get_session_facts
from global_methods import set_openai_key
from task_eval.rag_utils import get_embeddings

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


def main():
    set_openai_key()
    args = parse_args()

    data_file_path = Path(args.data_file)
    out_file_path = Path(args.out_file)

    with open(data_file_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # 检查并加载已有输出，用于断点续跑
    out_samples = {}
    if out_file_path.exists():
        with open(out_file_path, 'r', encoding='utf-8') as f:
            out_samples = {d['sample_id']: d for d in json.load(f)}

    dataset_prefix = data_file_path.stem

    # 确定 embedding 保存目录
    emb_dir = Path(args.emb_dir) if args.emb_dir else out_file_path.parent
    emb_dir.mkdir(parents=True, exist_ok=True)

    for data in samples:
        sample_id = data['sample_id']
        observations = []
        date_times = []
        context_ids = []

        # 使用 get 获取，代码更紧凑
        output = out_samples.get(sample_id, {'sample_id': sample_id})

        # 提取当前样本的 session 编号集合
        session_nums = [
            int(k.split('_')[-1]) 
            for k in data.get('conversation', {}).keys() 
            if 'session' in k and 'date_time' not in k
        ]
        
        # 边界保护，防止因为空会话报错
        if not session_nums:
            continue

        min_sess, max_sess = min(session_nums), max(session_nums)

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
            
            # 使用 `_` 替代未使用的变量 `k`
            for _, v in facts.items():
                for fact, dia_id in v:
                    observations.append(fact)
                    context_ids.append(dia_id)
                    date_times.append(date_time)

            # 边执行边保存，防止中断丢失 (去除了多余的 output.copy())
            out_samples[sample_id] = output
            with open(out_file_path, 'w', encoding='utf-8') as f:
                json.dump(list(out_samples.values()), f, indent=2, ensure_ascii=False)

        # 准备 RAG 检索的 Embeddings
        if args.use_date:
            inputs = [f"{dt}. {obs}" for dt, obs in zip(date_times, observations)]
            embeddings = get_embeddings(args.retriever, inputs, 'context')
        else:
            embeddings = get_embeddings(args.retriever, observations, 'context')
        
        assert embeddings.shape[0] == len(observations), "Embeddings dimension mismatch!"

        # 保存为 Pickle 格式
        database = {
            'embeddings': embeddings,
            'date_time': date_times,
            'dia_id': context_ids,
            'context': observations
        }

        pkl_path = emb_dir / f"{dataset_prefix}_observation_{sample_id}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(database, f)

        out_samples[sample_id] = output
    
    # 循环结束后最后一次全量保存
    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(list(out_samples.values()), f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()