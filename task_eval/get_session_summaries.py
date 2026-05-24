import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pickle
import argparse
from tqdm import tqdm

from global_methods import set_openai_key, run_chatgpt
from task_eval.rag_utils import get_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', type=str, required=True)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--prompt-dir', type=str, default="")
    parser.add_argument('--use-date', action="store_true")
    parser.add_argument('--overwrite', action="store_true", help="set flag to overwrite existing outputs")
    parser.add_argument('--retriever', type=str, default="dragon")

    return parser.parse_args()


def get_summary_query(session, date_time):
    conv_lines = [f"{date_time}\n"]
    
    for dialog in session:
        speaker = dialog.get('speaker', '')
        text = dialog.get('text', '')
        line = f"{speaker} said, \"{text}\""
        
        if 'blip_caption' in dialog:
            line += f" and shared {dialog['blip_caption']}."
        
        conv_lines.append(line + "\n")

    conv_text = "".join(conv_lines)
    
    query = (
        "Generate a concise summary of the following conversation using exact words "
        "from the conversation wherever possible. The summary should contain all facts "
        "about the two speakers, as well as references to time.\n"
        f"{conv_text}\n"
    )
    return query


def get_session_summary(session, date_time):
    query = get_summary_query(session, date_time)
    session_summary = run_chatgpt(
        query, 
        num_gen=1, 
        num_tokens_request=256, 
        model='chatgpt', 
        use_16k=False, 
        temperature=1.0, 
        wait_time=2
    )
    return session_summary


def main():
    args = parse_args()
    set_openai_key()

    data_file_path = Path(args.data_file)
    out_file_path = Path(args.out_file)

    with open(data_file_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    out_samples = {}
    if out_file_path.exists():
        with open(out_file_path, 'r', encoding='utf-8') as f:
            out_samples = {d['sample_id']: d for d in json.load(f)}

    for data in samples:
        sample_id = data['sample_id']
        summaries = []
        date_times = []
        context_ids = []

        # 获取已有结果，避免不必要的 copy
        output = out_samples.get(sample_id, {'sample_id': sample_id})

        # 提取 session 编号
        conversation = data.get('conversation', {})
        session_nums = [
            int(k.split('_')[-1]) 
            for k in conversation.keys() 
            if 'session' in k and 'date_time' not in k
        ]
        
        if not session_nums:
            continue
            
        min_sess, max_sess = min(session_nums), max(session_nums)

        for i in tqdm(range(min_sess, max_sess + 1), desc=f'Generating summaries for {sample_id}'):
            summary_key = f'session_{i}_summary'
            session_key = f'session_{i}'
            datetime_key = f'session_{i}_date_time'
            
            # 生成或读取 summary
            if summary_key not in output or args.overwrite:
                summary = get_session_summary(conversation[session_key], conversation[datetime_key])
                output[summary_key] = summary
            else:
                summary = output[summary_key]

            # 收集结果
            date_time = conversation[datetime_key]
            summaries.append(summary)
            date_times.append(date_time)
            context_ids.append(f'S{i}')

        print(f"Getting embeddings for {len(summaries)} summaries in sample {sample_id}...")
        embeddings = get_embeddings(args.retriever, summaries, 'context')
        assert embeddings.shape[0] == len(summaries), "Embeddings dimension mismatch!"
        
        database = {
            'embeddings': embeddings,
            'date_time': date_times,
            'dia_id': context_ids,
            'context': summaries
        }

        pkl_path = out_file_path.with_name(f"{out_file_path.stem}_{sample_id}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(database, f)

        out_samples[sample_id] = output
    
    # 全部跑完后，安全保存一份全量 JSON
    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(list(out_samples.values()), f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()