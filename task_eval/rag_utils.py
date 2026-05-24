import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
from tqdm import tqdm

from global_methods import get_remote_embedding


def save_eval(data_file, accs, key='exact_match'):
    data_file_path = Path(data_file)
    out_file_path = data_file_path.with_name(f"{data_file_path.stem}_scores.json")
    
    target_path = out_file_path if out_file_path.exists() else data_file_path

    with open(target_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert len(data['qa']) == len(accs), f"Mismatch: {len(data['qa'])} vs {len(accs)}"
    
    # 使用 zip 替代毫无 Pythonic 可言的 range(len())
    for qa_item, acc in zip(data['qa'], accs):
        qa_item[key] = acc
    
    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def init_context_model(retriever):
    # remote embeddings are provided by an external API; no local model needed
    return None, None


def init_query_model(retriever):
    return None, None


def get_embeddings(retriever, inputs, mode='context'):    
    all_embeddings = []
    batch_size = 24
    
    with torch.no_grad():
        # tqdm 加上 desc 提示，提升可读性
        for i in tqdm(range(0, len(inputs), batch_size), desc="Getting embeddings"):
            chunk = inputs[i : i + batch_size]
            emb = get_remote_embedding(chunk)
            emb_t = torch.tensor(emb, dtype=torch.float32)
            emb_t = torch.nn.functional.normalize(emb_t, dim=-1)
            all_embeddings.append(emb_t)

    # 兼容处理 inputs 为空的情况
    if not all_embeddings:
        return torch.empty(0).numpy()
        
    return torch.cat(all_embeddings, dim=0).cpu().numpy()


def get_context_embeddings(retriever, data, context_tokenizer, context_encoder, captions=None):
    context_embeddings = []
    context_ids = []
    
    for i in tqdm(range(1, 20), desc="Getting context encodings"):
        session_key = f'session_{i}'
        
        # 避免过度缩进，提前判断 continue
        if session_key not in data:
            continue
            
        contexts = []
        date_time_string = data.get(f'{session_key}_date_time', '')
        
        for dialog in data[session_key]:
            context_ids.append(dialog["dia_id"])
            
            # 用 dict.get() 平替 try-except KeyError 控制流，性能更好且阅读顺畅
            text = dialog.get('compressed_text', dialog.get('clean_text', ''))
            turn = f"{dialog['speaker']} said, \"{text}\"\n"
            
            if dialog.get("img_file"):
                turn += f"[shares {dialog.get('blip_caption', '')}]\n"
                
            # 使用 f-string 替代老旧的字符串 + 和 %s 拼接
            contexts.append(f"({date_time_string}) {turn}")

        # 如果会话里没有任何对话，跳过 embedding 请求以防报错
        if contexts:
            with torch.no_grad():
                emb = get_remote_embedding(contexts)
                emb_t = torch.tensor(emb, dtype=torch.float32)
                context_embeddings.append(torch.nn.functional.normalize(emb_t, dim=-1))

    if context_embeddings:
        context_embeddings = torch.cat(context_embeddings, dim=0)
    else:
        # 防御性编程：避免抛出 RuntimeError: torch.cat(): expected a non-empty list
        context_embeddings = torch.empty(0)

    return context_ids, context_embeddings