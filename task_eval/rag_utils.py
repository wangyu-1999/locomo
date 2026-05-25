import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from tqdm import tqdm
import numpy as np

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



def get_embeddings(inputs):
    """获取输入文本的远程嵌入表示，返回一个 numpy 数组"""
    all_embeddings = []
    batch_size = 24

    # tqdm 加上 desc 提示，提升可读性
    for i in tqdm(range(0, len(inputs), batch_size), desc="Getting embeddings"):
        chunk = inputs[i : i + batch_size]
        emb = get_remote_embedding(chunk)
        # L2 normalize embeddings
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        all_embeddings.append(emb)

    # 兼容处理 inputs 为空的情况
    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32)

    return np.concatenate(all_embeddings, axis=0)