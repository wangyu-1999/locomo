import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np

from global_methods import get_remote_embedding


def save_eval(data_file, accs, key='exact_match'):
    data_file_path = Path(data_file)
    out_file_path = data_file_path.with_name(f"{data_file_path.stem}_scores.json")

    target_path = out_file_path if out_file_path.exists() else data_file_path

    with open(target_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 使用 zip 替代毫无 Pythonic 可言的 range(len())
    for qa_item, acc in zip(data['qa'], accs):
        qa_item[key] = acc

    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def get_embeddings(inputs):
    """Get embeddings from remote API with L2 normalization.

    Args:
        inputs: List of text strings to embed

    Returns:
        Numpy array of embeddings (normalized to unit length)
    """
    print(f"Getting embeddings for {len(inputs)} items")
    emb = get_remote_embedding(inputs, batch_size=24)

    if emb.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    # L2 normalize embeddings
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb