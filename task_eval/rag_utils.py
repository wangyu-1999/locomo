import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import torch
from tqdm import tqdm

def save_eval(data_file, accs, key='exact_match'):

    
    if os.path.exists(data_file.replace('.json', '_scores.json')):
        data = json.load(open(data_file.replace('.json', '_scores.json')))
    else:
        data = json.load(open(data_file))

    assert len(data['qa']) == len(accs), (len(data['qa']), len(accs), accs)
    for i in range(0, len(data['qa'])):
        data['qa'][i][key] = accs[i]
    
    with open(data_file.replace('.json', '_scores.json'), 'w') as f:
        json.dump(data, f, indent=2)


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
        for i in tqdm(range(0, len(inputs), batch_size)):
            from global_methods import get_remote_embedding
            chunk = inputs[i:(i+batch_size)]
            emb = get_remote_embedding(chunk)
            emb_t = torch.tensor(emb, dtype=torch.float32)
            emb_t = torch.nn.functional.normalize(emb_t, dim=-1)
            all_embeddings.append(emb_t)

    return torch.cat(all_embeddings, dim=0).cpu().numpy()

def get_context_embeddings(retriever, data, context_tokenizer, context_encoder, captions=None):

    context_embeddings = []
    context_ids = []
    for i in tqdm(range(1,20), desc="Getting context encodings"):
        contexts = []
        if 'session_%s' % i in data:
            date_time_string = data['session_%s_date_time' % i]
            for dialog in data['session_%s' % i]:

                turn = ''
                # conv = conv + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n'
                try:
                    turn = dialog['speaker'] + ' said, \"' + dialog['compressed_text'] + '\"' + '\n'
                    # conv = conv + dialog['speaker'] + ': ' + dialog['compressed_text'] + '\n'
                except KeyError:
                    turn = dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n'
                    # conv = conv + dialog['speaker'] + ': ' + dialog['clean_text'] + '\n'
                if "img_file" in dialog and len(dialog["img_file"]) > 0:
                    turn += '[shares %s]\n' % dialog["blip_caption"]
                contexts.append('(' + date_time_string + ') ' + turn)

                context_ids.append(dialog["dia_id"])
            with torch.no_grad():
                from global_methods import get_remote_embedding
                emb = get_remote_embedding(contexts)
                emb_t = torch.tensor(emb, dtype=torch.float32)
                context_embeddings.append(torch.nn.functional.normalize(emb_t, dim=-1))


    # print(context_embeddings[0].shape[0])
    context_embeddings = torch.cat(context_embeddings, dim=0)
    # print(context_embeddings.shape[0])

    return context_ids, context_embeddings
