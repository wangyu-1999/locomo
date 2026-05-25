import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import random
import os
import json
from tqdm import tqdm
from global_methods import run_chatgpt
from task_eval.rag_utils import get_embeddings
import tiktoken
import numpy as np

MAX_LENGTH = {'gpt-3.5-turbo': 320000}
PER_QA_TOKEN_BUDGET = 50

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. 
Write the answers in the form of a json dictionary where each entry contains the question number as "key" and the short answer as "value". 
Use single-quote characters for named entities and double-quote characters for enclosing json elements. Answer with exact words from the conversations whenever possible.

"""

CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"


def process_ouput(text):
    single_quote_count = text.count("'")
    double_quote_count = text.count('"')
    if single_quote_count > double_quote_count:
        text = text.replace('"', "").replace("'", '"')
    return json.loads(text)


# --- Helpers for prepare_for_rag ---

def _load_pickle_database(pkl_path):
    """Load database from pickle file if it exists.

    Args:
        pkl_path: Path to pickle file

    Returns:
        Loaded database dict, or None if file doesn't exist
    """
    if not pkl_path.exists():
        return None

    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def _prepare_dialog_database(args, data, pkl_path):
    """Prepare dialog database with embeddings. Load from cache if it exists.

    Args:
        args: Command-line arguments
        data: Sample data
        pkl_path: Path to pickle file for caching

    Returns:
        Database dict with embeddings, date_time, dia_id, context
    """
    # Try to load from cache first
    cached_db = _load_pickle_database(pkl_path)
    if cached_db is not None:
        return cached_db

    # Generate new database
    dialogs = []
    date_times = []
    context_ids = []
    conversation = data['conversation']
    session_nums = [
        int(k.split('_')[-1])
        for k in conversation
        if 'session' in k and 'date_time' not in k
    ]

    for i in range(min(session_nums), max(session_nums) + 1):
        date_time = conversation[f'session_{i}_date_time']
        for dialog in conversation[f'session_{i}']:
            context_ids.append(dialog['dia_id'])
            date_times.append(date_time)

            dialog_text = f"{dialog['speaker']} said, \"{dialog['text']}\""
            if 'blip_caption' in dialog:
                dialog_text += f" and shared {dialog['blip_caption']}"
            dialogs.append(dialog_text)

    print(f"Getting embeddings for {len(dialogs)} dialogs")
    embeddings = get_embeddings(dialogs)

    database = {
        'embeddings': embeddings,
        'date_time': date_times,
        'dia_id': context_ids,
        'context': dialogs
    }

    # Save to cache
    with open(pkl_path, 'wb') as f:
        pickle.dump(database, f)

    return database

def prepare_for_rag(args, data):
    """Prepare RAG context by loading or generating database.

    Args:
        args: Command-line arguments with rag_mode
        data: Sample data

    Returns:
        Tuple of (database, question_embeddings)
    """
    dataset_prefix = Path(args.data_file).stem
    emb_dir = Path(args.emb_dir)
    sample_id = data['sample_id']

    if args.rag_mode == "summary":
        pkl_path = emb_dir / f"{dataset_prefix}_session_summary_{sample_id}.pkl"
        database = _load_pickle_database(pkl_path)
        if database is None:
            raise FileNotFoundError(f"Summary database not found: {pkl_path}")

    elif args.rag_mode == 'dialog':
        pkl_path = emb_dir / f"{dataset_prefix}_dialog_{sample_id}.pkl"
        database = _prepare_dialog_database(args, data, pkl_path)

    elif args.rag_mode == 'observation':
        pkl_path = emb_dir / f"{dataset_prefix}_observation_{sample_id}.pkl"
        database = _load_pickle_database(pkl_path)
        if database is None:
            raise FileNotFoundError(f"Observation database not found: {pkl_path}")

    else:
        raise ValueError(f"Unsupported rag_mode: {args.rag_mode}")
    
    qa_list = data.get('qa', [])
    print(f"Getting embeddings for {len(qa_list)} questions")
    question_embeddings = get_embeddings([q['question'] for q in qa_list])

    return database, question_embeddings


def get_cat_5_answer(model_prediction, answer_key):
    model_prediction = model_prediction.strip().lower()
    if len(model_prediction) == 1:
        return answer_key['a'] if 'a' in model_prediction else answer_key['b']
    elif len(model_prediction) == 3:
        return answer_key['a'] if '(a)' in model_prediction else answer_key['b']
    else:
        return model_prediction


def get_rag_context(context_database, query_vector, args):
    output = np.dot(query_vector, context_database['embeddings'].T)
    sorted_outputs = np.argsort(output)[::-1]
    
    top_k_idxs = sorted_outputs[:args.top_k]
    sorted_context = [context_database['context'][idx] for idx in top_k_idxs]
    
    sorted_context_ids = []
    for idx in top_k_idxs:
        context_id = context_database['dia_id'][idx]
        if isinstance(context_id, str) and ',' in context_id:
            context_id = [s.strip() for s in context_id.split(',')]
            
        if isinstance(context_id, list):
            sorted_context_ids.extend(context_id)
        else:
            sorted_context_ids.append(context_id)

    sorted_date_times = [context_database['date_time'][idx] for idx in top_k_idxs]
    
    join_str = '\n' if args.rag_mode in ('dialog', 'observation') else '\n\n'
    query_context = join_str.join(
        f"{date_time}: {context}" 
        for date_time, context in zip(sorted_date_times, sorted_context)
    )

    return query_context, sorted_context_ids


def get_input_context(data, num_question_tokens, encoding, args):
    query_conv = ''
    stop = False
    session_nums = [int(k.split('_')[-1]) for k in data if 'session' in k and 'date_time' not in k]
    
    if not session_nums:
        return query_conv

    budget_limit = MAX_LENGTH.get(args.model, 4096) - (PER_QA_TOKEN_BUDGET * args.batch_size)

    for i in range(min(session_nums), max(session_nums) + 1):
        session_key = f'session_{i}'
        if session_key in data:
            query_conv += "\n\n"
            for dialog in reversed(data[session_key]):
                turn = f"{dialog['speaker']} said, \"{dialog['text']}\"\n"
                if "blip_caption" in dialog:
                    turn += f" and shared {dialog['blip_caption']}.\n"
                else:
                    turn += "\n"
        
                num_tokens = len(encoding.encode(f"DATE: {data[f'{session_key}_date_time']}\nCONVERSATION:\n{turn}"))
                
                if (num_tokens + len(encoding.encode(query_conv)) + num_question_tokens) < budget_limit:
                    query_conv = turn + query_conv
                else:
                    stop = True
                    break
                    
            query_conv = f"DATE: {data[f'{session_key}_date_time']}\nCONVERSATION:\n{query_conv}"
            
        if stop:
            break
        
    return query_conv


# --- Helpers for get_gpt_answers ---

def _get_tokenizer(model_name):
    use_16k = any(k in model_name for k in ['16k', '12k', '8k', '4k'])
    model_for_tokenizer = 'gpt-3.5-turbo-16k' if use_16k else model_name
    try:
        encoding = tiktoken.encoding_for_model(model_for_tokenizer)
    except KeyError:
        encoding = tiktoken.get_encoding('cl100k_base')
    return encoding, use_16k

def _prepare_batch_questions(in_data, out_data, prediction_key, args, batch_start_idx):
    questions = []
    include_idxs = []
    cat_5_idxs = []
    cat_5_answers = []
    
    for i in range(batch_start_idx, batch_start_idx + args.batch_size):
        if i >= len(in_data['qa']):
            break

        qa = in_data['qa'][i]
        if prediction_key not in out_data['qa'][i] or args.overwrite:
            include_idxs.append(i)
        else:
            continue

        if qa['category'] == 2:
            questions.append(f"{qa['question']} Use DATE of CONVERSATION to answer with an approximate date.")
        elif qa['category'] == 5:
            adversarial_answer = qa.get('answer', qa.get('adversarial_answer'))
            question_template = f"{qa['question']} Select the correct answer: (a) {{}} (b) {{}}. "
            
            if random.random() < 0.5:
                question = question_template.format('Not mentioned in the conversation', adversarial_answer)
                answer = {'a': 'Not mentioned in the conversation', 'b': adversarial_answer}
            else:
                question = question_template.format(adversarial_answer, 'Not mentioned in the conversation')
                answer = {'b': 'Not mentioned in the conversation', 'a': adversarial_answer}

            cat_5_idxs.append(len(questions))
            questions.append(question)
            cat_5_answers.append(answer)
        else:
            questions.append(qa['question'])

    return questions, include_idxs, cat_5_idxs, cat_5_answers

def _build_query_context(args, questions, encoding, in_data, start_prompt, start_tokens, context_database, query_vectors, include_idxs):
    context_ids = []
    if args.use_rag:
        query_conv, context_ids = get_rag_context(context_database, query_vectors[include_idxs][0], args)
        question_prompt = ""
    else:
        question_prompt = QA_PROMPT_BATCH + "\n".join(f"{k}: {q}" for k, q in enumerate(questions))
        num_question_tokens = len(encoding.encode(question_prompt))
        query_conv = get_input_context(in_data['conversation'], num_question_tokens + start_tokens, encoding, args)
        query_conv = start_prompt + query_conv
    return query_conv, question_prompt, context_ids

def _log_single_prediction(args, out_data, idx, prediction_key):
    preds_path = getattr(args, 'preds_file', None)
    if not preds_path:
        return
    try:
        preds_dir = os.path.dirname(preds_path)
        if preds_dir:
            os.makedirs(preds_dir, exist_ok=True)
            
        record = {
            'sample_id': out_data.get('sample_id'),
            'qa_index': idx,
            'question': out_data['qa'][idx].get('question'),
            'prediction': out_data['qa'][idx].get(prediction_key),
        }
        with open(preds_path, 'a', encoding='utf-8') as pf:
            pf.write(json.dumps(record, ensure_ascii=False) + "\n")
            pf.flush()
    except Exception:
        pass

def _run_single_batch(args, query_conv, questions, cat_5_idxs, cat_5_answers, out_data, include_idxs, prediction_key, context_ids):
    query = query_conv + '\n\n' + (QA_PROMPT_CAT_5 if cat_5_idxs else QA_PROMPT).format(questions[0])
    answer = run_chatgpt(
        query, num_gen=1, num_tokens_request=32, 
        temperature=0, wait_time=2
    )
    
    if cat_5_idxs:
        answer = get_cat_5_answer(answer, cat_5_answers[0])

    idx = include_idxs[0]
    out_data['qa'][idx][prediction_key] = answer.strip()
    if args.use_rag:
        out_data['qa'][idx][f"{prediction_key}_context"] = context_ids

    _log_single_prediction(args, out_data, idx, prediction_key)

def _parse_multi_batch_answers(answer, include_idxs, cat_5_idxs, cat_5_answers, out_data, prediction_key):
    for k, idx in enumerate(include_idxs):
        is_cat_5 = k in cat_5_idxs
        cat_5_ans_idx = cat_5_idxs.index(k) if is_cat_5 else -1
        
        try:
            answers = process_ouput(answer.strip())
            if is_cat_5:
                out_data['qa'][idx][prediction_key] = get_cat_5_answer(answers[str(k)], cat_5_answers[cat_5_ans_idx])
            else:
                try:
                    out_data['qa'][idx][prediction_key] = str(answers[str(k)]).replace('(a)', '').replace('(b)', '').strip()
                except Exception:
                    out_data['qa'][idx][prediction_key] = ', '.join(str(n) for n in answers[str(k)].values())
        except Exception:
            try:
                answers = json.loads(answer.strip())
                if is_cat_5:
                    out_data['qa'][idx][prediction_key] = get_cat_5_answer(answers[k], cat_5_answers[cat_5_ans_idx])
                else:
                    out_data['qa'][idx][prediction_key] = answers[k].replace('(a)', '').replace('(b)', '').strip()
            except Exception:
                if is_cat_5:
                    out_data['qa'][idx][prediction_key] = get_cat_5_answer(answer.strip(), cat_5_answers[cat_5_ans_idx])
                else:
                    out_data['qa'][idx][prediction_key] = json.loads(answer.strip().replace('(a)', '').replace('(b)', '').split('\n')[k])[0]


def _run_multi_batch(args, query_conv, question_prompt, include_idxs, cat_5_idxs, cat_5_answers, out_data, prediction_key):
    query = f"{query_conv}\n{question_prompt}"
    trials = 0
    answer = ""
    while trials < 3:
        try:
            trials += 1
            print(f"Trial {trials}/3")
            answer = run_chatgpt(
                query, num_gen=1, num_tokens_request=args.batch_size * PER_QA_TOKEN_BUDGET, 
                temperature=0, wait_time=2
            )
            answer = answer.replace('\\"', "'").replace('json', '').replace('`', '').strip().replace("\\'", "")
            
            _ = process_ouput(answer.strip())
            break
        except Exception as e:
            print(f'Error at trial {trials}/3: {e}')
            if trials == 3:
                raise ValueError(f"Failed to process output after 3 retries: {e}")
                
    _parse_multi_batch_answers(answer, include_idxs, cat_5_idxs, cat_5_answers, out_data, prediction_key)


def get_gpt_answers(in_data, out_data, prediction_key, args):
    encoding, use_16k = _get_tokenizer(args.model)

    speakers_names = list({d['speaker'] for d in in_data['conversation']['session_1']})
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    start_tokens = len(encoding.encode(start_prompt))

    if args.use_rag:
        context_database, query_vectors = prepare_for_rag(args, in_data)
    else:
        context_database, query_vectors = None, None

    for batch_start_idx in tqdm(range(0, len(in_data['qa']), args.batch_size), desc='Generating answers'):

        questions, include_idxs, cat_5_idxs, cat_5_answers = _prepare_batch_questions(
            in_data, out_data, prediction_key, args, batch_start_idx
        )

        if not questions:
            continue

        query_conv, question_prompt, context_ids = _build_query_context(
            args, questions, encoding, in_data, start_prompt, start_tokens, context_database, query_vectors, include_idxs
        )

        if args.batch_size == 1:
            _run_single_batch(
                args, query_conv, questions, cat_5_idxs, cat_5_answers, 
                out_data, include_idxs, prediction_key, context_ids
            )
        else:
            _run_multi_batch(
                args, query_conv, question_prompt, include_idxs, cat_5_idxs, cat_5_answers, 
                out_data, prediction_key
            )