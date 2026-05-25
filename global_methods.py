import json
import time
import sys
import os
import re
import requests
import numpy as np
import openai
from tqdm import tqdm

_OPENAI_CLIENT = None

def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        base_url = os.environ.get('OPENAI_BASE_URL')
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        _OPENAI_CLIENT = openai.OpenAI(**client_kwargs)
        
    return _OPENAI_CLIENT


def _openai_retryable_errors():
    return (
        openai.APIError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError
    )


def get_openai_embedding(texts, model="text-embedding-ada-002"):
    texts = [text.replace("\n", " ") for text in texts]
    client = _get_openai_client()
    response = client.embeddings.create(input=texts, model=model)
    return np.array([item.embedding for item in response.data])


def get_remote_embedding(texts, url=None, api_key=None, model=None, batch_size=64, timeout=60):
    """Fetch embeddings from a remote embedding API.

    Args:
        texts: List of text strings to embed
        url: API endpoint URL (defaults to OPENAI_EMBEDDING_URL env var)
        api_key: API key (defaults to OPENAI_EMBEDDING_KEY env var)
        model: Model name (defaults to OPENAI_EMBEDDING_MODEL env var)
        batch_size: Number of texts per request
        timeout: Request timeout in seconds

    Returns:
        Numpy array of embeddings
    """
    url = url or os.environ.get('OPENAI_EMBEDDING_URL')
    api_key = api_key or os.environ.get('OPENAI_EMBEDDING_KEY')
    model = model or os.environ.get('OPENAI_EMBEDDING_MODEL')
    if not url:
        raise ValueError('No embedding URL provided (OPENAI_EMBEDDING_URL)')

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Getting embeddings"):
        chunk = texts[i:i+batch_size]
        payload = {"input": chunk}
        if model:
            payload["model"] = model

        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        embeddings.extend(_extract_embeddings_from_response(data))

    return np.array(embeddings, dtype=float)


def _extract_embeddings_from_response(data):
    """Extract embeddings from various API response formats.

    Handles multiple response formats:
    - OpenAI style: {'data': [{'embedding': [...]}, ...]}
    - Simple dict: {'embeddings': [[...], ...]}
    - Direct list: [[...], [...]]

    Args:
        data: Response data from embedding API

    Returns:
        List of embedding arrays

    Raises:
        ValueError: If response format is not recognized
    """
    # Direct list format
    if isinstance(data, list):
        return data

    # Dict with 'embeddings' key
    if isinstance(data, dict) and 'embeddings' in data:
        return data['embeddings']

    # OpenAI format with 'data' key
    if isinstance(data, dict) and 'data' in data:
        items = data['data']
        if not isinstance(items, list):
            raise ValueError('Expected data to be a list')

        result = []
        for item in items:
            if 'embedding' in item:
                result.append(item['embedding'])
            elif 'embeddings' in item:
                result.append(item['embeddings'])
            else:
                raise ValueError(f'Item missing both "embedding" and "embeddings" keys: {item}')
        return result

    raise ValueError(f'Unexpected response format: {type(data).__name__}')


def set_openai_key():
    global _OPENAI_CLIENT
    _OPENAI_CLIENT = None
    _get_openai_client()


def run_json_trials(query, num_gen=1, num_tokens_request=1000, 
                    model='davinci', use_16k=False, temperature=1.0, wait_time=1, examples=None, input=None):
    for counter in range(1, 11):
        try:
            if examples is not None and input is not None:
                output = run_chatgpt_with_examples(
                    query, examples, input, num_gen=num_gen, wait_time=wait_time,
                    num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature
                )
            else:
                output = run_chatgpt(
                    query, num_gen=num_gen, wait_time=wait_time,
                    num_tokens_request=num_tokens_request, temperature=temperature
                )
            clean_output = re.sub(r"^json)?|```$", "", output.strip(), flags=re.MULTILINE).strip()

            return json.loads(clean_output)
            
        except json.decoder.JSONDecodeError:
            print(f"Retrying to avoid JsonDecodeError, trial {counter} ...")
            print(output)
            time.sleep(1)
            
    print("Exiting after 10 trials")
    sys.exit(1)


def run_chatgpt(query, num_gen=1, num_tokens_request=1000, temperature=1.0, wait_time=1):

    client = _get_openai_client()
    retryable_errors = _openai_retryable_errors()
    
    chat_model = os.environ.get('OPENAI_CHAT_MODEL', 'gpt-3.5-turbo')
    messages = [{"role": "system", "content": query}]
    
    while True:
        try:
            completion = client.chat.completions.create(
                model=chat_model,
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                messages=messages
            )
            return completion.choices[0].message.content
            
        except retryable_errors as e:
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 60)
    

def run_chatgpt_with_examples(query, examples, input, num_gen=1, num_tokens_request=1000, 
                              use_16k=False, wait_time=1, temperature=1.0):

    client = _get_openai_client()
    retryable_errors = _openai_retryable_errors()
    
    messages = [{"role": "system", "content": query}]
    for inp, out in examples:
        messages.extend([
            {"role": "user", "content": inp},
            {"role": "system", "content": out}
        ])
    messages.append({"role": "user", "content": input})   
    
    model_name = "gpt-3.5-turbo-16k" if use_16k else "gpt-3.5-turbo"
    
    while True:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                messages=messages
            )
            return completion.choices[0].message.content
            
        except retryable_errors as e:
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 60)