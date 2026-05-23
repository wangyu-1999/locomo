import openai
import numpy as np
import json
import time
import sys
import os
import requests

import google.generativeai as genai
from anthropic import Anthropic


_OPENAI_CLIENT = None


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = os.environ['OPENAI_API_KEY']
        base_url = os.environ.get('OPENAI_BASE_URL')
        if base_url:
            _OPENAI_CLIENT = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            _OPENAI_CLIENT = openai.OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _openai_retryable_errors():
    error_names = ['APIError', 'APIConnectionError', 'RateLimitError', 'APITimeoutError', 'InternalServerError']
    errors = []
    for name in error_names:
        if hasattr(openai, name):
            errors.append(getattr(openai, name))
    return tuple(errors) if errors else (Exception,)


def get_openai_embedding(texts, model="text-embedding-ada-002"):
   texts = [text.replace("\n", " ") for text in texts]
   client = _get_openai_client()
   response = client.embeddings.create(input=texts, model=model)
   return np.array([item.embedding for item in response.data])


def get_remote_embedding(texts, url=None, api_key=None, model=None, batch_size=64, timeout=60):
    """Fetch embeddings from a remote embedding API.

    Expects env vars (if arguments not provided):
      OPENAI_EMBEDDING_URL, OPENAI_EMBEDDING_KEY, OPENAI_EMBEDDING_MODEL

    The helper will POST JSON {"model": model, "input": [texts...]}
    and accept responses in either OpenAI-style (`data` with `embedding`) or
    a simple `embeddings` list. Returns a numpy array of shape (N, D).
    """
    url = url or os.environ.get('OPENAI_EMBEDDING_URL')
    api_key = api_key or os.environ.get('OPENAI_EMBEDDING_KEY')
    model = model or os.environ.get('OPENAI_EMBEDDING_MODEL')
    if url is None:
        raise ValueError('No embedding URL provided (OPENAI_EMBEDDING_URL)')

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    embeddings = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        payload = {"input": chunk}
        if model:
            payload["model"] = model
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # try OpenAI style
        if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
            for item in data['data']:
                if 'embedding' in item:
                    embeddings.append(item['embedding'])
                elif 'embeddings' in item:
                    embeddings.append(item['embeddings'])
                else:
                    raise ValueError('Unexpected response item for embedding')
        elif isinstance(data, dict) and 'embeddings' in data:
            embeddings.extend(data['embeddings'])
        elif isinstance(data, list):
            embeddings.extend(data)
        else:
            raise ValueError('Unexpected response format from embedding service')

    return np.array(embeddings, dtype=float)

def set_anthropic_key():
    pass

def set_gemini_key():

    # Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def set_openai_key():
    openai.api_key = os.environ['OPENAI_API_KEY']
    base_url = os.environ.get('OPENAI_BASE_URL')
    if base_url:
        if hasattr(openai, 'base_url'):
            openai.base_url = base_url
        elif hasattr(openai, 'api_base'):
            openai.api_base = base_url
    _get_openai_client()


def run_json_trials(query, num_gen=1, num_tokens_request=1000, 
                model='davinci', use_16k=False, temperature=1.0, wait_time=1, examples=None, input=None):

    run_loop = True
    counter = 0
    while run_loop:
        try:
            if examples is not None and input is not None:
                output = run_chatgpt_with_examples(query, examples, input, num_gen=num_gen, wait_time=wait_time,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature).strip()
            else:
                output = run_chatgpt(query, num_gen=num_gen, wait_time=wait_time, model=model,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature)
            output = output.replace('json', '') # this frequently happens
            facts = json.loads(output.strip())
            run_loop = False
        except json.decoder.JSONDecodeError:
            counter += 1
            time.sleep(1)
            print("Retrying to avoid JsonDecodeError, trial %s ..." % counter)
            print(output)
            if counter == 10:
                print("Exiting after 10 trials")
                sys.exit()
            continue
    return facts


def run_claude(query, max_new_tokens, model_name):

    if model_name == 'claude-sonnet':
        model_name = "claude-3-sonnet-20240229"
    elif model_name == 'claude-haiku':
        model_name = "claude-3-haiku-20240307"

    client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    # print(query)
    message = client.messages.create(
        max_tokens=max_new_tokens,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model=model_name,
    )
    print(message.content)
    return message.content[0].text


def run_gemini(model, content: str, max_tokens: int = 0):

    try:
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        return None


def run_chatgpt(query, num_gen=1, num_tokens_request=1000, 
                model='chatgpt', use_16k=False, temperature=1.0, wait_time=1):

    completion = None
    client = _get_openai_client()
    retryable_errors = _openai_retryable_errors()
    while completion is None:
        wait_time = wait_time * 2
        try:
            chat_model = os.environ.get('OPENAI_CHAT_MODEL', 'gpt-3.5-turbo')
            messages = [
                    {"role": "system", "content": query}
                ]
            completion = client.chat.completions.create(
                model=chat_model,
                temperature = temperature,
                max_tokens = num_tokens_request,
                n=num_gen,
                messages = messages
            )
        except retryable_errors as e:
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
    
    return completion.choices[0].message.content
    

def run_chatgpt_with_examples(query, examples, input, num_gen=1, num_tokens_request=1000, use_16k=False, wait_time = 1, temperature=1.0):

    completion = None
    client = _get_openai_client()
    retryable_errors = _openai_retryable_errors()
    
    messages = [
        {"role": "system", "content": query}
    ]
    for inp, out in examples:
        messages.append(
            {"role": "user", "content": inp}
        )
        messages.append(
            {"role": "system", "content": out}
        )
    messages.append(
        {"role": "user", "content": input}
    )   
    
    while completion is None:
        wait_time = wait_time * 2
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo" if not use_16k else "gpt-3.5-turbo-16k",
                temperature = temperature,
                max_tokens = num_tokens_request,
                n=num_gen,
                messages = messages
            )
        except retryable_errors as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
    
    return completion.choices[0].message.content
