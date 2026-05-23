import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter
from bert_score import score
from nltk.stem import PorterStemmer

ps = PorterStemmer()
LENGTH_THRESHOLD = 5

_PUNCTUATION_SET = set(string.punctuation)
_ARTICLES_REGEX = regex.compile(r'\b(a|an|the|and)\b')
_ROUGE_EVALUATOR = None


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        self._regexp = regex.compile(
            f'({self.ALPHA_NUM})|({self.NON_WS})',
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = self._regexp.finditer(text)
        if uncased:
            return [m.group().lower() for m in matches]
        return [m.group() for m in matches]


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []
    for doc in ctxs:
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=None) -> bool:
    """Check if a document contains an answer string."""
    if tokenizer is None:
        tokenizer = SimpleTokenizer()
        
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def _get_reference_answer(line):
    return line.get('answer', line.get('adversarial_answer'))


def normalize_answer(s):
    s = s.lower().replace(',', "")
    s = ''.join(ch for ch in s if ch not in _PUNCTUATION_SET)
    s = _ARTICLES_REGEX.sub(' ', s)
    return ' '.join(s.split())


def exact_match_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    return set(prediction.split()) == set(ground_truth.split())


def bert_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
    return max(0, F1[0].item())


def ems(prediction, ground_truths):
    return max(exact_match_score(prediction, gt) for gt in ground_truths)


def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    return np.mean([max(f1_score(pred, gt) for pred in predictions) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
    global _ROUGE_EVALUATOR
    if _ROUGE_EVALUATOR is None:
        from rouge import Rouge
        _ROUGE_EVALUATOR = Rouge()
        
    prediction = ' '.join(ps.stem(w) for w in normalize_answer(prediction).split())
    ground_truth = ' '.join(ps.stem(w) for w in normalize_answer(ground_truth).split())
    
    try:
        scores = _ROUGE_EVALUATOR.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-1"]["f"]


def rl(prediction, ground_truths):
    return max(rougel_score(prediction, gt) for gt in ground_truths)


def eval_recall(infile):
    tokenizer = SimpleTokenizer()
    has_answer_count = 0
    answer_lengths = []
    
    with open(infile, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        lines = f.readlines()

    for line_str in lines:
        line = json.loads(line_str)
        answer = _get_reference_answer(line)
        output = ' || '.join(line['output'])

        if has_answer(answer, output, tokenizer):
            has_answer_count += 1

        answer_lengths.append(len(output.split()))

    recall = round(has_answer_count / len(lines), 4) if lines else 0.0
    lens = round(np.mean(answer_lengths), 4) if answer_lengths else 0.0
    return recall, lens


def eval_question_answering(qas, eval_key='prediction', metric='f1'):
    all_ems = []
    all_recall = []
    
    for i, line in enumerate(qas):
        if isinstance(line[eval_key], list):
            answer = _get_reference_answer(line)
        else:
            answer = str(_get_reference_answer(line))
            
        if line['category'] == 3:
            answer = answer.split(';')[0].strip()
        
        output = line[eval_key]
        category = line['category']
        
        if category in [2, 3, 4]:
            all_ems.append(f1_score(output, answer))
        elif category in [1]:
            all_ems.append(f1(output, answer))
        elif category in [5]:
            output_lower = output.lower()
            if 'no information available' in output_lower or 'not mentioned' in output_lower:
                all_ems.append(1.0)
            else:
                all_ems.append(0.0)
        else:
            print(line)
            raise ValueError(f"Unknown category: {category}")
        if f"{eval_key}_context" in line and len(line['evidence']) > 0:
            context = line[f"{eval_key}_context"]
            evidence = line["evidence"]
            
            if context[0].startswith('S'):
                sessions = {e[1:] for e in context}
                recall_acc = sum(1 for ev in evidence if ev.split(':')[0][1:] in sessions) / len(evidence)
            else:
                context_set = set(context)
                recall_acc = sum(1 for ev in evidence if ev in context_set) / len(evidence)
            all_recall.append(recall_acc)
        else:
            all_recall.append(1.0)

    print(f"{len(qas)} QA samples evaluated; {len(all_ems)} accuracy values")
    lens = 0.0
    return all_ems, lens, all_recall


def eval_fact_checking(infile):
    tokenizer = SimpleTokenizer()
    exact_match_count = 0
    answer_lengths = []
    
    with open(infile, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        lines = f.readlines()

    for line_str in lines:
        line = json.loads(line_str)
        answer = _get_reference_answer(line)
        output = line['output'][0]

        if answer == ["refutes"]:
            answer = ["refutes", "no", "false"]
        elif answer == ["supports"]:
            answer = ["supports", "yes", "true"]

        if has_answer(answer, output, tokenizer):
            exact_match_count += 1
        
        answer_lengths.append(len(output.split()))

    em = round(exact_match_count / len(lines), 4) if lines else 0.0
    lens = round(np.mean(answer_lengths), 4) if answer_lengths else 0.0
    return em, lens


def eval_dialogue_system(infile):
    f1_scores = []
    rl_scores = []
    answer_lengths = []
    
    with open(infile, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        lines = f.readlines()

    for line_str in lines:
        line = json.loads(line_str)
        answer = _get_reference_answer(line)
        output = line['output'][0]

        f1_scores.append(f1(output, answer))
        rl_scores.append(rl(output, answer))
        answer_lengths.append(len(output.split()))

    F1 = round(np.mean(f1_scores), 4) if f1_scores else 0.0
    RL = round(np.mean(rl_scores), 4) if rl_scores else 0.0
    lens = round(np.mean(answer_lengths), 4) if answer_lengths else 0.0

    return F1, RL, lens