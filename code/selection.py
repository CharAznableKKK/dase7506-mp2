"""Offline selection using supplied inputs only; no reference-file access."""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re

import numpy as np

from common import parse_answer

FEATURE_SETS = ('surface', 'embedding', 'embedding_surface', 'full')


def answer_groups(bank):
    return [[parse_answer(c['text']) for c in row['candidates']] for row in bank]


def features(bank, embeddings, feature_set):
    if feature_set not in FEATURE_SETS:
        raise ValueError('Unknown feature set')
    if np.shape(embeddings) != (len(bank), 4, 1024):
        raise ValueError('Expected aligned embeddings of shape [N,4,1024]')
    if not np.isfinite(embeddings).all():
        raise ValueError('Nonfinite embeddings')
    if feature_set == 'embedding':
        return np.asarray(embeddings, dtype=np.float64)
    surface = np.zeros((len(bank), 4, 12), dtype=np.float64)
    consensus = np.zeros((len(bank), 4, 3), dtype=np.float64)
    for q, (row, answers) in enumerate(zip(bank, answer_groups(bank))):
        counts = Counter(a for a in answers if a is not None)
        for i, candidate in enumerate(row['candidates']):
            text = candidate['text']
            answer = answers[i]
            # Clip only a numeric feature, never the official parsed answer.
            value = Fraction(answer) if answer is not None else Fraction(0)
            log_abs = np.log1p(float(min(abs(value), Fraction(10**12))))
            surface[q, i] = [
                np.log1p(len(text)), np.log1p(len(text.split())),
                np.log1p(text.count('\n')), np.log1p(len(re.findall(r'\d+', text))),
                np.log1p(text.count('=')), np.log1p(text.count('\\')),
                np.log1p(text.count('**')), float('####' in text),
                float('\\boxed' in text), float(answer is not None),
                float(log_abs) * (1 if value >= 0 else -1),
                float(answer is not None and value.denominator == 1),
            ]
            if answer is not None:
                count = counts[answer]
                consensus[q, i] = [count / 4, float(count == max(counts.values())), float(count == 1)]
    if feature_set == 'surface':
        return surface
    parts = [np.asarray(embeddings, dtype=np.float64), surface]
    if feature_set == 'full':
        parts.append(consensus)
    return np.concatenate(parts, axis=-1)


def choose(bank, scores, pooling='none'):
    if np.shape(scores) != (len(bank), 4):
        raise ValueError('Expected one score per candidate')
    if pooling not in ('none', 'answer_mean', 'majority'):
        raise ValueError('Unknown pooling strategy')
    choices = []
    for answers, row_scores in zip(answer_groups(bank), scores):
        adjusted = np.full(4, -np.inf)
        for i, answer in enumerate(answers):
            if answer is None:
                continue
            peers = [j for j, a in enumerate(answers) if a == answer]
            if pooling == 'majority':
                adjusted[i] = len(peers)
            elif pooling == 'answer_mean':
                adjusted[i] = np.mean(row_scores[peers])
            else:
                adjusted[i] = row_scores[i]
        # Stable first-index tie-breaking, including an all-invalid external group.
        choices.append(int(np.argmax(adjusted)))
    return choices


def linear_scores(x, model):
    normalized = (np.asarray(x, dtype=np.float64) - model['mean']) / model['scale']
    return normalized @ model['weights'] + float(model['bias'])


def load_model(checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    if config.get('mode', 'learned') != 'learned':
        return None, config
    with np.load(checkpoint / 'weights.npz', allow_pickle=False) as data:
        model = {key: data[key].copy() for key in ('mean', 'scale', 'weights', 'bias')}
    return model, config


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    model, config = load_model(checkpoint) if checkpoint is not None else (None, {'mode': 'majority'})
    mode = config.get('mode', 'learned')
    if mode == 'first':
        return [0] * len(bank)
    if mode == 'majority':
        return choose(bank, np.zeros((len(bank), 4)), 'majority')
    if mode == 'shortest':
        return choose(bank, -np.array([[len(c['text']) for c in row['candidates']] for row in bank]), 'none')
    if mode != 'learned':
        raise ValueError('Unknown method mode')
    x = features(bank, arrays['embeddings'], config['feature_set'])
    return choose(bank, linear_scores(x, model), config['pooling'])
