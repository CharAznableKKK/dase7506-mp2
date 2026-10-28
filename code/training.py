"""Small ridge rankers; training inputs/references are explicitly train-only."""
import json
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from selection import answer_groups


def labels_from_references(bank, references):
    if len(bank) != len(references):
        raise ValueError('Reference length mismatch')
    labels = []
    for row, reference, answers in zip(bank, references, answer_groups(bank)):
        if row['id'] != reference['id'] or row['question'] != reference['question']:
            raise ValueError('Reference alignment mismatch')
        gold = parse_answer(reference['answer'])
        if gold is None:
            raise ValueError('Invalid reference answer')
        target = [float(answer is not None and answer == gold) for answer in answers]
        if not any(target):
            raise ValueError('No correct candidate')
        labels.append(target)
    return np.asarray(labels, dtype=np.float64)


def load_training(split='train', data_dir=DATA):
    if split != 'train':
        raise ValueError('Fitting is restricted to the supplied training split')
    bank, arrays = load_inputs(split, data_dir)
    references = read_jsonl(Path(data_dir) / 'gsm8k_train.jsonl')
    return bank, arrays, labels_from_references(bank, references)


def fit_ridge(x, labels, valid, alpha, objective):
    x = np.asarray(x, dtype=np.float64)
    if alpha <= 0 or objective not in ('pairwise', 'pointwise'):
        raise ValueError('Positive alpha and valid objective required')
    mean = x[valid].mean(axis=0)
    scale = x[valid].std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.)
    z = (x - mean) / scale
    bias = 0.
    if objective == 'pointwise':
        design = z[valid]
        target = labels[valid]
        bias = float(target.mean())
        target = target - bias
        normalizer = len(target)
    else:
        designs, targets = [], []
        groups = 0
        for values, y, mask in zip(z, labels, valid):
            positive = np.flatnonzero((y == 1) & mask)
            negative = np.flatnonzero((y == 0) & mask)
            if len(positive) == 0 or len(negative) == 0:
                continue
            pairs = len(positive) * len(negative)
            weight = 1 / np.sqrt(pairs)
            differences = (values[positive, None, :] - values[None, negative, :]).reshape(pairs, -1)
            designs.append(differences * weight)
            targets.append(np.ones(pairs) * weight)
            groups += 1
        if not designs:
            raise ValueError('No valid correct/incorrect pairs for ranking')
        design = np.concatenate(designs)
        target = np.concatenate(targets)
        normalizer = groups
    gram = design.T @ design / normalizer
    gram.flat[::len(gram) + 1] += alpha
    weights = np.linalg.solve(gram, design.T @ target / normalizer)
    return {'mean': mean, 'scale': scale, 'weights': weights, 'bias': np.array(bias)}


def save_model(directory, model, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if model is not None:
        np.savez(directory / 'weights.npz', **model)
    (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8', newline='\n')
