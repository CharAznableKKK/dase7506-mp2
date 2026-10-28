"""Reproducible descriptive analysis, restricted to train and validation."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, read_jsonl
from selection import answer_groups, choose
from training import labels_from_references


def main():
    results = {}
    for split in ('train', 'validation'):
        bank, arrays = load_inputs(split)
        y = labels_from_references(bank, read_jsonl(DATA / f'gsm8k_{split}.jsonl'))
        groups = answer_groups(bank)
        distinct = np.array([len(set(a for a in row if a is not None)) for row in groups])
        valid = np.array([[a is not None for a in row] for row in groups])
        lengths = np.array([[len(c['text']) for c in row['candidates']] for row in bank])
        choices = choose(bank, np.zeros((len(bank), 4)), 'majority')
        correct = y[np.arange(len(bank)), choices]
        norm = np.linalg.norm(arrays['embeddings'], axis=-1)
        results[split] = {
            'questions': len(bank), 'correct_count_histogram': dict(Counter(map(int, y.sum(1)))),
            'distinct_answer_histogram': dict(Counter(map(int, distinct))),
            'parseable_candidates': int(valid.sum()), 'candidate_count': int(valid.size),
            'first_correct': int(y[:, 0].sum()), 'majority_correct': int(correct.sum()),
            'disagreement_questions': int((distinct >= 2).sum()),
            'majority_disagreement_correct': int(correct[distinct >= 2].sum()),
            'mean_chars_correct': float(lengths[y == 1].mean()),
            'mean_chars_wrong': float(lengths[y == 0].mean()),
            'chars_quantiles': np.quantile(lengths, [0, .25, .5, .75, 1]).tolist(),
            'embedding_shape': list(arrays['embeddings'].shape),
            'embedding_dtype': str(arrays['embeddings'].dtype),
            'embedding_norm_min_mean_max': [float(norm.min()), float(norm.mean()), float(norm.max())],
        }
    output = Path(__file__).resolve().parents[1] / 'records' / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'data-summary.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
