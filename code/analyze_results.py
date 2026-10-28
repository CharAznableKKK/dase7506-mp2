"""Paired validation comparisons and deterministic failure-case selection."""
import json
import math
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, read_jsonl
from selection import answer_groups
from training import labels_from_references

ROOT = Path(__file__).resolve().parents[1]


def main():
    study = json.loads((ROOT/'records/experiments/validation-study-v1.json').read_text())
    bank, _ = load_inputs('validation')
    y = labels_from_references(bank, read_jsonl(DATA/'gsm8k_validation.jsonl'))
    hard = np.array([len(set(a for a in group if a is not None)) >= 2 for group in answer_groups(bank)])
    rows = {row['experiment_id']: row for row in study['results']}
    predictions, correct = {}, {}
    for name, row in rows.items():
        directory = (ROOT/row['checkpoint']).parent
        predictions[name] = np.array(json.loads((directory/'predictions.json').read_text())['selected'])
        correct[name] = y[np.arange(len(y)), predictions[name]].astype(bool)
    comparisons = []
    pairs = [(study['winner'], 'baseline_majority'),
             ('embedding_pairwise_a0.1_none', 'embedding_pointwise_a0.1_none'),
             (study['winner'], 'embedding_pairwise_a0.1_none'),
             ('full_pairwise_a0.1_none', study['winner']),
             ('embedding_surface_pairwise_a0.1_answer_mean', study['winner']),
             (study['winner'], 'surface_pairwise_a0.1_none')]
    for better, reference in pairs:
        a, b = correct[better][hard], correct[reference][hard]
        wins = int((a & ~b).sum())
        losses = int((~a & b).sum())
        discordant = wins + losses
        probability = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses)+1)) / 2**discordant) if discordant else 1.
        comparisons.append({'method': better, 'reference': reference, 'n': int(hard.sum()),
                            'wins': wins, 'losses': losses, 'net_correct': wins-losses,
                            'difference_pp': 100*(wins-losses)/int(hard.sum()),
                            'descriptive_exact_mcnemar_p': probability,
                            'warning': 'Exploratory: validation used for model selection; no multiple-testing correction.'})
    examples = []
    winner = study['winner']
    for category, mask in [('improvement_over_vote', correct[winner] & ~correct['baseline_majority']),
                           ('regression_vs_vote', ~correct[winner] & correct['baseline_majority']),
                           ('both_wrong', ~correct[winner] & ~correct['baseline_majority'])]:
        for index in np.flatnonzero(mask & hard)[:2]:
            row = bank[int(index)]
            examples.append({'category': category, 'validation_index': int(index),
                             'question': row['question'], 'candidate_texts': [c['text'] for c in row['candidates']],
                             'parsed_answers': answer_groups([row])[0], 'correct_candidates': np.flatnonzero(y[index]).tolist(),
                             'selected': int(predictions[winner][index]),
                             'majority_selected': int(predictions['baseline_majority'][index])})
    out = ROOT/'records/analysis'
    (out/'paired-comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n', encoding='utf-8')
    (out/'validation-error-examples.json').write_text(json.dumps(examples, indent=2)+'\n', encoding='utf-8')
    for item in comparisons:
        print(json.dumps(item))
    print('Deterministic examples:', [(e['category'], e['validation_index']) for e in examples])


if __name__ == '__main__':
    main()
