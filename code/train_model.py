"""Reproduce one declared configuration using training data only."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'

import argparse
import json
from pathlib import Path
import time

import numpy as np

from selection import answer_groups, features
from training import fit_ridge, load_training, save_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing model directory')
    tick = time.perf_counter()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    model = None
    if config.get('mode', 'learned') == 'learned':
        bank, arrays, labels = load_training()
        valid = np.array([[a is not None for a in group] for group in answer_groups(bank)])
        x = features(bank, arrays['embeddings'], config['feature_set'])
        model = fit_ridge(x, labels, valid, config['alpha'], config['objective'])
    save_model(args.output, model, config)
    record = {'config': config, 'training_seconds': time.perf_counter() - tick,
              'training_split': 'train', 'validation_used_for_fit': False, 'test_read': False}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
