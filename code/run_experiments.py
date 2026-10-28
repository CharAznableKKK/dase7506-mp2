"""Fixed validation-only experiment matrix, followed by official scoring."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np

from common import DATA, load_inputs, read_jsonl
from selection import answer_groups, features
from training import fit_ridge, labels_from_references, load_training, save_model

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
SEED = 20261024  # No stochastic fitting; retained for deterministic extensions.
ALPHAS = (0.1, 1., 10.)
FAMILIES = (('surface', 'pairwise'), ('embedding', 'pointwise'),
            ('embedding', 'pairwise'), ('embedding_surface', 'pairwise'), ('full', 'pairwise'))


def asset_record(directory):
    files = []
    for file in sorted(directory.iterdir()):
        if not file.is_file():
            continue
        raw = file.read_bytes()
        size = len(raw)
        if file.suffix == '.npz':
            with zipfile.ZipFile(file) as archive:
                size = sum(entry.file_size for entry in archive.infolist())
        files.append({'file': file.name, 'sha256': hashlib.sha256(raw).hexdigest(),
                      'file_bytes': len(raw), 'uncompressed_bytes': size})
    return {'files': files, 'uncompressed_bytes': sum(item['uncompressed_bytes'] for item in files)}


def evaluate_validation(checkpoint, output):
    tick = time.perf_counter()
    environment = dict(os.environ)
    environment['PYTHONHASHSEED'] = str(SEED)
    commands = [
        [sys.executable, 'predict.py', '--method', 'selection.py', '--checkpoint', str(checkpoint),
         '--split', 'validation', '--output', str(output / 'predictions.json')],
        [sys.executable, 'score_predictions.py', '--split', 'validation',
         '--predictions', str(output / 'predictions.json'), '--output', str(output / 'score.json')],
    ]
    logs = []
    for command in commands:
        result = subprocess.run(command, cwd=CODE, env=environment, capture_output=True, text=True, check=True)
        logs.append({'argv': command, 'stdout': result.stdout, 'stderr': result.stderr})
    elapsed = time.perf_counter() - tick
    (output / 'commands.json').write_text(json.dumps(logs, indent=2) + '\n', encoding='utf-8')
    return json.loads((output / 'score.json').read_text()), elapsed


def main():
    tick = time.perf_counter()
    experiment_root = CODE / 'runs' / 'validation-study-v1'
    if experiment_root.exists():
        raise FileExistsError('Refusing to overwrite existing experiment results')
    experiment_root.mkdir(parents=True)
    bank, arrays, y = load_training()
    vbank, varrays = load_inputs('validation')
    vy = labels_from_references(vbank, read_jsonl(DATA / 'gsm8k_validation.jsonl'))
    valid = np.array([[a is not None for a in group] for group in answer_groups(bank)])
    disagreement = np.array([len(set(a for a in group if a is not None)) >= 2 for group in answer_groups(vbank)])
    rows = []

    def record(name, model, config, fit_seconds, feature_seconds):
        output = experiment_root / name
        checkpoint = output / 'checkpoint'
        config = dict(config, seed=SEED)
        save_model(checkpoint, model, config)
        score, elapsed = evaluate_validation(checkpoint, output)
        predictions = json.loads((output / 'predictions.json').read_text())['selected']
        correct = vy[np.arange(len(vy)), predictions]
        assert int(correct.sum()) == score['correct']
        row = {'experiment_id': name, 'config': config, 'accuracy': score['accuracy'],
               'correct': score['correct'], 'n': score['n'],
               'disagreement_accuracy': float(100 * correct[disagreement].mean()),
               'disagreement_correct': int(correct[disagreement].sum()),
               'disagreement_n': int(disagreement.sum()),
               'fit_seconds': fit_seconds, 'feature_seconds': feature_seconds,
               'prediction_plus_scoring_seconds': elapsed, 'assets': asset_record(checkpoint),
               'checkpoint': checkpoint.relative_to(ROOT).as_posix()}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('experiment_id', 'accuracy', 'disagreement_accuracy', 'prediction_plus_scoring_seconds')}), flush=True)

    for mode in ('first', 'majority', 'shortest'):
        record('baseline_' + mode, None, {'mode': mode}, 0., 0.)
    for feature_set, objective in FAMILIES:
        begin = time.perf_counter()
        x = features(bank, arrays['embeddings'], feature_set)
        feature_seconds = time.perf_counter() - begin
        for alpha in ALPHAS:
            begin = time.perf_counter()
            model = fit_ridge(x, y, valid, alpha, objective)
            fit_seconds = time.perf_counter() - begin
            for pooling in ('none', 'answer_mean'):
                name = f'{feature_set}_{objective}_a{alpha:g}_{pooling}'
                config = {'mode': 'learned', 'feature_set': feature_set, 'objective': objective,
                          'alpha': alpha, 'pooling': pooling}
                record(name, model, config, fit_seconds, feature_seconds)
        del x
    rows.sort(key=lambda row: (-row['disagreement_correct'], -row['correct'], row['assets']['uncompressed_bytes'], row['experiment_id']))
    source_hashes = {name: hashlib.sha256((CODE/name).read_bytes()).hexdigest() for name in (
        'selection.py', 'training.py', 'run_experiments.py', 'common.py', 'predict.py', 'score_predictions.py')}
    result = {'seed': SEED, 'alphas': list(ALPHAS), 'families': FAMILIES,
              'selection_rule': 'disagreement correct DESC, full correct DESC, asset bytes ASC, experiment ID ASC',
              'source_sha256': source_hashes, 'total_search_seconds': time.perf_counter() - tick,
              'fit_count': 15, 'configuration_count': len(rows), 'winner': rows[0]['experiment_id'], 'results': rows}
    out = ROOT / 'records' / 'experiments'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'validation-study-v1.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('WINNER', result['winner'], 'SEARCH SECONDS', result['total_search_seconds'], flush=True)


if __name__ == '__main__':
    main()
