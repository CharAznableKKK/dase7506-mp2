"""Validate a frozen candidate method on validation inputs, without references."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'

import argparse
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_info

from common import load_inputs
import selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    bank, arrays = load_inputs('validation')
    state = {'active': False, 'reads': set()}

    def audit(event, arguments):
        if not state['active']:
            return
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise AssertionError('Inference attempted network access')
        if event == 'open' and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(arguments[0])).resolve()
            if checkpoint not in path.parents:
                raise AssertionError(f'Inference read an undeclared file: {path}')
            state['reads'].add(path.name)

    sys.addaudithook(audit)

    def restricted_predict(rows, embeddings):
        state['active'] = True
        try:
            return selection.predict(rows, {'embeddings': embeddings}, checkpoint)
        finally:
            state['active'] = False

    tick = time.perf_counter()
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        try:
            full = restricted_predict(bank, arrays['embeddings'])
            assert len(full) == len(bank) and all(type(i) is int and 0 <= i <= 3 for i in full)
            subset = list(range(36, -1, -1))
            renamed = [dict(bank[i], id=f'unseen-{j}') for j, i in enumerate(subset)]
            actual = restricted_predict(renamed, arrays['embeddings'][subset])
            assert actual == [full[i] for i in subset]
            assert restricted_predict(renamed[:1], arrays['embeddings'][subset[:1]]) == actual[:1]
            assert restricted_predict([], np.empty((0, 4, 1024), dtype=np.float32)) == []
        finally:
            os.chdir(previous)
    pools = threadpool_info()
    assert pools and sum(pool['num_threads'] for pool in pools) <= 4, pools
    cpu = platform.processor()
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
    result = {'status': 'passed', 'checks': ['native integer indices', '798-question bank',
              '37-question reversed subset', 'unseen IDs', 'single question', 'empty bank',
              'empty working directory', 'checkpoint-only file reads', 'network audit hook',
              'loaded library thread count <= 4'], 'files_read_during_inference': sorted(state['reads']),
              'threadpools': pools, 'cpu': cpu.strip(), 'os': platform.platform(),
              'logical_cpu_count': os.cpu_count(), 'verification_seconds': time.perf_counter() - tick,
              'reference_cpu_verified': False, 'references_read': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
