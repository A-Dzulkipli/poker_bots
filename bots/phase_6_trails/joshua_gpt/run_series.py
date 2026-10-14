"""Run a declared sequence of standard bench.py commands in this directory."""
import subprocess
import sys
from pathlib import Path

candidate, seed, tables, *opponents = sys.argv[1:]
for opponent in opponents:
    log = f'{candidate}_vs_{opponent}_s{seed}.log'
    print(f'Starting {candidate} versus {opponent}, seed {seed}, {tables} tables', flush=True)
    with Path(log).open('w') as output:
        subprocess.run([sys.executable, 'bench.py', candidate + ':Bot', opponent + ':Bot', '--seed', seed, '--tables', tables], stdout=output, stderr=subprocess.STDOUT, check=True)
    print(Path(log).read_text(), flush=True)
