"""Run held-out earlier-version checks once a finalist's field run is complete."""
import json
import subprocess
import sys
import time
from pathlib import Path

version, *opponents=sys.argv[1:]
field=Path(version+'_field_s1000.log')
while not field.exists() or 'weighted score:' not in field.read_text():
    time.sleep(10)
for opponent in opponents:
    if Path('selection.json').exists() and json.loads(Path('selection.json').read_text())['final'] != version:
        print('Final choice made; skipping remaining non-deployed comparisons.',flush=True)
        break
    path=Path(f'{version}_vs_{opponent}_s1000.log')
    if path.exists():
        raise RuntimeError('Comparison already exists: '+str(path))
    print(f'Starting {version} versus {opponent}',flush=True)
    with path.open('w') as output:
        subprocess.run([sys.executable,'bench.py',version+':Bot',opponent+':Bot','--seed','1000','--tables','32'],stdout=output,stderr=subprocess.STDOUT,check=True)
    print('Completed '+str(path),flush=True)
