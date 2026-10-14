"""Wait for development results, then run exactly one held-out field evaluation."""
import subprocess
import sys
import time
from pathlib import Path

version=sys.argv[1]
training=Path(version+'_field_s0.log')
while 'weighted score:' not in training.read_text():
    time.sleep(10)
output=Path(version+'_field_s1000.log')
if output.exists():
    raise RuntimeError('Held-out evaluation already exists; refusing to repeat it.')
print('Starting held-out evaluation for '+version,flush=True)
with output.open('w') as log:
    subprocess.run([sys.executable,'bench.py',version+':Bot','field','--seed','1000','--tables','32'],stdout=log,stderr=subprocess.STDOUT,check=True)
print(output.read_text(),flush=True)
