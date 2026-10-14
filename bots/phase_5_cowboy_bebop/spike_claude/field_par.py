"""Scratch: run bench.py vs a chunk of field opponents (2 parallel processes), appending lines to logs/<cand>_s<seed>.txt;
then print the weighted score (bench.weight) if the log covers the whole field.
usage: python field_par.py v2:Bot SEED opp1,opp2,...|all|agg [tables]"""
import sys, subprocess, re
from concurrent.futures import ThreadPoolExecutor
import bench
cand, seed, which = sys.argv[1], sys.argv[2], sys.argv[3]
tables = sys.argv[4] if len(sys.argv) > 4 else "64"
log = f"logs/{cand.replace(':', '_')}_s{seed}.txt"
opps = [] if which == "agg" else (bench.FIELD if which == "all" else which.split(","))
def run(o):
    out = subprocess.run([sys.executable, "bench.py", cand, o, "--seed", seed, "--tables", tables], capture_output=True, text=True).stdout.strip()
    with open(log, "a") as f:
        f.write(out + "\n")
    print(out, flush=True)
with ThreadPoolExecutor(int(sys.argv[5]) if len(sys.argv) > 5 else 1) as ex:
    list(ex.map(run, opps))
vals = {}
for line in open(log):
    m = re.search(r"vs\s+(\S+):\s+([+-][\d.]+) bb/100", line)
    if m: vals[m.group(1)] = float(m.group(2))
print("have", len(vals), "of", len(bench.FIELD))
if all(n in vals for n in bench.FIELD):
    w = sum(bench.weight(n, vals[n]) * vals[n] for n in bench.FIELD) / len(bench.FIELD)
    print(f"{cand} weighted score: {w:+8.2f}  seed {seed}")
