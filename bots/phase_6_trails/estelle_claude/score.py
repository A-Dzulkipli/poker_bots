"""Scratch: weighted score from a logs/*.log file (same formula as bench.py)."""
import re, sys
ORDER = ["random", "jack_joyce_gemini", "jesse_faden_glm", "bassander_lin_glm", "max_payne_gpt", "alan_wake_claude",
         "aggressor", "valka_onderra_claude", "vicious_glm", "hadrian_marlowe_gpt", "meepo_claude", "invoker_gpt",
         "windrunner_glm", "faye_gpt", "spike_claude", "megalodon_glm", "leviathan_claude", "kraken_gpt"]
res = {}
for line in open(sys.argv[1]):
    m = re.search(r"vs\s+(\S+):\s+([-+\d.]+) bb/100\s+\(SE ([\d.]+)", line)
    if m: res[m.group(1)] = (float(m.group(2)), float(m.group(3)))
for name in reversed(ORDER):
    if name in res: print(f"| {name} | {res[name][0]:+.1f} | {res[name][1]:.1f} |")
if len(res) == 18:
    w = sum((ORDER.index(n) + 1 if v > 0 else 18 - ORDER.index(n)) / 9.5 * v for n, (v, _) in res.items()) / 18
    print(f"weighted score {w:+.2f}  plain mean non-random {sum(v for n,(v,_) in res.items() if n!='random')/17:+.2f}")
else:
    print(f"{len(res)}/18 opponents")
