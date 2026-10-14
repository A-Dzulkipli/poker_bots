"""Scratch: weighted score from a per-opponent log, same formula as bench.py."""
import re, sys
RANK = {"windrunner_glm": 12, "invoker_gpt": 11, "meepo_claude": 10, "hadrian_marlowe_gpt": 9, "valka_onderra_claude": 8,
        "aggressor": 7, "bassander_lin_glm": 6, "alan_wake_claude": 5, "max_payne_gpt": 4, "jesse_faden_glm": 3,
        "jack_joyce_gemini": 2, "random": 1}
for path in sys.argv[1:]:
    res = {m.group(1): float(m.group(2)) for m in re.finditer(r"vs\s+(\S+):\s+([+-][\d.]+)", open(path).read())}
    w = sum((RANK[o] if v > 0 else 13 - RANK[o]) / 6.5 * v for o, v in res.items()) / 12
    print(f"{path}: {len(res)}/12 opponents, weighted {w:+.2f}" + ("" if len(res) == 12 else " (incomplete)"))
