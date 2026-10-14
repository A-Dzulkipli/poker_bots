"""Scratch: turn bench logs into markdown result tables for the README."""
import re, sys
ORDER = ["windrunner_glm", "invoker_gpt", "meepo_claude", "hadrian_marlowe_gpt", "valka_onderra_claude", "aggressor",
         "bassander_lin_glm", "alan_wake_claude", "max_payne_gpt", "jesse_faden_glm", "jack_joyce_gemini", "random"]
PAT = re.compile(r"vs\s+(\S+):\s+([+-][\d.]+) bb/100\s+\(SE ([\d.]+); first 50 hands ([+-][\d.]+), rest ([+-][\d.]+); (\d+)x(\d+), seed (\d+)")
for path in sys.argv[1:]:
    rows = {m.group(1): m.groups() for m in PAT.finditer(open(path).read())}
    print(f"\n`{path}`\n\n| opponent | seed | tables × hands | bb/100 | SE | first 50 | rest |\n|---|---|---|---|---|---|---|")
    for name in ORDER + [k for k in rows if k not in ORDER]:
        if name in rows:
            o, bb, se, early, late, t, h, seed = rows[name]
            print(f"| {o} | {seed} | {t} × {h} | {bb} | {se} | {early} | {late} |")
