"""Local reporting utilities; never used by a bot at runtime."""
import ast
import importlib
import json
import re
from pathlib import Path
import bench

RESULT = re.compile(r"^(\S+) vs\s+(.+?):\s+([+-]?[\d.]+) bb/100\s+\(SE ([\d.]+);.*; (\d+)x(\d+), seed (\d+), (\d+)s\)", re.MULTILINE)
SCORE = re.compile(r"(v\d+):Bot weighted score:\s+([+-]?[\d.]+)")

def records():
    rows = []
    seen = set()
    for path in sorted(Path('.').glob('*.log')):
        for candidate, opponent, mean, se, tables, hands, seed, seconds in RESULT.findall(path.read_text()):
            identity=(candidate,opponent.strip(),seed,tables,hands)
            if int(hands) == 250 and identity not in seen:
                seen.add(identity)
                rows.append(dict(version=candidate.split(':')[0], opponent=opponent.strip(), mean=float(mean), se=float(se), tables=int(tables), hands=int(hands), seed=int(seed), seconds=int(seconds), log=path.name))
    return rows

def decision_lines(version):
    source=Path(version + '.py').read_text()
    methods=[node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name not in ('__init__','init_state','metadata')]
    return sum(sum(bool(line.strip()) and not line.lstrip().startswith('#') for line in source.splitlines()[node.lineno-1:node.end_lineno]) for node in methods)

def result_table(rows):
    lines=['| Opponent | Seed | Tables × hands | bb/100 | SE |', '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['opponent']} | {row['seed']} | {row['tables']} × {row['hands']} | {row['mean']:+.2f} | {row['se']:.2f} |")
    return '\n'.join(lines)

PARAMETERS = {
    'informative_raise_pot_fraction': ('raise increment / pot before action', '0.50: half-pot raises carry full evidence; smaller raises carry proportional evidence, reasoning'),
    'samples': ('holdings/decision', '128 shared range and showdown samples; reasoning'),
    'range_power': ('dimensionless likelihood exponent', '1.0 baseline; 0.35 tests discounted evidence; 0.0 disables ranges for controls'),
    'likelihood_floor': ('probability', '0.12 contamination floor; reasoning'),
    'likelihood_width': ('percentile fraction', '0.12 smooth transition; reasoning'),
    'open_percentile': ('starting-hand percentile', '0.38 first preflop raise cutoff; reasoning'),
    'reraise_percentile': ('holding percentile', '0.82 second/later raise cutoff; reasoning'),
    'call_percentile': ('holding percentile', '0.45 call cutoff; reasoning'),
    'bet_percentile': ('made-hand percentile', '0.64 first postflop bet cutoff; reasoning'),
    'check_discount': ('probability fraction', '0.65 discounts strong hands after checks/limps; reasoning'),
    'pair_bonus_ranks': ('summed rank points', '13-point pocket-pair bonus; reasoning'),
    'suited_bonus_ranks': ('summed rank points', '3-point suited bonus; reasoning'),
    'call_margin_equity': ('equity fraction', '0.04 preflop margin, declines linearly to zero on river; reasoning'),
    'open_equity': ('equity fraction', '0.53 baseline; 0.49 in v3; 0.44 in v5, hypothesis-tested on seed 0'),
    'enter_equity': ('equity fraction', '0.44 minimum unopened button entry and maximum bluff equity; reasoning'),
    'value_equity': ('equity fraction', '0.68 baseline free-action value threshold; 0.64 in betting experiment'),
    'raise_equity': ('equity fraction', '0.80 value threshold facing a wager; reasoning'),
    'bet_pot_fraction': ('fraction of pot after calling', '0.65 bet/raise increment; reasoning'),
    'open_size_bb': ('big blinds, raise-to', '2.5 opening/limp-isolation size; reasoning'),
    'bluff_frequency': ('probability per eligible free action', '0.07 baseline; 0.25 in v3; 0.50 in v5, hypothesis-tested on seed 0'),
    'draw_bonus_percentile': ('percentile fraction', '0.18 for four-card flush/straight draws on flop/turn; reasoning'),
}

def parameter_table(version):
    bot=importlib.import_module('joshua_gpt.' + version).Bot()
    lines=['| Parameter | Value | Unit | Choice |', '|---|---:|---|---|']
    for name, value in bot.metadata()['params'].items():
        unit, reason=PARAMETERS[name]
        lines.append(f'| `{name}` | {value} | {unit} | {reason} |')
    return '\n'.join(lines)

if __name__ == '__main__':
    for path in sorted(Path('.').glob('*field*.log')):
        contents=path.read_text()
        rows=RESULT.findall(contents)
        scores=SCORE.findall(contents)
        print(path.name, len(rows), 'opponents', scores or 'in progress')
