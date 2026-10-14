"""Build the requested README from saved benchmark output, never inferred scores."""
import ast
import datetime
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path
import bench

DESCRIPTIONS = {
1: ('Baseline uniform equity', 'Uses 64 random-opponent showdown samples, price-based calls, value raises and occasional postflop bluffs. Explicit suit, rank, pairing and draw features adjust postflop calls, bluff eligibility and sizes.', 'Establish an executable baseline with board texture and draws.', 'Kept as the initial baseline; its loose interpretation of random-hand equity lost heavily to strong ranges.'),
2: ('Range selection correction', 'Discounts random-hand equity by the estimated fraction of the opponent strength distribution still present. Betting and repeated raising imply narrower ranges than checking.', 'Test whether treating a bettor as a random hand caused the baseline losses.', 'Kept as the next parent: a broad field improvement, while still losing to strong aggressive opponents.'),
3: ('Postflop behavior tracking', 'Tracks postflop raise frequency and responses to our bets with fixed-shape counts. Bluffs more when observed folds exceed the immediate break-even rate and widens the estimated betting range against frequent raisers.', 'Recover pots surrendered by a rigid, conservative range model.', 'Kept: substantial field improvement and a clear head-to-head win against v2.'),
4: ('Preflop behavior tracking', 'Adds preflop raise frequency to reduce the strength discount against frequent preflop raisers. It also exposes the previously fixed semibluff and pressure thresholds as constructor arguments, without changing their values.', 'Played-hand diagnostics showed large losses in preflop pots conceded without showdown.', 'Retained for subsequent sizing experiments; the parent match alone did not establish an improvement.'),
5: ('Smaller postflop bets', 'Uses 0.40-pot base bets instead of 0.65. Everything else matches v4, so steals need fewer folds and cost less when unsuccessful.', 'Test a materially different sizing rule after the diagnostics showed non-showdown losses.', 'Kept: broad gains against the strong field and a positive parent match.'),
6: ('Street-appropriate fold statistics', 'Counts only postflop responses in the fold statistic used for postflop bluffs. Preflop calls no longer suppress the postflop steal estimate.', 'Condition the statistic on the situation in which it is used.', 'Kept for statistical relevance; the modest aggregate score change is not strong evidence on its own.'),
7: ('Value three-bets', 'Uses the ordinary value threshold against a first preflop raise, reserving the strict reraise threshold for an already-reraised pot. The postflop strategy is otherwise unchanged.', 'Avoid treating an ordinary preflop open as though it were a postflop value reraise.', 'Kept as the parent of v8, v9, v10 and v11, and as a finalist. Not deployed: v10 had the higher held-out weighted score and the corrected private-card draw model.'),
8: ('Quarter-pot sizing', 'Reduces the postflop base fraction from 0.40 to 0.25, with the same texture increment. This tests whether further reducing the price of a steal improves the full field result.', 'Test the small-bet hypothesis at a distinct second size.', 'A seed-2 parent match won 9.85 bb/100 (SE 3.44), so this frozen version was included in held-out evaluation. Discarded for deployment after its held-out weighted score of +45.46 fell below v7 and v10.'),
9: ('Adaptive repeated-raise ranges', 'Widens the repeated-raise range when the opponent raises postflop frequently. Other decisions match v7.', 'Test whether a fixed narrow range was causing excessive folds against frequent reraisers.', 'Discarded after larger targeted tests: no demonstrated gain against the strong probe and a negative, inconclusive parent result. Full field not run.'),
10: ('Draw-supported semibluffs', 'Excludes straight-completing cards that only improve the shared board. Adds the probability of improving to the steal calculation, allowing drawing hands to semibluff when immediate folds alone would be insufficient.', 'Make drawing potential affect semibluff frequency directly and remove shared-board-only straight outs.', 'Deployed: it has the highest held-out weighted score of the three feature-enabled finalists (+51.86), a corrected private-card draw model, and no clear loss to its parent. The modest margin over v7 is not a claim of statistically proven superiority.'),
11: ('Sampled current-hand ranges', 'Uses 128 random deals and weights opponents by the rank of their current made hand, retaining a small below-cutoff bluff likelihood. Future cards affect showdown equity but never determine which sampled opponents enter the current betting range.', 'Test a more direct range model than the linear strength-percentile approximation.', 'Discarded: it lost the 128-table parent match by 19.72 bb/100 (SE 5.30), with no clear targeted strong-opponent gain. Full field not run.'),
}
UNITS = {
'samples': 'Monte Carlo deals/decision', 'open_equity': 'showdown pot share',
'value_equity': 'adjusted pot share', 'reraise_equity': 'adjusted pot share',
'call_margin': 'pot-share probability points', 'pressure_penalty': 'pot-share points/unit pressure',
'open_size_bb': 'big blinds, street raise-to total', 'bet_pot_fraction': 'fraction of pot after calling',
'texture_weight': 'equity points and extra pot fraction per normalized feature',
'draw_weight': 'equity points per next-card improvement probability',
'bluff_frequency': 'probability on eligible free actions', 'features_enabled': 'Boolean',
'bet_range_fraction': 'fraction of current strength distribution',
'quiet_range_fraction': 'fraction of current strength distribution',
'reraise_range_fraction': 'fraction of current strength distribution',
'prior_actions': 'pseudo-count actions', 'prior_fold_rate': 'fold/response probability',
'prior_raise_rate': 'raise/action probability', 'exploit_bluff_frequency': 'bet probability when steal condition holds',
'fold_profit_margin': 'fold-probability points above break-even',
'semibluff_draw_probability': 'next-card improvement probability', 'semibluff_equity': 'random-hand pot share',
'pressure_odds_cap': 'pot-odds fraction', 'range_bluff_weight': 'relative likelihood below current-strength cutoff',
}
ROW = re.compile(r'(?P<candidate>\w+):Bot vs\s+(?P<opponent>\S+):\s+(?P<mean>[+-][\d.]+) bb/100\s+\(SE (?P<se>[\d.]+);.*; (?P<tables>\d+)x(?P<hands>\d+), seed (?P<seed>\d+), (?P<seconds>\d+)s\)')
SCORE = re.compile(r'(\w+):Bot weighted score:\s+([+-][\d.]+)')


def collect():
    results, scores = {}, []
    for path in sorted(Path('.').glob('*.log')):
        if path.name.startswith('smoke'):
            continue
        text = path.read_text()
        rows = []
        for match in ROW.finditer(text):
            row = match.groupdict()
            for key in ('tables', 'hands', 'seed', 'seconds'):
                row[key] = int(row[key])
            for key in ('mean', 'se'):
                row[key] = float(row[key])
            row['log'] = path.name
            key = tuple(row[k] for k in ('candidate', 'opponent', 'seed', 'tables', 'hands'))
            results[key] = row
            rows.append(row)
        match = SCORE.search(text)
        if match:
            assert len(rows) == 15, path
            scores.append({'candidate': match[1], 'score': float(match[2]),
                           'seed': rows[0]['seed'], 'tables': rows[0]['tables'], 'log': path.name})
    return list(results.values()), scores


def decision_lines(path):
    source = path.read_text()
    lines = source.splitlines()
    methods = [m for c in ast.parse(source).body if isinstance(c, ast.ClassDef)
               for m in c.body if isinstance(m, ast.FunctionDef) and m.name not in {'__init__', 'init_state', 'metadata'}]
    return sum(bool(line.strip()) and not line.lstrip().startswith('#') for m in methods for line in lines[m.lineno-1:m.end_lineno])


def table(rows):
    output = ['| Opponent | Seed | Tables × hands | bb/100 | SE | Log |',
              '|---|---:|---:|---:|---:|---|']
    for row in sorted(rows, key=lambda r: (r['seed'], r['tables'], -bench.STRENGTH_RANK.get(r['opponent'], 0), r['opponent'])):
        output.append(f"| {row['opponent']} | {row['seed']} | {row['tables']} × {row['hands']} | {row['mean']:+.2f} | {row['se']:.2f} | [{row['log']}]({row['log']}) |")
    return '\n'.join(output)


def build(final, finished):
    results, scores = collect()
    start = datetime.datetime.fromisoformat('2026-10-14T05:36:28+00:00')
    end = datetime.datetime.fromisoformat(finished)
    elapsed = (end - start).total_seconds()
    notes = Path('run_notes.txt').read_text()
    check_output = notes[notes.index('python      '):notes.index('Audit:')].strip()
    sections = [f'''# faye_gpt — phase_5_cowboy_bebop/faye_gpt

- **Model:** GPT-6 (Codex); no more specific runtime model suffix is available to this session.
- **Prompt:** botgen-v5.
- **Engine contract:** decafb9, engine_sha256 `295477eff7637056`; hash verified by the supplied check.
- **Executed:** yes, on the existing project Python 3.13.12 / JAX 0.11.2 CPU environment, invoked through `uv run python` from this directory. No packages were installed; `.venv`, `uv.lock`, `bench.py` and `equity.py` were not edited.
- **Start:** 2026-10-14 05:36:28 UTC / 16:06:28 Australia/Adelaide, after the successful check.
- **Finish:** {finished}. **Wall-clock used:** {elapsed / 60:.2f} minutes. A wrap-up reminder was scheduled; the iteration deadline was 06:26:28 UTC and the overall deadline 06:36:28 UTC.
- **Deployment:** `from .{final} import Bot as DeployedBot`.

The first check encountered a sandbox restriction on the existing uv cache. The identical command then succeeded with approved execution access:

```text
{check_output}
```

## Access and historical audit

Before reading the pasted directory-only restriction, the agent read the requester-provided attachment and issued `pwd` plus a search for AGENTS.md under the wider named project. The search returned no file paths or contents. No opponent source, engine source, other bot source, experiment results outside this directory, or git state was read by the agent. Subsequent reads and all authored project files stayed inside this directory; the supplied benchmark loaded the engine and black-box opponents as authorized. The initial cache error and exact audit are preserved in [run_notes.txt](run_notes.txt).

Versions were frozen after their first benchmark. **Historical parameter-exposure limitation:** v1–v3 left the 0.15 semibluff draw threshold, 0.40 semibluff equity threshold, and 0.50 pressure cap as literals. v4 exposed them without changing their values; v1–v3 remain unchanged for honest provenance and are not deployment candidates. The later versions expose the betting, calling, pressure and semibluff thresholds as constructor parameters. Structural card counts, rank definitions and poker arithmetic remain constants.

## Design and experimental protocol

The main lineage combines small value bets and steals with conservative calls against selected betting ranges. Fixed-shape float32 counters track how often the opponent raises preflop and postflop and folds to a postflop bet. The strategy adapts to those behaviors, never to identity. Explicit board and draw features alter equity margins, sizes, and bluff eligibility; no keys or hand histories are stored.

All performance results below use 250-hand sessions, balanced candidate seating, and the supplied table-clustered standard error. The required 2 × 20 smoke tests are execution checks only and are excluded from performance tables. Training used only seeds 0–9; seed 1000 is held out and used once per finalist for the field. A repeated result with identical candidate, opponent, seed and sample size is displayed once, never counted as an independent sample. Field scores use the exact rank/sign weighting printed by bench.py. No aggregate score SE is invented from opponent SEs.

The first field run's printed per-opponent durations sum to {sum(r['seconds'] for r in results if r['candidate']=='v1' and r['log']=='field_v1_seed0.log')} seconds. Later concurrent runs shared CPU resources, so their timings are not single-bot speed comparisons.

### Completed field scores

| Candidate | Seed | Tables × 250 per opponent | Weighted score | Log |
|---|---:|---:|---:|---|''']
    for score in sorted(scores, key=lambda s: (s['seed'], int(s['candidate'].split('v')[-1]), s['candidate'].startswith('ablation'))):
        sections.append(f"| {score['candidate']} | {score['seed']} | {score['tables']} × 250 | {score['score']:+.2f} | [{score['log']}]({score['log']}) |")
    for version, (title, what, why, verdict) in DESCRIPTIONS.items():
        path = Path(f'v{version}.py')
        if not path.exists():
            continue
        bot = bench.load(f'v{version}:Bot')
        metadata = bot.metadata()
        parent = metadata['parent']
        parent_params = bench.load(f'v{parent}:Bot').metadata()['params'] if parent else {}
        sections.append(f'''\n## v{version}: {title}

**What it does:** {what}

**Why it exists:** {why}

**Parameters:** Values are defaults for this immutable version. “Inherited” means unchanged from parent v{parent}; new reasoning choices were tested in the runs below, not fitted to held-out seed 1000.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|''')
        for name, value in metadata['params'].items():
            choice = 'Inherited' if name in parent_params and value == parent_params[name] else 'Reasoned experiment/default; tested in listed training runs'
            sections.append(f'| `{name}` | `{value}` | {UNITS[name]} | {choice} |')
        rows = [r for r in results if r['candidate'] == f'v{version}']
        sections.append('\n**Results:**\n\n' + (table(rows) if rows else 'Not run.'))
        for score in scores:
            if score['candidate'] == f'v{version}':
                sections.append(f"\nWeighted field score, seed {score['seed']}, {score['tables']} × 250: **{score['score']:+.2f}**.")
        sections.append(f'\n**Verdict:** {verdict}\n\n**Decision code:** {decision_lines(path)} nonblank, noncomment lines in act, observe and defined helpers, including signatures and multiline returns; constructors, init_state and metadata excluded. The shorter early baselines did not use the entire allowed budget.')
    sections.append(Path('final_notes.md').read_text())
    sections.append('''\n## Files and reproduction

- `v1.py` onward are immutable version sources; `__init__.py` names the deployed version.
- `ablation_v*.py` are self-contained copies of the indicated strategy with only the default `features_enabled=False` (and descriptive docstring) changed. They import no other versions. Their metadata reports the originating version and disabled parameter explicitly.
- `smoke_*.log` are short execution checks, not scored results. `field_*.log`, `parent_*.log`, `targeted_*.log`, `ablation_match_*.log` and `heldout_self_*.log` preserve raw benchmark output, including first-50/rest comparisons and timings.
- `diagnose.py` and its log summarize played public actions and outcomes; they do not inspect opponent code or hidden cards.
- `validate.py` checks metadata, sample limits, action dtypes, legal action sizes and feature-sensitive decisions on constructed observations; `validation_*.log` records its output.
- `run_notes.txt`, `final_notes.md`, `report.py`, `finish_notes.py`, `results.json` and `frozen_manifest.json` are audit/reporting scratch files. `frozen_manifest.json` records source fingerprints and decision counts.

Examples, run from this directory with the existing environment:

```sh
uv run python bench.py check
uv run python bench.py DEPLOYED:Bot field --seed 1000 --tables 128 --hands 250
uv run python validate.py DEPLOYED:Bot
```

The held-out command documents reproduction only; it was not repeatedly rerun to select a favorable result.
'''.replace('DEPLOYED', final))
    rendered = '\n'.join(sections).replace('“Inherited” means unchanged from parent vNone;', 'This initial version has no parent;')
    Path('README.md').write_text(rendered + '\n')
    Path('results.json').write_text(json.dumps({'results': results, 'field_scores': scores}, indent=2) + '\n')


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])
