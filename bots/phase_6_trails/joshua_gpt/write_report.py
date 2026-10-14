"""Build the README from local measured results and explicit selection notes."""
import datetime
import json
from pathlib import Path
from report_tools import records, result_table, parameter_table, decision_lines, SCORE
import bench

selection=json.loads(Path('selection.json').read_text())
final=selection['final']
all_rows=records()
start=datetime.datetime(2026,10,14,7,28,2,tzinfo=datetime.timezone.utc)
end=datetime.datetime.now(datetime.timezone.utc)
minutes=(end-start).total_seconds()/60
notes={
'v7': ('Keeps v5 decisions and range features. Reconstructs prior contributions and pot sizes from history; a raise increment below half the prior pot supplies proportionally less strength evidence.', 'Test whether distinguishing inexpensive raises from substantial wagers makes action-based ranges more robust.'),
'v1': ('Baseline importance-sampled range equity. Every opponent action updates each holding. Calls, value bets and sparse bluffs use the resulting equity.', 'Establish an executable, compact, stateless baseline.'),
'v2': ('Adds four-card straight/flush draws to flop and turn range quality. Button limps favor weaker holdings rather than using the calling-an-open model. Decision thresholds remain unchanged.', 'Test a more realistic passive-action and drawing-hand model.'),
'v3': ('Keeps the v2 range model. Lowers opening equity from 0.53 to 0.49 and free-action value equity from 0.68 to 0.64; raises eligible bluff frequency from 0.07 to 0.25.', 'Test a related increase in betting frequency without changing inference or bet sizes.'),
'v4': ('Keeps v2 decisions and features, but raises the full action likelihood to power 0.35 instead of 1.0. Each action is less conclusive, while the whole line still matters.', 'Test whether model uncertainty makes the baseline posterior too concentrated.'),
'v5': ('Keeps v3 inference, sizing, and value thresholds. Lowers opening equity to 0.44, matching minimum button entry, and raises eligible bluff frequency to 0.50.', 'Test raising all playable unopened button hands and applying more pressure.'),
}
parts=[f'# joshua_gpt — deployed {final}',
'''- **Lineage:** `phase_6_trails/joshua_gpt`
- **Model:** GPT-6 (Codex), as identified by this session; a more precise model identifier was not supplied.
- **Prompt:** `botgen-v6`
- **Engine:** `decafb9`, SHA-256 prefix `295477eff7637056`
- **Executed:** yes; existing repository environment, Python 3.13.12, JAX 0.11.2 CPU. No packages installed or environment changes.
- **Access:** execution and requested file writes used sandbox escalation because the assigned directory and existing uv cache are outside the desktop chat's default writable roots.''',
f'- **Wall-clock:** {minutes:.2f} minutes, {start.isoformat()} to {end.isoformat()}. Iteration cutoff: 08:18:02 UTC; deadline: 08:28:02 UTC.',
'''- **Sandbox deviation:** before reading the full pasted task, an initial `rg --files` listing accidentally ranged across the repository and exposed filenames outside this directory, including opponent, engine, and earlier-phase filenames. No opponent source, engine source, earlier result contents, or git state were opened. Subsequent inspection and all authored files stayed inside this directory. The explicitly supplied task attachment was read to obtain the request.

## Environment check

The first attempt hit a desktop sandbox restriction on the existing uv cache. The retry with sandbox escalation succeeded:''',
'```text\n'+Path('check.log').read_text().rstrip()+'\n```',
'## Design\n\n'+selection['design'],
'''The bot uses at most 128 candidate holdings and legal board completions per decision, sharing samples between inference and showdown evaluation. It reconstructs the posterior from this hand's history each action, without persistent learning or opponent identification. Versions are independent and benchmarked sources were frozen.

## Benchmark protocol

Performance tables use 250 hands per table, balanced candidate seats, and table-clustered SE in bb/100. Separate 2 × 20 smoke tests establish execution and are excluded from performance tables. Development seeds are 0–9; held-out field evaluation uses seed 1000 once per finalist. Results smaller than roughly twice their SE are inconclusive. Small score differences are treated cautiously.

The first field run used 64 tables per opponent; later screening used 32 to preserve time for final checks. The v8 uniform control used 16 tables per opponent; held-out field and earlier-version checks use 32. Exact sizes appear below. Some runs overlapped, so timing includes resource contention.''',
f"The first field run's sum of printed opponent timings was {sum(row['seconds'] for row in all_rows if row['log']=='v1_field_s0.log')} seconds."]
versions=sorted([path.stem for path in Path('.').glob('v[0-9]*.py')],key=lambda name:int(name[1:]))
for version in versions:
    what, why=notes.get(version, ('Identical strategy to its metadata parent, with range_power=0.0 so every legal sampled opponent holding has unit weight. This uniform-hand control is not eligible for phase-6 deployment.', 'Measure the effect of range conditioning while keeping other strategy and sampling settings fixed.'))
    parts.extend([f'## {version}',f'**What it does:** {what}\n\n**Why it exists:** {why}\n\n**Decision-code lines:** {decision_lines(version)}; includes helper and act definitions, excludes imports, constructor, init_state, metadata, comments, blanks and shared helpers.', '### Parameters\n\n'+parameter_table(version), '### Results\n\n'+result_table([row for row in all_rows if row['version']==version])])
    for path in sorted(Path('.').glob(version+'_field*.log')):
        score=SCORE.findall(path.read_text())
        parts.append(f'Weighted score from `{path.name}`: **{float(score[0][1]):+.2f}**.' if score else f'`{path.name}`: incomplete; full weighted score not available.')
    parts.append('**Verdict:** '+selection['verdicts'].get(version, 'Experimental control only; not eligible for deployment.'))
parts.extend(['## Deployed range model',
'''The prior is uniform over legal opponent holdings after removing our cards and visible board cards. Preflop merit sums both ranks, adding named pair and suited bonuses. Later streets use evaluated made-hand percentiles on each historical board prefix; ties receive shared midranks. Four-card straight/flush draws get a named percentile bonus on flop and turn.

Each opponent action contributes a smooth likelihood. First preflop raises use the open threshold; first postflop bets use the betting threshold; second or later raises on a street use the tighter reraise threshold. Calls favor the upper holding distribution. Checks discount strong hands. Position distinguishes a big blind's free check from a button completion; button limps use the weaker passive model. Every valid opponent bet, raise, call and check contributes; blind posts do not. Folds terminate the relevant heads-up sequence.

Likelihoods have a nonzero floor for unusual play. In v7, a raise increment smaller than half the pot before the action contributes proportionally less log-likelihood evidence; calls and checks retain full weight. Prior pots are reconstructed from actual contribution increments, including blinds. Their product raised to range_power gives the holding weight. New board cards remove impossible holdings, while earlier actions retain their historical board context. Range-weighted showdown share drives entry, folds, calls, value bets, raises and bluff eligibility. Pot odds plus a declining future-street margin price calls. Sizing uses legal raise-to totals.

### Measured range table

`measure_ranges.py` calls the deployed range function on constructed histories. Flop: **K♠ 9♥ 4♠**; turn: **7♣**; river: **2♦**. Examples use a 2.5bb open, 9bb 3-bet, 3.5bb flop bet, 11bb flop check-raise, 8bb turn bet and 17.5bb river bet. Versions 1–6 distinguish action order but ignore nonblind bet sizes in inference. Version 7 additionally scales raise evidence by the raise increment divided by the prior pot.

Each row averages 10 batches of 128 sampled legal holdings, seeds 0–9, with no fixed hero hand. Retained mass is mean unnormalised weight multiplied by the fraction of all 1,326 starting combinations compatible with the board. This is a **soft weighted share of all starting hands**, not a hard exclusion: every legal holding retains positive weight. These are Monte Carlo measurements, not exact enumeration. Pocket-pair and two-broadway columns describe normalized posterior composition.'''])
ranges=json.loads(Path(final+'_ranges.json').read_text())
descriptions=[
'Broad; favors pairs, higher ranks and suited hands.',
'Weaker completions; premiums discounted, still possible.',
'Medium/high preflop merit, pairs and suited high cards.',
'High pairs and broadways favored; weak-hand tail retained.',
'Opening range that connects, strong draws, some bluffs.',
'Preflop caller that checked, then represented strength or a semibluff.',
'Whole opening and flop line, narrowed again on the turn.',
'Complete betting line; strong made hands favored, missed draws reduced.',
]
lines=['| Opponent line | Weighted starting-hand mass | Posterior pocket pairs | Posterior two broadways | Description |','|---|---:|---:|---:|---|']
for row, description in zip(ranges, descriptions):
    lines.append(f"| {row['line']} | {100*row['weighted_starting_share']:.2f}% | {100*row['pair_posterior_share']:.2f}% | {100*row['two_broadway_posterior_share']:.2f}% | {description} |")
parts.append(f"Effective sample size in the deployed model averages {ranges[0]['effective_samples']:.1f} of 128 after the button open and {ranges[7]['effective_samples']:.1f} after the three-street line. The extra small-turn-re-raise diagnostic is retained in the JSON measurements.")
parts.extend(['\n'.join(lines),'### What range conditioning is worth\n\n'+selection['ablation'],'## Final choice\n\n'+selection['choice']])
parts.append('### Held-out deployed results against the field\n\n'+result_table([row for row in all_rows if row['version']==final and row['seed']==1000 and row['opponent'] in bench.FIELD]))
parts.append('### Results against every earlier version\n\n'+result_table([row for row in all_rows if row['version']==final and row['opponent'].endswith(':Bot')]))
parts.append('''### Known weaknesses

This is an action-likelihood model, not an equilibrium solution. It does not fit calibrated bet-size distributions or learn between hands. Preflop merit omits explicit connectivity. Draw bonuses are approximate and include some shared-board draws. Repeated evidence can be overconfident, and 128 importance samples can leave few effective samples on narrow lines. Value sizing and future-street margins are coarse. Call pricing uses the displayed pot and capped call cost without separately modeling unmatched-bet refunds. One held-out seed cannot establish universal superiority or results against unseen phase-6 lineages.

## Verification and scratch files

`verify_delivery.py` checks complete field coverage, the deployment export, earlier-version comparisons and the matching uniform control. `audit.py` checks statelessness, unchanged state return, sample limits, decision lines, serializable metadata, one Bot class per version, and source hashes. Smoke tests and full benchmarks exercise engine compilation and action compatibility. `bench.py` and `equity.py` were not edited. No commits were made.

Reproducibility files retained: `check.log`, `vN_smoke.log`, field/head-to-head logs, measured `vN_ranges.json` and `.log` files, `measure_ranges.py`, `report_tools.py`, `write_report.py`, `audit.py`, `audit.log`, `source_hashes.json`, `selection.json`, `run_notes.txt`, `run_series.py`, `evaluate_finalist.py`, `evaluate_pairs.py`, `verify_delivery.py`, and `delivery_checks.log`. The bot only needs its deployed version, the one-line package export, and the provided helper at runtime.
''')
Path('README.md').write_text('\n\n'.join(parts)+'\n')
print(f'Wrote README.md; elapsed {minutes:.2f} minutes; deployed {final}.')
