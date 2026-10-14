"""Build the requested report from the actual benchmark logs and audited metadata."""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from bench import weight

parser = argparse.ArgumentParser()
parser.add_argument('--final', required=True)
parser.add_argument('--ablation', required=True)
parser.add_argument('--finalists', nargs='+', required=True)
args = parser.parse_args()
audit = json.loads(Path('audit_results.json').read_text())
versions = {entry['version']: entry for entry in audit}
pattern = re.compile(r'^(v\d+):Bot vs\s+(\S+):\s+([+-][\d.]+) bb/100\s+\(SE ([\d.]+); first 50 hands ([+-][\d.]+), rest ([+-][\d.]+); (\d+)x(\d+), seed (\d+), (\d+)s\)', re.M)
results = []
scores = []
for path in sorted(Path('.').glob('*.log')):
    content = path.read_text()
    rows = []
    for match in pattern.finditer(content):
        version, opponent, mean, se, early, late, tables, hands, seed, seconds = match.groups()
        row = dict(version=version, opponent=opponent, mean=float(mean), se=float(se), early=float(early), late=float(late), tables=int(tables), hands=int(hands), seed=int(seed), seconds=int(seconds), log=path.name)
        if row['hands'] == 250:
            rows.append(row)
            results.append(row)
    score = re.search(r'(v\d+):Bot weighted score:\s+([+-][\d.]+)', content)
    if score:
        assert len(rows) == 12, path
        scores.append(dict(version=score[1], score=float(score[2]), seed=rows[0]['seed'], tables=rows[0]['tables'], log=path.name))

# Audit complete logs and recompute the public score from printed matchup results.
for version in versions:
    smoke = Path(f'smoke_{version}.log').read_text()
    assert '2x20' in smoke and 'Traceback' not in smoke, version
    assert any(row['version'] == version and row['seed'] == 0 for row in scores), version
for score in scores:
    field_rows = [row for row in results if row['log'] == score['log']]
    recomputed = sum(weight(row['opponent'], row['mean']) * row['mean'] for row in field_rows) / 12
    assert abs(recomputed - score['score']) < 0.02, score

notes = {
 'v10': ('Adds occasional out-of-position checks with value hands before the river. The check frequency rises with the opponent’s observed free-betting rate, inviting bluffs while preserving the rest of v6’s strategy.', 'Test whether inducing bets earns more than immediately betting value into frequent bettors.'),
 'v1': ('Opens every button, isolates stronger hands, and mixes light three-bets. Postflop it bets value and some weaker hands, then uses betting pressure to narrow the opponent range before calling or raising.', 'Establish a compact stateless baseline before adding any learning.'),
 'v2': ('Adds smoothed counts of free betting and folds to our bets or raises. Frequent bettors receive less range credit, and frequent folders face more bluffs. Its starting strategy is otherwise exactly v1.', 'Test whether observed behavior improves the same strategy without relying on opponent identity.'),
 'v3': ('Keeps v2’s two learned rates and maintains its bluff frequency through the turn and river. Bet sizes and all other parameters stay unchanged.', 'Test whether automatically reducing later-street bluffs gives up profitable opportunities against persistent folders.'),
 'v4': ('Uses v3’s decisions and learning, with postflop bets and raises sized at 90% instead of 65% of the pot after calling.', 'Test larger value extraction and greater fold pressure; check its robustness against its parent.'),
 'v5': ('Adds occasional low-equity raises to v3 when facing the first opposing bet on a street. Their frequency is scaled by observed free-betting and fold-to-aggression rates.', 'Test whether fighting back with bluff raises improves on passive bluff catching and value raises.'),
 'v6': ('Separates bets made when checking is free from raises made when facing our aggression. Each gets its own observed frequency and range-strength prior, while v3’s bet sizes and bluff policy remain unchanged.', 'A player can bet a wide range but re-raise a narrow one. Avoid applying a loose free-betting estimate to every raise.'),
 'v7': ('Uses v6’s separate betting and raising estimates with a 90%-pot postflop size. The tracking rules and action frequencies are unchanged.', 'Retest the larger-size idea after repairing the interpretation of re-raises.'),
}
units = {
 'trap_frequency': 'maximum value-check probability, scaled by free-betting rate',
 'num_samples': 'Monte Carlo deals per decision', 'open_size_bb': 'big blinds, raise-to',
 'reraise_multiple': 'multiple of the current preflop bet', 'threebet_equity': 'equity probability',
 'light_threebet_frequency': 'mixing probability', 'isolate_equity': 'equity probability',
 'value_equity': 'equity probability', 'raise_equity': 'adjusted equity probability',
 'bluff_frequency': 'mixing probability with air', 'semibluff_frequency': 'mixing probability with medium equity',
 'bluff_split_equity': 'equity probability', 'bluff_street_decay': 'frequency multiplier per later street',
 'bet_pot_fraction': 'fraction of pot after calling', 'range_retention': 'range fraction retained per ordinary pressure unit',
 'prior_street_weight': 'current-street bet equivalents', 'realization_fraction': 'fraction of equity realized before river',
 'call_margin_equity': 'equity probability above pot odds', 'prior_events': 'pseudo-opportunities per estimated rate',
 'prior_bet_rate': 'probability of betting when checking is free', 'prior_fold_rate': 'probability of folding to our aggression',
 'betting_adjustment': 'retention change per unit of observed frequency deviation',
 'bluff_adjustment': 'bluff probability change per unit of fold-rate deviation',
 'bluff_raise_frequency': 'maximum bluff-raise probability before rate scaling',
 'prior_raise_rate': 'probability of raising when facing our aggression',
 'reraise_range_retention': 'range fraction retained per re-raise pressure unit',
}


def result_table(rows):
    lines = ['| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['opponent']} | {row['seed']} | {row['tables']} × 250 | {row['mean']:+.2f} | {row['se']:.2f} | {row['early']:+.1f} | {row['late']:+.1f} |")
    return '\n'.join(lines)


start = datetime(2026, 10, 14, 3, 57, 3, tzinfo=timezone.utc)
finished = datetime.now(timezone.utc)
elapsed = (finished - start).total_seconds() / 60
total_elapsed = (finished - datetime(2026, 10, 14, 3, 54, 30, tzinfo=timezone.utc)).total_seconds() / 60
local_start = start.astimezone(ZoneInfo("Australia/Adelaide"))
local_finished = finished.astimezone(ZoneInfo("Australia/Adelaide"))
final_meta = versions[args.final]['metadata']
held_scores = {row['version']: row for row in scores if row['seed'] == 1000}
assert all(version in held_scores for version in args.finalists)
assert args.final in held_scores
held_rows = [row for row in results if row['version'] == args.final and row['seed'] == 1000 and not row['opponent'].endswith(':Bot')]
assert len(held_rows) == 12
out = [f'# kraken_gpt — {args.final} deployed', '',
       '- Lineage: `phase_4_sea_monsters/kraken_gpt`.',
       '- Model: GPT-6 (Codex; exact serving variant unavailable).',
       '- Prompt: `botgen-v4`.',
       '- Engine: `decafb9`, SHA-256 prefix `295477eff7637056`.',
       '- Executed: **yes**. Every version passed its required 40-hand smoke test.',
       f'- Wall-clock development time: **{elapsed:.1f} minutes**, {local_start.isoformat()} to {local_finished.isoformat()} (Australia/Adelaide). Initial permission troubleshooting before this successful-check session is excluded.',
       f'- Total recorded elapsed time including initial permission troubleshooting: **{total_elapsed:.1f} minutes**.',
       '- The one-shot reminder service rejected the schedule; elapsed time was checked between steps.',
       f'- Deployment: `__init__.py` imports `Bot` from `{args.final}.py` as `DeployedBot`.', '',
       '## Environment check', '',
       'The first sandboxed attempts failed while opening the existing uv cache. The approved retry returned:', '',
       '```text',
       'python      /Users/alik/Documents/Projects/poker_bots/poker_bots/.venv/bin/python3 (3.13.12)',
       'jax         0.11.2 (cpu)',
       'poker_bots  /Users/alik/Documents/Projects/poker_bots/poker_bots/src/poker_bots',
       'engine      sha256 295477eff7637056 (matches the contract)',
       'smoke test  random vs aggressor, 40 hands in 2s: OK',
       '```', '',
       '## Protocol and evidence', '',
       'All performance results below use 250-hand sessions with fresh state at each table, balanced candidate seats, 100bb initial stacks, and the supplied rebuy/cash-out rules. Smoke tests use 2 × 20 hands and are excluded from performance tables. Development uses seeds 0–5; seed 1000 was used exactly once for each of the three finalists. Standard errors are table-clustered, as printed by the supplied benchmark. Small differences under roughly two standard errors are not treated as conclusive.', '',
       'The weighted score is the supplied twelve-opponent score, with rank-dependent weights that differ for positive and negative results. The first-50 and remaining-200 columns are bb/100, not cumulative winnings. Early/late differences alone do not prove learning: stack distributions and which hands reach later streets also change.', '',
       f"The first full-field run took approximately {sum(row['seconds'] for row in results if row['version']=='v1' and row['log']=='field_v1_seed0.log')} seconds by summing the benchmark's printed timings. Some later runs overlapped, so their individual durations include contention.", '',
       '| Version | Seed | Tables × hands per opponent | Weighted score |', '|---|---:|---:|---:|']
for row in sorted(scores, key=lambda row: (int(row['version'][1:]), row['seed'])):
    out.append(f"| {row['version']} | {row['seed']} | {row['tables']} × 250 | {row['score']:+.2f} |")

for version, entry in versions.items():
    metadata = entry['metadata']
    parent = metadata['parent']
    parent_name = f'v{parent}' if parent is not None else None
    if version not in notes:
        notes[version] = (f"A stateless control for {parent_name}: replaces learned rates with their exact prior values. All action logic, sampling, sizes and thresholds are retained.", 'Isolate the effect of learning in the finalist, without changing its underlying poker strategy.')
    what, why = notes[version]
    out += ['', f'## {version}', '', f'**What it does.** {what}', '', f'**Why it exists.** {why}', '', f"**Decision line count:** {entry['decision_lines']}. Imports, comments, blank lines, constructor, initialization and metadata are excluded.", '', '**Parameters.** All probabilities are dimensionless. Constants for card/street indices, array layout, clipping to [0, 1], and floating-point numerical protection are structural rather than fitted parameters.', '', '| Parameter | Value | Unit | Choice |', '|---|---:|---|---|']
    parent_params = versions[parent_name]['metadata']['params'] if parent_name else {}
    for name, value in metadata['params'].items():
        if name in parent_params and value == parent_params[name]:
            chosen = f'Inherited from {parent_name}'
        elif name in parent_params:
            chosen = f'One-variable development experiment; {parent_name} used {parent_params[name]}'
        else:
            chosen = 'Reasoned/source-informed default; no fitted table'
        out.append(f'| `{name}` | {value} | {units[name]} | {chosen} |')
    rows = sorted([row for row in results if row['version'] == version], key=lambda row: (row['seed'], row['log'], row['opponent']))
    out += ['', '**Results.**', '', result_table(rows)]
    version_scores = [row for row in scores if row['version'] == version]
    if version_scores:
        out += ['', 'Weighted field scores: ' + '; '.join(f"seed {row['seed']}: **{row['score']:+.2f}** ({row['tables']} × 250 per opponent)" for row in version_scores) + '.']
    if version == args.final:
        verdict = 'Deployed after the held-out comparison; see final choice below.'
    elif version in {'v8', 'v9', 'v11'}:
        verdict = f'Retained as a frozen no-learning control. Its adaptive parent {parent_name} won the direct ablation match; see that parent’s results. Not selected for deployment.'
    elif version == 'v4':
        verdict = 'Retained as an informative sizing experiment, but not deployed: its field gain came with a material loss to v3.'
    elif version == 'v7':
        verdict = 'Discarded for deployment despite its development field lead: it lost to v6 on both tested seeds, approximately -14.79 bb/100 with pooled SE 3.87 over 96,000 hands.'
    elif version == 'v10':
        verdict = 'Not deployed: the value-checking idea did not earn the final choice on held-out weighted score; its uncertain parent advantage is reported rather than treated as decisive.'
    elif version == 'v5':
        verdict = 'Discarded as a strategy branch: extra bluff-raise complexity did not establish an improvement over v3.'
    else:
        children = [name for name, child in versions.items() if child['metadata']['parent'] == int(version[1:]) and name not in {'v8', 'v9', 'v11'}]
        verdict = ('Kept as parent of ' + ', '.join(children) + '. ') if children else 'Retained for comparison. '
        verdict += 'Final selection uses the held-out evidence and parent comparisons below.'
    out += ['', f'**Verdict.** {verdict}']

out += ['', '## Opponent tracking', '',
        'Only named float32 summary counts persist across hands. No action logs, identities, random keys, hidden cards or external memory are stored. `observe` counts valid postflop opponent actions. An action immediately after our bet/raise on the same street is a facing-bet opportunity; all other voluntary postflop actions are free-check opportunities. Fold and raise counts use these denominators. Facing-bet matching requires both the current and previous history entries to be valid.', '',
        'The deployed tracker starts with 20 pseudo-opportunities: 8 free bets (40%), 10 folds facing aggression (50%), and 3 raises facing aggression (15%). The raise and fold estimates share the same facing-bet denominator. Priors shrink early estimates toward a sound opening strategy.', '',
        '- Ordinary-bet range retention = `clip(0.58 + 0.65 × (free_bet_rate − 0.40), epsilon, 1)`.',
        '- Re-raise range retention = `clip(0.40 + 0.65 × (raise_rate − 0.15), epsilon, 1)`.',
        '- Each current-street event contributes one pressure unit; prior-street events contribute 0.5. Their retained fractions multiply. Preflop keeps the fixed 0.58 model.',
        '- Calling uses `max(raw_equity + remaining_range − 1, 0) / remaining_range`, pot odds, a 0.02 equity margin, and 90% realization when future betting is possible. Value raises use 0.85 adjusted equity.',
        '- Air bluffs start at 35%; medium-equity bluffs at 50%. Add `1.25 × (fold_rate − 0.50)` and clip to [0, 1]. A ten-percentage-point increase in folding therefore adds 12.5 percentage points of bluff probability. No automatic later-street decay remains.', '',
        'For example, a 70% free-betting rate gives 0.775 retention for ordinary bets, while a 15% re-raising rate still gives 0.40 retention for re-raises. Thus frequent betting does not automatically justify loose calls against re-raises. These are interpretable range heuristics, not exact conditional-equity estimates.', '',
        f'**Exact ablation:** `{args.ablation}.py` is the deployed strategy with rates fixed to the same priors and no persistent state. Its head-to-head results against {args.final} and field comparison quantify learning separately from the underlying strategy.', '',]
ablation_rows = [row for row in results if {row['version'], row['opponent'].replace(':Bot','')} == {args.final, args.ablation}]
out.append(result_table(ablation_rows))
if 'trap_frequency' in final_meta['params']:
    out.append('\nThe deployed value-checking rule uses probability `0.65 × free_bet_rate` with value hands out of position before the river. It starts at 26% and adapts with observed betting; all eligible value hands still use the normal call/raise model after a bet.')

for version in [args.final, args.ablation]:
    selected = [row for row in scores if row['version'] == version]
    out.append('\n' + version + ' weighted scores: ' + '; '.join(f"seed {row['seed']}: {row['score']:+.2f}" for row in selected) + '.')

out += ['', '## Final choice and held-out results', '',
        f"**Deploy {args.final}.** Held-out weighted score: **{held_scores[args.final]['score']:+.2f}**, seed 1000, {held_scores[args.final]['tables']} × 250 hands against each of twelve opponents.", '',
        'Finalist held-out scores: ' + '; '.join(f"{version}: {held_scores[version]['score']:+.2f}" for version in args.finalists) + '.', '',
        result_table(held_rows), '',
        (Path('selection_note.txt').read_text().strip() if Path('selection_note.txt').exists() else 'Selection rationale pending.') + '\n\n' + 'The final choice also considers direct parent performance and clarity. The highest development score alone is insufficient evidence when an experiment loses to its own parent. All version files were frozen before benchmarking; no held-out result was used to change a strategy.', '',
        '### Known weaknesses', '',
        'Equity is sampled against random cards and transformed using a coarse range-strength model. It does not reconstruct opponent card combinations, account fully for prior calls, or model different bet sizes. Monte Carlo noise can change borderline decisions. The bot opens every button and learns only postflop behavior, leaving preflop adaptation limited.', '',
        'Rates pool all postflop streets and are conditional on opportunities created by our own strategy. Twenty-event priors delay learning rare re-raises. An adapting opponent can establish a misleading early pattern, then change behavior; cumulative counts have no forgetting and may react slowly. Aggressive bluff increases can be exploited by an opponent who begins calling or check-raising more. Tests against our own adaptive versions help, but do not establish strength against the unseen phase-4 finals.', '',
        '### Validation, provenance and files', '',
        '`audit.py` checks allowed bot imports, one Bot class per version, constructor/metadata consistency, decision-line and sample budgets, immutable version hashes, exact JIT tracking counts, float32 state shape/dtype preservation, and persistence across hands. Full simulator benchmarks exercise action tracing and play. `protected_files.sha256` records the supplied helpers and frozen opponents; final verification checks that they remain unchanged.', '',
        'Ideas borrowed from the supplied frozen windrunner_glm include wide heads-up opening, pressure counts, light three-bets and mixed postflop aggression. The supplied invoker_gpt informed range truncation and excluding unmatched all-in chips from pot odds. The frozen meepo_claude strategy was also read for comparison. No identity recognition, fingerprints or opponent-specific branches are used.', '',
        'Scope disclosure: before reading the original pasted directory restriction, an initial repository-wide filename listing was made. No source contents outside this bot directory were inspected. Subsequent source inspection and all created/modified project files were confined to this directory; the supplied benchmark used the authorized Python environment. No commits were made.', '',
        'Scratch evidence is retained here: `smoke_*.log`, `field_*.log`, `heldout_*.log`, `parent_*.log`, `probe_*.log`, `audit.log`, `audit_results.json`, `protected_files.sha256`, `protected_files_check.log`, `session_notes.txt`, `audit.py`, and this report’s generator `build_readme.py`, `benchmark_results.json`, and `selection_note.txt`. These utilities are not imported by the bot. Every performance table is parsed from recorded benchmark output. Pooled parent statistics are explicitly derived from the reported means, errors and table counts. No benchmark results were invented.', '']
Path('README.md').write_text('\n'.join(out))
Path('benchmark_results.json').write_text(json.dumps({'runs': results, 'weighted_scores': scores}, indent=2) + '\n')
print(f'Wrote README.md for {args.final}; {elapsed:.1f} minutes; {len(results)} reported matchups')
