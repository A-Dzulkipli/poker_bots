"""Assemble the experiment report from actual local benchmark logs and metadata."""
import ast
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import re
import sys

from bench import load

STRATEGIES = {
    1: ("Opens a broad range on the button and isolates limps more selectively from the big blind. Calls according to pot odds after a multiplicative aggression discount, and uses conservative value raises. Caps raises at the effective stack.", "Establish a simple heads-up baseline with position-aware preflop play.", "Retained as parent of v2. Its range discount forgets earlier streets and over-discounts near-nut hands; its 43-line baseline also falls below the nominal 45-line lower end."),
    2: ("Replaces the multiplicative equity haircut with an approximate truncated-range adjustment. Earlier opponent raises retain half their weight on later streets, while equity near one remains near one.", "Fix the baseline's disappearing range information and excessive caution with very strong hands as one related range-model change.", "Retained as parent of v3. The large benchmark improvement repeated on seed 1, and it beat v1 directly by +51.93 bb/100 (SE 8.84)."),
    3: ("Keeps v2's range model. Bets 1.25 pots when adjusted equity is at least 0.90, and otherwise retains the 0.66-pot sizing.", "Test whether very strong hands earn more with larger value bets rather than using one size for every hand.", "Retained as parent of v4. Improvements against Alan repeated on the larger second-seed run; the direct result against v2 was a tie within error, not evidence of dominance."),
    4: ("Removes an opponent's unmatched overbet from the pot used for calling odds. Otherwise uses v3's decisions and sizing.", "Fix a chip-accounting issue when the hero has fewer chips than the opponent's wager.", "Retained as parent of v5 and v6 for the accounting correction. Score differences were small and are not claimed as statistically meaningful."),
    5: ("Adds a 15% bluff chance with less than 0.35 raw equity when checked to on the button. Medium-strength showdown hands remain in the checking range; value betting is unchanged.", "Test a small positional bluff component against the value-only parent, without recognizing opponent identities or exact bet sizes.", "Retained as parent of v9. The development gain was small and did not establish a clear advantage; v9 applies the all-in correction before final evaluation."),
    6: ("Uses v4's value-only strategy, but removes the realization penalty if calling leaves no future betting. Keeps the unmatched-overbet correction and effective-stack sizing.", "Do not discount showdown equity for future betting decisions that cannot occur after an all-in call.", "Finalist; retained as parent of v7 and v8. The change is a strategic correction, with little measured development impact."),
    7: ("Gives opponent overbets additional range-strength weight according to the effective calling odds. All other decisions match v6.", "Test generic bet-size information rather than treating a small raise and a large overbet as equally informative.", "Discarded. Its four seed-0 finals results exactly matched v6, and its parent match showed no advantage. The extra parameter had no demonstrated benefit."),
    8: ("Uses exactly v6's strategy with 256 Monte Carlo samples instead of 128. This consumes the full allowed sampling budget.", "Test whether lower equity-estimation noise improves strength enough to justify twice the candidate sampling work.", "Discarded before held-out evaluation. The development mean did not improve, and the parent comparison was within error; 128 samples is the simpler compute choice."),
    9: ("Combines v5's positional bluffing with full equity realization when calling ends betting. Inlines the final legality conjunction to keep the decision method at 55 readable lines without changing that check.", "Apply the separately tested all-in correction to the bluffing branch, then compare frozen candidates.", "Finalist; no descendant and no strategy changes after the evaluation lock."),
}
UNITS = {
    'num_samples': 'Monte Carlo samples per decision',
    'button_open_equity': 'raw equity probability',
    'isolation_equity': 'raw equity probability',
    'preflop_value_equity': 'adjusted equity probability',
    'postflop_value_equity': 'adjusted equity probability',
    'reraise_margin_equity': 'equity probability increment when facing a bet',
    'raise_discount_fraction': 'equity fraction removed per current-street opposing raise',
    'range_exclusion_fraction': 'weak-range fraction removed per raise-equivalent',
    'prior_raise_weight': 'current-street raise-equivalents per earlier raise',
    'call_margin_equity': 'equity probability above break-even call odds',
    'realization_fraction': 'fraction of adjusted equity before possible future betting',
    'opening_size_bb': 'big blinds (street raise-to total)',
    'bet_pot_fraction': 'fraction of contestable pot after calling',
    'preflop_reraise_multiple': 'multiple of the current street bet',
    'premium_equity': 'adjusted equity probability',
    'premium_pot_fraction': 'fraction of contestable pot after calling',
    'bluff_frequency': 'probability per eligible positional bluff opportunity',
    'bluff_max_equity': 'maximum raw equity probability eligible to bluff',
    'pressure_reference_odds': 'call equity probability; 1/3 corresponds to a pot bet',
}
PATTERN = re.compile(r'(v[0-9]+):Bot vs\s+(\S+):\s+([+-][0-9.]+) bb/100  \(SE ([0-9.]+), ([0-9]+) hands, seed ([0-9]+), ([0-9]+)s\)')


def read_results():
    rows = []
    paths = list(Path('.').glob('benchmark_*.log')) + list(Path('.').glob('smoke_*.log'))
    for path in sorted(paths):
        match_tables = re.search(r'_t([0-9]+)', path.name)
        tables = int(match_tables[1]) if match_tables else (2 if path.name.startswith('smoke_') else 64)
        for match in PATTERN.finditer(path.read_text()):
            version, opponent, score, se, hands, seed, seconds = match.groups()
            rows.append(dict(version=int(version[1:]), opponent=opponent, score=float(score), se=float(se),
                             hands=int(hands), seed=int(seed), seconds=int(seconds), tables=tables, log=path.name))
    return rows


def table(rows):
    lines = ['| Opponent | Seed | Tables × hands | bb/100 | SE |', '|---|---:|---:|---:|---:|']
    for row in sorted(rows, key=lambda r: (r['seed'], r['tables'], r['opponent'])):
        per_table = row['hands'] // row['tables']
        lines.append(f"| {row['opponent']} | {row['seed']} | {row['tables']} × {per_table} | {row['score']:+.2f} | {row['se']:.2f} |")
    return lines


def main():
    rows = read_results()
    finals = {path.stem for path in Path('opponents').glob('*.py')}
    held = {v: [r for r in rows if r['version'] == v and r['seed'] == 1000 and r['opponent'] in finals] for v in (6, 9)}
    assert all(len(held[v]) == 4 for v in held), 'Wait for complete held-out finals results'
    for version in (6, 9):
        observed = {r['opponent'] for r in rows if r['version'] == version and r['seed'] == 1000}
        expected = finals | {'random', f'v{version}:Bot'} | {f'v{k}:Bot' for k in range(1, version)}
        assert expected <= observed, f'Incomplete held-out benchmarks for v{version}'
    means = {v: float(re.search(r'mean vs finals:\s+([+-][0-9.]+) bb/100', Path(f'benchmark_v{v}_finals_s1000_t512.log').read_text())[1]) for v in held}
    se_bounds = {v: sum(r['se'] for r in held[v]) / 4 for v in held}
    difference_bound = se_bounds[6] + se_bounds[9]
    selected = max(means, key=means.get) if abs(means[9] - means[6]) > 2 * difference_bound else 6
    assert sys.argv[1] == f'v{selected}', 'Requested deployment disagrees with the frozen selection rule'
    lock = json.loads(Path('evaluation_lock.json').read_text())
    start = datetime.fromisoformat(lock['recorded_start_utc'])
    finish = datetime.now(timezone.utc)
    local = timezone(timedelta(hours=10, minutes=30))
    minutes = (finish - start).total_seconds() / 60
    frozen_minutes = (datetime.fromisoformat(lock['frozen_utc']) - start).total_seconds() / 60
    header = [
        '# hadrian_marlowe_gpt', '',
        '- **Lineage:** `phase_2_suneater/hadrian_marlowe_gpt`',
        '- **Model:** GPT-6 (Codex; exact serving variant unavailable)',
        '- **Prompt:** `botgen-v2`',
        '- **Engine:** `02abc49` + working tree, `engine_sha256 295477eff7637056`',
        '- **Execution:** All nine versions compiled and were benchmarked on the machine holding this directory, using the existing project Python environment. No packages were installed.',
        f'- **Recorded wall-clock time:** {minutes:.1f} minutes, {start.astimezone(local):%Y-%m-%d %H:%M:%S} to {finish.astimezone(local):%H:%M:%S} Australia/Adelaide. The attachment and initial directory listing preceded the recorded start by approximately one minute.',
        f'- **Iteration stopped:** {frozen_minutes:.1f} minutes after the recorded start, before the 50-minute deadline; a local 50-minute reminder was started. Only evaluation, documentation and verification followed the freeze.',
        f'- **Deployment:** `from .v{selected} import Bot as DeployedBot`', '',
        '## Protocol, evidence and limitations', '',
        'Every benchmark number below comes from a saved local log. Development used seeds 0–2; seed 1000 was reserved for frozen evaluation. Each finalist was run exactly once against each phase-1 final and RandomBot on seed 1000 at 512 tables × 250 hands, and once against every earlier version at 128 × 250. The primary score is the unweighted mean against the four phase-1 finals; RandomBot never enters that score.', '',
        'The provided benchmark balances button position and reports table-clustered standard errors. Individual SEs are reported as printed. A mean across opponents is not assigned an independence-based SE because the seed/table streams are shared. For selection, the average marginal SE bounds each candidate mean SE by the triangle inequality; the sum of those bounds conservatively bounds uncertainty in their difference. This is an uncertainty heuristic using estimated SEs, not a multiple-comparison-adjusted significance test. The selection rule was saved in `evaluation_lock.json` before reading held-out results.', '',
        '**Disclosed deviations:** The first discovery command listed filenames outside this bot directory while the task attachment was being read. No engine source, other-lineage source, or git state was opened or used. All subsequent project reads and writes stayed in this directory, apart from invoking the explicitly allowed Python environment. Also, v1 has 43 decision lines, two below the nominal 45–55 range; it was preserved once benchmarked instead of silently edited. Versions v2–v9 are within range. The initial plain `uv run` hit a cache permission error; later invocations used an offline, no-sync cache inside this directory.', '',
        'All versions are standalone, stateless classes: `init_state` returns `None`, `act` returns the supplied state unchanged, and none defines `observe`. They import only JAX, the documented `poker_bots` contract and the shared equity helpers. Constructor defaults are reasoning choices, not fitted lookup tables. Versioned changes are frozen; their hashes are in `version_hashes.json`.', '',
        '## Ideas borrowed from the supplied opponents', '',
        'I read only the four frozen opponents supplied here. Alan suggested discounting equity after opponent aggression; Max supplied the general realization-discount and explicit-opening-size ideas; Jesse illustrated pot-odds/value-betting structure; Jack suggested mixed bluffing and stronger-hand sizing. The new policy uses ordinary within-hand raises, position, pot odds and effective stacks. It never identifies a bot, matches a particular opponent bet size, or remembers prior hands.', '',
    ]
    for version in range(1, 10):
        bot = load(f'v{version}:Bot')
        metadata = bot.metadata()
        strategy, hypothesis, verdict = STRATEGIES[version]
        source = Path(f'v{version}.py').read_text()
        method = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == 'act')
        code_lines = sum(bool(line.strip()) and not line.strip().startswith('#') for line in source.splitlines()[method.lineno - 1:method.end_lineno])
        header += [f'## v{version}', '', f'**What it does:** {strategy}', '', f'**Why it exists:** {hypothesis}', '',
                   f'**Parent:** {metadata["parent"]}. **Decision line count:** {code_lines} including the `act` declaration ({code_lines - 1} body lines); no extra decision helpers.', '',
                   '**Parameters:** All unchanged defaults are inherited reasoning choices. No grid search or automated fitting was performed.', '',
                   '| Parameter | Value | Unit | Choice |', '|---|---:|---|---|']
        parent_params = load(f'v{metadata["parent"]}:Bot').metadata()['params'] if metadata['parent'] else {}
        for name, value in metadata['params'].items():
            origin = 'Inherited reasoning default' if parent_params.get(name) == value else 'Reasoning default tested in this version'
            if version == 8 and name == 'num_samples':
                origin = 'Compute-cap experiment against v6; not retained'
            header.append(f'| `{name}` | {value:g} | {UNITS[name]} | {origin} |')
        header += ['', '**Results:**', ''] + table([r for r in rows if r['version'] == version])
        groups = {}
        for row in rows:
            if row['version'] == version and row['opponent'] in finals:
                groups.setdefault((row['seed'], row['tables']), []).append(row)
        for (seed, tables), group in sorted(groups.items()):
            assert len(group) == 4
            mean = float(re.search(r'mean vs finals:\s+([+-][0-9.]+) bb/100', Path(group[0]['log']).read_text())[1])
            header += ['', f'Mean against finals, seed {seed}, {tables} tables: **{mean:+.2f} bb/100**.']
        if version == selected:
            verdict += ' Selected for deployment by the frozen held-out rule.'
        elif version in (6, 9):
            verdict += ' Not deployed under the frozen held-out rule.'
        header += ['', f'**Verdict:** {verdict}', '']
    header += ['## Final choice', '', f'**Deploy v{selected}.** Held-out mean: **{means[selected]:+.2f} bb/100** against the four finals.', '',
               f'The held-out means were v6 **{means[6]:+.2f}** and v9 **{means[9]:+.2f}** bb/100. Their difference (v9 − v6) was **{means[9] - means[6]:+.2f}**. The predeclared conservative two-SE difference threshold was **{2 * difference_bound:.2f} bb/100**.']
    if selected == 6:
        header += ['', 'The difference was within that uncertainty threshold, so the predeclared rule chose the value-only policy with two fewer parameters. V6 also had the higher held-out point estimate. V9 led the first-seed development results, but its apparent advantage did not repeat on held-out evaluation. No parameters were changed after seeing seed 1000.']
    else:
        header += ['', 'The difference exceeded the predeclared uncertainty threshold, so the higher-scoring bluffing policy was selected. No parameters were changed after seeing seed 1000.']
    header += ['', '### Deployed version: all held-out benchmarks', ''] + table([r for r in rows if r['version'] == selected and r['seed'] == 1000])
    parent = load(f'v{selected}:Bot').metadata()['parent']
    parent_row = next(r for r in rows if r['version'] == selected and r['seed'] == 1000 and r['opponent'] == f'v{parent}:Bot')
    controls = [r for r in rows if r['seed'] == 1000 and r['opponent'] == f"v{r['version']}:Bot"]
    header += ['', '### Self-play control for the parent-match warning', ''] + table(controls)
    control = next(r for r in controls if r['version'] == selected)
    header += ['', f'The deployed policy scored {control["score"]:+.2f} bb/100 (SE {control["se"]:.2f}) against an identical copy on the same seed and table count. This control has zero true policy advantage by symmetry. Compare it with the parent result below: a similar negative value suggests shared-deal noise rather than evidence that the change caused a regression. These controls were added after noticing the repeated losses, are diagnostic only, and did not change the frozen selection rule. They are not used to subtract a correction from any published score.']
    header += ['', f'Direct held-out result against parent v{parent}: **{parent_row["score"]:+.2f} bb/100 (SE {parent_row["se"]:.2f})**. Interpret small differences as noise; a positive finals score does not establish strength against unseen phase-2 bots.', '',
               '### Known weaknesses', '',
               '- The range correction treats raw showdown equity approximately like a hand-strength percentile; it is not a calibrated posterior over opponent holdings. It can misjudge shared-board ties, draws, blockers and unusual betting ranges.',
               '- Preflop opens and re-raises use noisy random-hand equity rather than solved ranges. There is no opponent learning, no equilibrium solver and no action-history model beyond aggregate aggression.',
               '- v6 has no deliberate postflop bluffs. Its sizing can reveal strength to an adaptive opponent; v9 adds only a small heuristic positional bluff component.',
               '- The conservative range correction can overfold against frequent bluffers. Realization and betting thresholds are simple fixed heuristics, and prior raises are not distinguished by size or detailed sequence.',
               '- Success against four phase-1 bots and close relatives is limited evidence about independent phase-2 opponents. All tests are heads-up; the three-seat contract smoke check is not strategic multiway validation.', '',
               '## Verification and reproduction', '',
               'The contract checker exercises every street under `jit` and `vmap`, scalar int32 action fields, legal raise bounds, free checks, disabled raising, unchanged state, JSON metadata and a non-hard-coded seat dimension. Versions v4 onward also pass the unmatched-overbet calling-invariance regression. The protected reference hashes match the originals, and all version hashes match the pre-held-out freeze. Simulation smoke tests and all logged benchmarks completed without an exception.', '',
               'Decision-line counts include the one-line method declaration and exclude comments/blank lines. No version defines another decision helper. Numeric 0/1 identities, card/street identifiers and the 1e-6 division guard are structural or numerical safeguards rather than strategy knobs. Monte Carlo counts are 128 in all deployed candidates and 256 only in v8, with constructor validation rejecting counts outside 1–256.', '',
               'Run from this directory using the existing environment:', '', '```sh',
               f'uv run --offline --no-sync --cache-dir .uv-cache python bench.py v{selected}:Bot finals --seed 0',
               'uv run --offline --no-sync --cache-dir .uv-cache python check_contract.py v1 v2 v3 v4 v5 v6 v7 v8 v9', '```', '',
               '### Retained scratch artifacts', '',
               '- `benchmark_*.log`: unedited development and held-out benchmark output; names encode candidate, opponent group, seed, and nondefault table count. Default runs use 64 × 250.',
               '- `smoke_v1.log`: initial 2 × 20 import/compilation check, too small for any strength inference.',
               '- `check_contract.py` and `contract_*.log`: reproducible contract checks and their outputs.',
               '- `reference_hashes.json`, `version_hashes.json`, `evaluation_lock.json`: reference integrity, frozen source integrity and selection protocol.',
               '- `write_readme.py`: generates this report from logs and each version metadata; it does not modify version code.',
               '- `final_static_audit.log`: import, purity, single-class and frozen-source audit; a comment-punctuation false positive in the first checker was corrected without editing bot code.',
               '- `experiment_summary.json`: machine-readable selected version, held-out scores, runtime and benchmark totals.',
               '- `.uv-cache/` and `__pycache__/`: local runtime caches.', '',
               'The source references `equity.py`, `bench.py` and `opponents/` were not edited. No repository commits were made.', '']
    Path('README.md').write_text('\n'.join(header))
    summary = dict(selected=f'v{selected}', heldout_means=means, conservative_two_se_difference=2 * difference_bound,
                   recorded_elapsed_minutes=minutes, finish_utc=finish.isoformat(), total_logged_hands=sum(r['hands'] for r in rows),
                   benchmark_matchups=len(rows), version_count=9)
    Path('experiment_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
