"""Build the audit report from saved benchmark outputs; not bot decision code."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from bench import load

CORE = ['aggressor', 'bassander_lin_glm', 'hadrian_marlowe_gpt', 'valka_onderra_claude']
LEGACY = ['alan_wake_claude', 'jack_joyce_gemini', 'jesse_faden_glm', 'max_payne_gpt']
STORIES = {
    1: ('Open a wide button range, isolate strong hands, and call using equity discounted for current-hand aggression. Bet for value and mix weak-hand bluffs when in position or holding initiative. Cap raises at effective stacks and account for unmatched chips in calling odds.', 'Sound stateless heads-up baseline. Range exclusion and effective-stack handling borrow ideas from hadrian_marlowe_gpt; the mixed bluffing is new.', 'Kept as the parent of v2, v3, v5 and v6, and as a reference opponent. Larger seed-1 results confirmed its positive core mean and its aggressor weakness.'),
    2: ('Use the baseline strategy but allow the same weak-hand bluff frequency from either position. This adds leads from the big blind without requiring earlier initiative.', 'Test whether giving away too many out-of-position checks causes the aggressor loss.', 'Discarded: the core mean fell, particularly against bassander; the parent result was inconclusive. No statistically established overall improvement justified promotion.'),
    3: ('Distinguish the initial preflop opener from a player re-raising our bet. Add extra range pressure when facing a raise after our own bet; retain the baseline opening, bluffing and sizing rules.', 'The baseline softens the first opposing preflop raise even when it is actually a three-bet. This version tests more cautious interpretation of re-raises.', 'Kept as the parent of v4, then discarded as a finalist. Two core seeds did not show an overall gain over v1, although results against hadrian improved.'),
    4: ('Keep the re-raise distinction from v3, but reduce the pressure attributed to ordinary postflop bets and previous aggression. Continue to apply the extra penalty for a raise of our own bet.', 'Test broader bluff-catching without abandoning caution against re-raises.', 'Discarded: lower core performance and a negative parent match gave no support for the looser defense.'),
    5: ('Use one Monte Carlo batch for both random-hand equity and opponent-range equity. Rank sampled opponent holdings using only the dealt board, give weak holdings less weight, and retain each sample’s future drawing outcome. Keep the baseline scalar discount preflop.', 'Test whether scalar equity discounts throw away too much live draw equity. The idea of conditioning sampled ranges on the visible board is inspired by valka_onderra_claude, but this implementation uses made-hand percentiles rather than its pair/miss rule.', 'Discarded: markedly lower core mean and no parent improvement. The extra range-model complexity was not supported by the measurements.'),
    6: ('Expand value re-raises preflop and mix in some weaker suited hands or ace blockers against a single opening raise. Keep the baseline postflop strategy and bet sizing.', 'Contest wide opening ranges before the flop, rather than relying entirely on postflop bluff-catching.', 'Kept as the parent of v7, v8 and v9, and as a held-out finalist. The core mean was similar on two seeds; both training parent matches were positive, though individually noisy.'),
    7: ('Keep v6’s hand selection and sizing but increase the frequency of its eligible preflop bluff re-raises. Other decisions remain the same.', 'Test a more polarized, aggressive response to wide opens.', 'Discarded as a finalist: improved aggressor balance on the first seed did not improve the overall core mean, and the parent result was uncertain. The mixed-checking branch offered a better-supported balance.'),
    8: ('Keep v6’s betting ranges but size ordinary postflop bets at 40% of the pot. Retain the larger premium-hand sizing.', 'Test whether cheaper bets preserve sufficient fold equity to offset reduced value extraction.', 'Discarded: lower core mean and an essentially even parent match did not establish a benefit.'),
    9: ('Keep v6’s wider preflop strategy. Before the river, occasionally check value hands out of position, so checking does not always announce a weak or middling hand. Continue normal value betting in position and on the river.', 'Protect the checking range and induce bets from frequent bettors, while allowing more than weak hands to reach a check-call line.', 'Kept as the parent of v10 and v11 and as a held-out finalist. Its roughly +11 bb/100 training aggressor result repeated on a larger second seed without reducing the core mean materially.'),
    10: ('Use v9’s postflop plan with cheaper two-big-blind preflop opens and an almost-all-hands button range. The common opening size also applies when isolating a limp.', 'Test a coherent wider, cheaper opening plan that risks fewer chips with the bottom of the range.', 'Discarded: lower initial core mean and no parent win did not justify replacing v9. This is not a claim that the small observed difference is statistically decisive.'),
    11: ('Use the same decisions and parameters as v9, but estimate equity with 256 rather than 128 Monte Carlo deals. This uses the full allowed sampling budget to reduce decision noise.', 'Test whether more precise equity estimates help around call and raise thresholds.', 'Retained as a held-out finalist. Core training results were close to v9, so no small one-seed gain was treated as proof; the second seed and parent matches were also recorded.'),
}
UNITS = {
    'num_samples': 'Monte Carlo deals per decision',
    'open_equity': 'equity probability versus random',
    'isolation_equity': 'equity probability versus random',
    'preflop_raise_equity': 'adjusted equity probability',
    'value_equity': 'adjusted equity probability',
    'raise_equity': 'adjusted equity probability',
    'open_size_bb': 'raise-to big blinds',
    'reraise_multiple': 'multiple of the current bet',
    'bet_pot_fraction': 'fraction of the pot after calling',
    'premium_equity': 'adjusted equity probability',
    'premium_pot_fraction': 'fraction of the pot after calling',
    'raise_exclusion_fraction': 'weakest-range fraction excluded per pressure unit',
    'prior_raise_weight': 'current-street raise equivalents',
    'opening_raise_weight': 'ordinary-raise equivalents for the initial opposing preflop raise',
    'realization_fraction': 'fraction of equity realized with future betting',
    'call_margin_equity': 'equity probability above pot odds',
    'bluff_frequency': 'probability for eligible weak hands',
    'bluff_ceiling_equity': 'equity probability versus random',
    'reraise_extra_weight': 'additional raise-pressure equivalents',
    'postflop_pressure_weight': 'multiplier of ordinary postflop pressure',
    'weak_range_weight': 'relative likelihood of an excluded weak holding',
    'threebet_bluff_frequency': 'probability for eligible preflop bluff hands',
    'threebet_bluff_ceiling_equity': 'equity probability versus random',
    'trap_frequency': 'probability of checking a value hand out of position before the river',
}
CHANGE_REASONS = {
    3: 'reasoned re-raise experiment; core seeds 0 and 1',
    4: 'reasoned wider-defense trial; core seed 0',
    5: 'reasoned weak-range mixture; core seed 0',
    6: 'reasoned wider/polarized preflop trial; core seeds 0 and 2',
    7: 'large frequency contrast; core seed 0, not fine grid tuning',
    8: 'smaller-bet hypothesis; core seed 0',
    9: 'reasoned checking-range protection; core seeds 0 and 2',
    10: 'reasoned cheaper/wider opening plan; core seed 0',
    11: 'maximum legal sample budget; core seeds 0 and 2',
}


def read_results():
    pattern = re.compile(r'(v\d+:Bot) vs\s+(\S+):\s+([+-]?\d+\.\d+) bb/100\s+\(SE (\d+\.\d+), (\d+) hands, seed (\d+), (\d+)s\)')
    rows = []
    means = {}
    for path in sorted(Path('.').glob('*.log')):
        content = path.read_text()
        matches = list(pattern.finditer(content))
        for match in matches:
            candidate, opponent, score, se, hands, seed, seconds = match.groups()
            hands = int(hands)
            rows.append(dict(candidate=candidate, opponent=opponent, bb100=float(score), se=float(se), hands=hands,
                             seed=int(seed), seconds=int(seconds), tables=2 if hands == 40 else hands // 250,
                             hands_per_table=20 if hands == 40 else 250, log=path.name))
        for candidate, group, value in re.findall(r'(v\d+:Bot) mean vs (core|legacy):\s+([+-]?\d+\.\d+) bb/100', content):
            seed = int(matches[0].group(6))
            means.setdefault(candidate, {}).setdefault(str(seed), {})[group] = float(value)
    Path('results.json').write_text(json.dumps({'results': rows, 'means': means}, indent=2) + '\n')
    return rows, means


def table(rows):
    output = ['| Opponent | Seed | Tables × hands | bb/100 | SE |', '|---|---:|---:|---:|---:|']
    for row in rows:
        name = row['opponent'] + (' (smoke only)' if row['hands'] == 40 else '')
        output.append(f"| {name} | {row['seed']} | {row['tables']} × {row['hands_per_table']} | {row['bb100']:+.2f} | {row['se']:.2f} |")
    return output


def main():
    rows, means = read_results()
    plan = json.loads(Path('evaluation_plan.json').read_text())
    deployment = json.loads(Path('deployment.json').read_text())
    audit = json.loads(Path('source_audit.json').read_text())
    final = deployment['version']
    now = datetime.now(timezone.utc)
    elapsed = int((now - datetime.fromisoformat(plan['start_utc'])).total_seconds())
    freeze_elapsed = int((datetime.fromisoformat(plan['freeze_utc']) - datetime.fromisoformat(plan['start_utc'])).total_seconds())
    final_rows = [row for row in rows if row['candidate'] == f'v{final}:Bot' and row['seed'] == 1000]
    for opponent in CORE + LEGACY + ['random'] + [f'v{n}:Bot' for n in range(1, 12) if n != final]:
        assert sum(row['opponent'] == opponent for row in final_rows) == 1, ('Missing or duplicate final comparison', opponent)
    assert Path('__init__.py').read_text() == f'from .v{final} import Bot as DeployedBot\n'
    out = [
        '# invoker_gpt', '',
        '- **Lineage:** `phase_3_dota/invoker_gpt`',
        '- **Model:** GPT-6 (Codex; exact serving variant unavailable)',
        '- **Prompt:** `botgen-v3`',
        '- **Engine:** `02abc49` + working tree; `engine_sha256 295477eff7637056`',
        '- **Code executed:** yes; all versions passed their smoke test and the contract checks.',
        f'- **Deployed:** `v{final}:Bot`, exported as `DeployedBot` from `__init__.py`.',
        f'- **Wall-clock time used:** {elapsed // 60} minutes {elapsed % 60} seconds through report generation.',
        f'- **Run start:** {plan["start_utc"]} (14 October 2026, 03:45:56 Australia/Adelaide).',
        f'- **Strategy freeze:** {plan["freeze_utc"]}, after {freeze_elapsed // 60} minutes {freeze_elapsed % 60} seconds; before the 50-minute stop point.',
        f'- **Report generated:** {now.isoformat()}.', '',
        '## Environment issue and successful check', '',
        'The initial session and the first retry after repository access was granted both failed before executing the check, because the shell sandbox could not open the existing uv cache. No bot code was written during the initial failed session. The successful run used elevated execution from this bot directory to access the existing environment; no packages were installed or environment files deliberately changed.', '',
        'Exact failed output:', '', '```text',
        'error: Failed to initialize cache at `/Users/alik/.cache/uv`',
        '  Caused by: failed to open file `/Users/alik/.cache/uv/sdists-v9/.git`: Operation not permitted (os error 1)',
        '```', '', 'Successful `uv run python bench.py check` output:', '', '```text',
        '\n'.join(Path('check.log').read_text().splitlines()[2:]), '```', '',
        'The wall-clock budget above starts with the resumed attempt; it excludes the earlier stopped attempt and the wait for the requester. Clock checks were made between experiments. All intentional file reads and writes for the bot were confined to this directory, apart from the explicitly authorized use of the project Python environment and the requester-provided task attachment. No engine source, other lineages, git state, or synced project sources were inspected or modified. The shared helpers, benchmark script and frozen opponents were not edited.', '',
        '## Protocol and interpretation', '',
        'Training used seeds 0, 1 and 2 only. Each new version received a 2 × 20-hand RandomBot smoke test. Initial core runs used 64 × 250 hands per opponent; selected comparisons were repeated at 128 × 250. The first core run reported approximately 58 seconds across its four matchups, excluding process overhead.', '',
        'Finalists v6, v9 and v11 were fixed in `evaluation_plan.json` before any held-out results were inspected. Each ran `all`, RandomBot and its parent once on seed 1000 at 256 × 250 hands per opponent. The deployed version was also evaluated once against every other saved version on that seed. The additional comparisons used 128 × 250 hands; the parent comparison retained 256 × 250. No strategy or parameter changes followed the freeze. Seed 1000 is a selection set as requested, so the selected score is not a fresh, untouched post-selection estimate.', '',
        'All SEs below are the benchmark’s table-clustered standard errors, not confidence intervals. Core and legacy means are unweighted means over their four opponents; the benchmark does not report an aggregate SE. Small differences were treated as uncertain. RandomBot smoke outcomes are compatibility checks, not strength evidence. The full raw outputs are preserved in the named `.log` files, and `results.json` contains their parsed values.', '',
        '## Final choice', '', deployment['reason'], '',
        '| Finalist | Held-out core mean | Held-out legacy mean |', '|---|---:|---:|',
    ]
    for number in plan['finalists']:
        result = means[f'v{number}:Bot']['1000']
        out.append(f"| v{number} | {result['core']:+.2f} | {result['legacy']:+.2f} |")
    out += ['', '### Deployed version: held-out results', '']
    order = CORE + LEGACY + ['random'] + [f'v{n}:Bot' for n in range(1, 12) if n != final]
    out += table(sorted(final_rows, key=lambda row: order.index(row['opponent'])))
    out += ['', '### Known weaknesses', '',
            '- The deployed range model is a heuristic transformation of equity against random cards. It can undervalue live draws and does not fully model opponent calls, board texture or bet-size-dependent ranges.',
            '- The first opposing preflop raise receives partial pressure even if it is a three-bet of our opening raise. Version 3 tested stricter treatment, but did not establish an overall benchmark improvement.',
            '- The bot cannot adapt across hands, and its fixed opening, bluffing and trapping frequencies can be exploited by an opponent designed to counter them.',
            '- Premium sizing reveals extra strength. Equity sampling adds decision noise; the sample count and computation cap are explicit.',
            '- Positive means in individual matchups are not automatically statistically established wins. These benchmarks do not establish strength against the independently developed phase-3 finals.', '',
            '## Version history', '',
            'Every listed parameter is a constructor argument; values are defaults actually benchmarked. Unchanged parameters are explicitly marked as inherited. Changes were selected as interpretable hypotheses, not fitted lookup tables or opponent-specific action signatures. The decision-line count conservatively includes method signature lines and nested helper lines, while excluding blank lines, comments, imports, constructors, initialization and metadata.', '']
    for number in range(1, 12):
        bot = load(f'v{number}:Bot')
        metadata = bot.metadata()
        parent = metadata['parent']
        parent_params = load(f'v{parent}:Bot').metadata()['params'] if parent else {}
        what, why, verdict = STORIES[number]
        if number == final:
            verdict += ' Deployed after held-out selection; see the final-choice rationale above.'
        elif number in plan['finalists']:
            verdict += ' Not deployed; see the final-choice comparison above.'
        out += [f'### v{number} (parent: ' + (f'v{parent}' if parent else 'none') + ')', '',
                '**What it does:** ' + what, '', '**Why it exists:** ' + why, '',
                '**Parameters:**', '', '| Parameter | Value | Unit | Choice |', '|---|---:|---|---|']
        for name, value in metadata['params'].items():
            choice = 'reasoned baseline default; not fitted' if parent is None else f'inherited unchanged from v{parent}'
            if parent and (name not in parent_params or parent_params[name] != value):
                choice = CHANGE_REASONS[number]
            out.append(f'| `{name}` | {value} | {UNITS[name]} | {choice} |')
        version_rows = [row for row in rows if row['candidate'] == f'v{number}:Bot']
        version_rows.sort(key=lambda row: (row['seed'], row['hands'] != 40, row['opponent'], row['tables']))
        out += ['', '**Results:**', ''] + table(version_rows)
        version_means = means.get(f'v{number}:Bot', {})
        for seed, groups in sorted(version_means.items(), key=lambda item: int(item[0])):
            out += ['', f"Seed {seed} group means: " + '; '.join(f'{group} {score:+.2f} bb/100' for group, score in groups.items()) + '.']
        if not any(row['opponent'] in LEGACY for row in version_rows):
            out += ['', 'Legacy: not run for this discarded candidate.']
        out += ['', '**Verdict:** ' + verdict, '', f"**Decision-code line count:** {audit[f'v{number}.py']['decision_lines']}.", '']
    out += ['## Verification and saved files', '',
            '`verification.log` records successful checks of all 11 versions: one `Bot` class, no `observe`, complete JSON-serializable parameter metadata, valid sample limits, immutable source fingerprints, `None` state, JIT/vmap execution, int32 scalar action fields, no folding a free check, no raising when prohibited, and bounded raise sizes. Each version also completed real engine smoke tests and core benchmarks.', '',
            '`source_audit.json` contains decision-line counts and SHA-256 fingerprints. `check.log` preserves the successful environment check. `evaluation_plan.json` records the precommitted finalist list and freeze time. `deployment.json` records the final selection. `results.json` is generated from raw benchmark logs. `verify.py` and `build_readme.py` are local audit/report utilities and are not imported by any bot. All these scratch and audit files stay in the bot directory.', '',
            'The one-line `__init__.py` is the deployment entry point. Version files are independent and never import another version. No version was edited after it was benchmarked.', '']
    Path('README.md').write_text('\n'.join(out))
    print(f'Wrote README.md for v{final}; {len(rows)} recorded matchups; elapsed {elapsed // 60}m {elapsed % 60}s.')


if __name__ == '__main__':
    main()
