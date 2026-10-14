# invoker_gpt

- **Lineage:** `phase_3_dota/invoker_gpt`
- **Model:** GPT-6 (Codex; exact serving variant unavailable)
- **Prompt:** `botgen-v3`
- **Engine:** `02abc49` + working tree; `engine_sha256 295477eff7637056`
- **Code executed:** yes; all versions passed their smoke test and the contract checks.
- **Deployed:** `v9:Bot`, exported as `DeployedBot` from `__init__.py`.
- **Wall-clock time used:** 40 minutes 35 seconds through report generation.
- **Run start:** 2026-10-13T17:15:56+00:00 (14 October 2026, 03:45:56 Australia/Adelaide).
- **Strategy freeze:** 2026-10-13T17:39:10.168987+00:00, after 23 minutes 14 seconds; before the 50-minute stop point.
- **Report generated:** 2026-10-13T17:56:31.367471+00:00.

## Environment issue and successful check

The initial session and the first retry after repository access was granted both failed before executing the check, because the shell sandbox could not open the existing uv cache. No bot code was written during the initial failed session. The successful run used elevated execution from this bot directory to access the existing environment; no packages were installed or environment files deliberately changed.

Exact failed output:

```text
error: Failed to initialize cache at `/Users/alik/.cache/uv`
  Caused by: failed to open file `/Users/alik/.cache/uv/sdists-v9/.git`: Operation not permitted (os error 1)
```

Successful `uv run python bench.py check` output:

```text
python      /Users/alik/Documents/Projects/poker_bots/poker_bots/.venv/bin/python3 (3.13.12)
jax         0.11.2 (cpu)
poker_bots  /Users/alik/Documents/Projects/poker_bots/poker_bots/src/poker_bots
engine      sha256 295477eff7637056 (matches the contract)
smoke test  random vs aggressor, 40 hands in 2s: OK
```

The wall-clock budget above starts with the resumed attempt; it excludes the earlier stopped attempt and the wait for the requester. Clock checks were made between experiments. All intentional file reads and writes for the bot were confined to this directory, apart from the explicitly authorized use of the project Python environment and the requester-provided task attachment. No engine source, other lineages, git state, or synced project sources were inspected or modified. The shared helpers, benchmark script and frozen opponents were not edited.

## Protocol and interpretation

Training used seeds 0, 1 and 2 only. Each new version received a 2 × 20-hand RandomBot smoke test. Initial core runs used 64 × 250 hands per opponent; selected comparisons were repeated at 128 × 250. The first core run reported approximately 58 seconds across its four matchups, excluding process overhead.

Finalists v6, v9 and v11 were fixed in `evaluation_plan.json` before any held-out results were inspected. Each ran `all`, RandomBot and its parent once on seed 1000 at 256 × 250 hands per opponent. The deployed version was also evaluated once against every other saved version on that seed. The additional comparisons used 128 × 250 hands; the parent comparison retained 256 × 250. No strategy or parameter changes followed the freeze. Seed 1000 is a selection set as requested, so the selected score is not a fresh, untouched post-selection estimate.

All SEs below are the benchmark’s table-clustered standard errors, not confidence intervals. Core and legacy means are unweighted means over their four opponents; the benchmark does not report an aggregate SE. Small differences were treated as uncertain. RandomBot smoke outcomes are compatibility checks, not strength evidence. The full raw outputs are preserved in the named `.log` files, and `results.json` contains their parsed values.

## Final choice

Deploy **v9**, the 128-sample mixed-checking version. It scored **+41.03 bb/100 against core** and **+159.33 against legacy** on seed 1000, with positive point estimates against every core opponent. Its aggressor result was **+11.74 ± 3.49**, consistent with +11.59 ± 6.12 and +11.06 ± 4.73 on the two training seeds. Its parent result was **+3.82 ± 5.58**: inconclusive, rather than evidence of regression.

**The numerically best core finalist was v11 (+41.83), but it is not deployed.** Its 0.80 bb/100 advantage over v9 is much smaller than the per-match sampling errors, the training difference changed sign across seeds, and v9 versus v11 was −0.80 ± 7.58. The two versions implement the same strategy; v11 doubles the equity sample count from 128 to 256. There is no established gain warranting that extra sampling, so retain the less expensive parent rather than promote a noisy improvement. V6 had a lower held-out core mean (+37.05) and retained a negative aggressor point estimate (−3.82 ± 2.97). The direct version comparisons do not establish that v9 dominates all alternatives, and the result against hadrian (+5.57 ± 4.13) remains uncertain.

The separate v11-versus-v9 parent evaluation was also inconclusive (+1.46 ± 5.46 for v11 at 64,000 hands); it does not overturn the choice.

| Finalist | Held-out core mean | Held-out legacy mean |
|---|---:|---:|
| v6 | +37.05 | +174.48 |
| v9 | +41.03 | +159.33 |
| v11 | +41.83 | +164.25 |

### Deployed version: held-out results

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| aggressor | 1000 | 256 × 250 | +11.74 | 3.49 |
| bassander_lin_glm | 1000 | 256 × 250 | +97.26 | 8.52 |
| hadrian_marlowe_gpt | 1000 | 256 × 250 | +5.57 | 4.13 |
| valka_onderra_claude | 1000 | 256 × 250 | +49.55 | 6.93 |
| alan_wake_claude | 1000 | 256 × 250 | +83.26 | 5.28 |
| jack_joyce_gemini | 1000 | 256 × 250 | +256.88 | 9.33 |
| jesse_faden_glm | 1000 | 256 × 250 | +243.83 | 10.14 |
| max_payne_gpt | 1000 | 256 × 250 | +53.36 | 4.66 |
| random | 1000 | 256 × 250 | +359.05 | 6.83 |
| v1:Bot | 1000 | 128 × 250 | +7.32 | 7.30 |
| v2:Bot | 1000 | 128 × 250 | +1.07 | 7.49 |
| v3:Bot | 1000 | 128 × 250 | +1.36 | 6.44 |
| v4:Bot | 1000 | 128 × 250 | +24.20 | 6.98 |
| v5:Bot | 1000 | 128 × 250 | +8.98 | 8.31 |
| v6:Bot | 1000 | 256 × 250 | +3.82 | 5.58 |
| v7:Bot | 1000 | 128 × 250 | +0.75 | 7.64 |
| v8:Bot | 1000 | 128 × 250 | +14.56 | 6.67 |
| v10:Bot | 1000 | 128 × 250 | +3.50 | 7.44 |
| v11:Bot | 1000 | 128 × 250 | -0.80 | 7.58 |

### Known weaknesses

- The deployed range model is a heuristic transformation of equity against random cards. It can undervalue live draws and does not fully model opponent calls, board texture or bet-size-dependent ranges.
- The first opposing preflop raise receives partial pressure even if it is a three-bet of our opening raise. Version 3 tested stricter treatment, but did not establish an overall benchmark improvement.
- The bot cannot adapt across hands, and its fixed opening, bluffing and trapping frequencies can be exploited by an opponent designed to counter them.
- Premium sizing reveals extra strength. Equity sampling adds decision noise; the sample count and computation cap are explicit.
- Positive means in individual matchups are not automatically statistically established wins. These benchmarks do not establish strength against the independently developed phase-3 finals.

## Version history

Every listed parameter is a constructor argument; values are defaults actually benchmarked. Unchanged parameters are explicitly marked as inherited. Changes were selected as interpretable hypotheses, not fitted lookup tables or opponent-specific action signatures. The decision-line count conservatively includes method signature lines and nested helper lines, while excluding blank lines, comments, imports, constructors, initialization and metadata.

### v1 (parent: none)

**What it does:** Open a wide button range, isolate strong hands, and call using equity discounted for current-hand aggression. Bet for value and mix weak-hand bluffs when in position or holding initiative. Cap raises at effective stacks and account for unmatched chips in calling odds.

**Why it exists:** Sound stateless heads-up baseline. Range exclusion and effective-stack handling borrow ideas from hadrian_marlowe_gpt; the mixed bluffing is new.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | reasoned baseline default; not fitted |
| `open_equity` | 0.38 | equity probability versus random | reasoned baseline default; not fitted |
| `isolation_equity` | 0.56 | equity probability versus random | reasoned baseline default; not fitted |
| `preflop_raise_equity` | 0.67 | adjusted equity probability | reasoned baseline default; not fitted |
| `value_equity` | 0.7 | adjusted equity probability | reasoned baseline default; not fitted |
| `raise_equity` | 0.86 | adjusted equity probability | reasoned baseline default; not fitted |
| `open_size_bb` | 2.5 | raise-to big blinds | reasoned baseline default; not fitted |
| `reraise_multiple` | 3.0 | multiple of the current bet | reasoned baseline default; not fitted |
| `bet_pot_fraction` | 0.67 | fraction of the pot after calling | reasoned baseline default; not fitted |
| `premium_equity` | 0.92 | adjusted equity probability | reasoned baseline default; not fitted |
| `premium_pot_fraction` | 1.15 | fraction of the pot after calling | reasoned baseline default; not fitted |
| `raise_exclusion_fraction` | 0.28 | weakest-range fraction excluded per pressure unit | reasoned baseline default; not fitted |
| `prior_raise_weight` | 0.4 | current-street raise equivalents | reasoned baseline default; not fitted |
| `opening_raise_weight` | 0.4 | ordinary-raise equivalents for the initial opposing preflop raise | reasoned baseline default; not fitted |
| `realization_fraction` | 0.92 | fraction of equity realized with future betting | reasoned baseline default; not fitted |
| `call_margin_equity` | 0.025 | equity probability above pot odds | reasoned baseline default; not fitted |
| `bluff_frequency` | 0.35 | probability for eligible weak hands | reasoned baseline default; not fitted |
| `bluff_ceiling_equity` | 0.5 | equity probability versus random | reasoned baseline default; not fitted |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (smoke only) | 0 | 2 × 20 | +632.50 | 187.50 |
| aggressor | 0 | 64 × 250 | -19.61 | 6.30 |
| alan_wake_claude | 0 | 64 × 250 | +86.37 | 10.66 |
| bassander_lin_glm | 0 | 64 × 250 | +115.68 | 15.61 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +10.20 | 8.46 |
| jack_joyce_gemini | 0 | 64 × 250 | +261.33 | 20.37 |
| jesse_faden_glm | 0 | 64 × 250 | +285.77 | 18.54 |
| max_payne_gpt | 0 | 64 × 250 | +63.29 | 8.03 |
| valka_onderra_claude | 0 | 64 × 250 | +48.46 | 13.27 |
| aggressor | 1 | 128 × 250 | -13.74 | 4.62 |
| bassander_lin_glm | 1 | 128 × 250 | +109.63 | 12.41 |
| hadrian_marlowe_gpt | 1 | 128 × 250 | +2.53 | 5.39 |
| valka_onderra_claude | 1 | 128 × 250 | +50.10 | 9.68 |

Seed 0 group means: core +38.68 bb/100; legacy +174.19 bb/100.

Seed 1 group means: core +37.13 bb/100.

**Verdict:** Kept as the parent of v2, v3, v5 and v6, and as a reference opponent. Larger seed-1 results confirmed its positive core mean and its aggressor weakness.

**Decision-code line count:** 61.

### v2 (parent: v1)

**What it does:** Use the baseline strategy but allow the same weak-hand bluff frequency from either position. This adds leads from the big blind without requiring earlier initiative.

**Why it exists:** Test whether giving away too many out-of-position checks causes the aggressor loss.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | inherited unchanged from v1 |
| `open_equity` | 0.38 | equity probability versus random | inherited unchanged from v1 |
| `isolation_equity` | 0.56 | equity probability versus random | inherited unchanged from v1 |
| `preflop_raise_equity` | 0.67 | adjusted equity probability | inherited unchanged from v1 |
| `value_equity` | 0.7 | adjusted equity probability | inherited unchanged from v1 |
| `raise_equity` | 0.86 | adjusted equity probability | inherited unchanged from v1 |
| `open_size_bb` | 2.5 | raise-to big blinds | inherited unchanged from v1 |
| `reraise_multiple` | 3.0 | multiple of the current bet | inherited unchanged from v1 |
| `bet_pot_fraction` | 0.67 | fraction of the pot after calling | inherited unchanged from v1 |
| `premium_equity` | 0.92 | adjusted equity probability | inherited unchanged from v1 |
| `premium_pot_fraction` | 1.15 | fraction of the pot after calling | inherited unchanged from v1 |
| `raise_exclusion_fraction` | 0.28 | weakest-range fraction excluded per pressure unit | inherited unchanged from v1 |
| `prior_raise_weight` | 0.4 | current-street raise equivalents | inherited unchanged from v1 |
| `opening_raise_weight` | 0.4 | ordinary-raise equivalents for the initial opposing preflop raise | inherited unchanged from v1 |
| `realization_fraction` | 0.92 | fraction of equity realized with future betting | inherited unchanged from v1 |
| `call_margin_equity` | 0.025 | equity probability above pot odds | inherited unchanged from v1 |
| `bluff_frequency` | 0.35 | probability for eligible weak hands | inherited unchanged from v1 |
| `bluff_ceiling_equity` | 0.5 | equity probability versus random | inherited unchanged from v1 |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (smoke only) | 0 | 2 × 20 | +608.75 | 186.25 |
| aggressor | 0 | 64 × 250 | -6.37 | 6.37 |
| bassander_lin_glm | 0 | 64 × 250 | +54.23 | 15.89 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +11.59 | 10.60 |
| valka_onderra_claude | 0 | 64 × 250 | +36.29 | 16.23 |
| v1:Bot | 1 | 64 × 250 | +7.23 | 10.46 |

Seed 0 group means: core +23.93 bb/100.

Legacy: not run for this discarded candidate.

**Verdict:** Discarded: the core mean fell, particularly against bassander; the parent result was inconclusive. No statistically established overall improvement justified promotion.

**Decision-code line count:** 56.

### v3 (parent: v1)

**What it does:** Distinguish the initial preflop opener from a player re-raising our bet. Add extra range pressure when facing a raise after our own bet; retain the baseline opening, bluffing and sizing rules.

**Why it exists:** The baseline softens the first opposing preflop raise even when it is actually a three-bet. This version tests more cautious interpretation of re-raises.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | inherited unchanged from v1 |
| `open_equity` | 0.38 | equity probability versus random | inherited unchanged from v1 |
| `isolation_equity` | 0.56 | equity probability versus random | inherited unchanged from v1 |
| `preflop_raise_equity` | 0.67 | adjusted equity probability | inherited unchanged from v1 |
| `value_equity` | 0.7 | adjusted equity probability | inherited unchanged from v1 |
| `raise_equity` | 0.86 | adjusted equity probability | inherited unchanged from v1 |
| `open_size_bb` | 2.5 | raise-to big blinds | inherited unchanged from v1 |
| `reraise_multiple` | 3.0 | multiple of the current bet | inherited unchanged from v1 |
| `bet_pot_fraction` | 0.67 | fraction of the pot after calling | inherited unchanged from v1 |
| `premium_equity` | 0.92 | adjusted equity probability | inherited unchanged from v1 |
| `premium_pot_fraction` | 1.15 | fraction of the pot after calling | inherited unchanged from v1 |
| `raise_exclusion_fraction` | 0.28 | weakest-range fraction excluded per pressure unit | inherited unchanged from v1 |
| `prior_raise_weight` | 0.4 | current-street raise equivalents | inherited unchanged from v1 |
| `opening_raise_weight` | 0.4 | ordinary-raise equivalents for the initial opposing preflop raise | inherited unchanged from v1 |
| `realization_fraction` | 0.92 | fraction of equity realized with future betting | inherited unchanged from v1 |
| `call_margin_equity` | 0.025 | equity probability above pot odds | inherited unchanged from v1 |
| `bluff_frequency` | 0.35 | probability for eligible weak hands | inherited unchanged from v1 |
| `bluff_ceiling_equity` | 0.5 | equity probability versus random | inherited unchanged from v1 |
| `reraise_extra_weight` | 0.8 | additional raise-pressure equivalents | reasoned re-raise experiment; core seeds 0 and 1 |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (smoke only) | 0 | 2 × 20 | +280.00 | 177.50 |
| aggressor | 0 | 64 × 250 | -19.61 | 6.30 |
| bassander_lin_glm | 0 | 64 × 250 | +79.88 | 13.03 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +20.17 | 7.51 |
| valka_onderra_claude | 0 | 64 × 250 | +30.86 | 12.56 |
| aggressor | 1 | 128 × 250 | -13.74 | 4.62 |
| bassander_lin_glm | 1 | 128 × 250 | +89.46 | 10.88 |
| hadrian_marlowe_gpt | 1 | 128 × 250 | +11.35 | 4.96 |
| v1:Bot | 1 | 64 × 250 | +8.88 | 8.50 |
| valka_onderra_claude | 1 | 128 × 250 | +38.32 | 8.56 |

Seed 0 group means: core +27.82 bb/100.

Seed 1 group means: core +31.35 bb/100.

Legacy: not run for this discarded candidate.

**Verdict:** Kept as the parent of v4, then discarded as a finalist. Two core seeds did not show an overall gain over v1, although results against hadrian improved.

**Decision-code line count:** 66.

### v4 (parent: v3)

**What it does:** Keep the re-raise distinction from v3, but reduce the pressure attributed to ordinary postflop bets and previous aggression. Continue to apply the extra penalty for a raise of our own bet.

**Why it exists:** Test broader bluff-catching without abandoning caution against re-raises.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | inherited unchanged from v3 |
| `open_equity` | 0.38 | equity probability versus random | inherited unchanged from v3 |
| `isolation_equity` | 0.56 | equity probability versus random | inherited unchanged from v3 |
| `preflop_raise_equity` | 0.67 | adjusted equity probability | inherited unchanged from v3 |
| `value_equity` | 0.7 | adjusted equity probability | inherited unchanged from v3 |
| `raise_equity` | 0.86 | adjusted equity probability | inherited unchanged from v3 |
| `open_size_bb` | 2.5 | raise-to big blinds | inherited unchanged from v3 |
| `reraise_multiple` | 3.0 | multiple of the current bet | inherited unchanged from v3 |
| `bet_pot_fraction` | 0.67 | fraction of the pot after calling | inherited unchanged from v3 |
| `premium_equity` | 0.92 | adjusted equity probability | inherited unchanged from v3 |
| `premium_pot_fraction` | 1.15 | fraction of the pot after calling | inherited unchanged from v3 |
| `raise_exclusion_fraction` | 0.28 | weakest-range fraction excluded per pressure unit | inherited unchanged from v3 |
| `prior_raise_weight` | 0.4 | current-street raise equivalents | inherited unchanged from v3 |
| `opening_raise_weight` | 0.4 | ordinary-raise equivalents for the initial opposing preflop raise | inherited unchanged from v3 |
| `realization_fraction` | 0.92 | fraction of equity realized with future betting | inherited unchanged from v3 |
| `call_margin_equity` | 0.025 | equity probability above pot odds | inherited unchanged from v3 |
| `bluff_frequency` | 0.35 | probability for eligible weak hands | inherited unchanged from v3 |
| `bluff_ceiling_equity` | 0.5 | equity probability versus random | inherited unchanged from v3 |
| `reraise_extra_weight` | 0.8 | additional raise-pressure equivalents | inherited unchanged from v3 |
| `postflop_pressure_weight` | 0.6 | multiplier of ordinary postflop pressure | reasoned wider-defense trial; core seed 0 |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (smoke only) | 0 | 2 × 20 | +275.00 | 172.50 |
| aggressor | 0 | 64 × 250 | -6.70 | 6.84 |
| bassander_lin_glm | 0 | 64 × 250 | +46.28 | 15.81 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | -1.53 | 9.03 |
| valka_onderra_claude | 0 | 64 × 250 | +16.21 | 13.61 |
| v3:Bot | 1 | 64 × 250 | -20.84 | 10.00 |

Seed 0 group means: core +13.57 bb/100.

Legacy: not run for this discarded candidate.

**Verdict:** Discarded: lower core performance and a negative parent match gave no support for the looser defense.

**Decision-code line count:** 67.

### v5 (parent: v1)

**What it does:** Use one Monte Carlo batch for both random-hand equity and opponent-range equity. Rank sampled opponent holdings using only the dealt board, give weak holdings less weight, and retain each sample’s future drawing outcome. Keep the baseline scalar discount preflop.

**Why it exists:** Test whether scalar equity discounts throw away too much live draw equity. The idea of conditioning sampled ranges on the visible board is inspired by valka_onderra_claude, but this implementation uses made-hand percentiles rather than its pair/miss rule.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | inherited unchanged from v1 |
| `open_equity` | 0.38 | equity probability versus random | inherited unchanged from v1 |
| `isolation_equity` | 0.56 | equity probability versus random | inherited unchanged from v1 |
| `preflop_raise_equity` | 0.67 | adjusted equity probability | inherited unchanged from v1 |
| `value_equity` | 0.7 | adjusted equity probability | inherited unchanged from v1 |
| `raise_equity` | 0.86 | adjusted equity probability | inherited unchanged from v1 |
| `open_size_bb` | 2.5 | raise-to big blinds | inherited unchanged from v1 |
| `reraise_multiple` | 3.0 | multiple of the current bet | inherited unchanged from v1 |
| `bet_pot_fraction` | 0.67 | fraction of the pot after calling | inherited unchanged from v1 |
| `premium_equity` | 0.92 | adjusted equity probability | inherited unchanged from v1 |
| `premium_pot_fraction` | 1.15 | fraction of the pot after calling | inherited unchanged from v1 |
| `raise_exclusion_fraction` | 0.28 | weakest-range fraction excluded per pressure unit | inherited unchanged from v1 |
| `prior_raise_weight` | 0.4 | current-street raise equivalents | inherited unchanged from v1 |
| `opening_raise_weight` | 0.4 | ordinary-raise equivalents for the initial opposing preflop raise | inherited unchanged from v1 |
| `realization_fraction` | 0.92 | fraction of equity realized with future betting | inherited unchanged from v1 |
| `call_margin_equity` | 0.025 | equity probability above pot odds | inherited unchanged from v1 |
| `bluff_frequency` | 0.35 | probability for eligible weak hands | inherited unchanged from v1 |
| `bluff_ceiling_equity` | 0.5 | equity probability versus random | inherited unchanged from v1 |
| `weak_range_weight` | 0.15 | relative likelihood of an excluded weak holding | reasoned weak-range mixture; core seed 0 |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (smoke only) | 0 | 2 × 20 | +223.75 | 206.25 |
| aggressor | 0 | 64 × 250 | -21.20 | 8.36 |
| bassander_lin_glm | 0 | 64 × 250 | +60.29 | 20.47 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | -12.05 | 9.77 |
| valka_onderra_claude | 0 | 64 × 250 | -5.13 | 15.65 |
| v1:Bot | 1 | 64 × 250 | -13.10 | 10.49 |

Seed 0 group means: core +5.48 bb/100.

Legacy: not run for this discarded candidate.

**Verdict:** Discarded: markedly lower core mean and no parent improvement. The extra range-model complexity was not supported by the measurements.

**Decision-code line count:** 88.

### v6 (parent: v1)

**What it does:** Expand value re-raises preflop and mix in some weaker suited hands or ace blockers against a single opening raise. Keep the baseline postflop strategy and bet sizing.

**Why it exists:** Contest wide opening ranges before the flop, rather than relying entirely on postflop bluff-catching.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | inherited unchanged from v1 |
| `open_equity` | 0.38 | equity probability versus random | inherited unchanged from v1 |
| `isolation_equity` | 0.56 | equity probability versus random | inherited unchanged from v1 |
| `preflop_raise_equity` | 0.6 | adjusted equity probability | reasoned wider/polarized preflop trial; core seeds 0 and 2 |
| `value_equity` | 0.7 | adjusted equity probability | inherited unchanged from v1 |
| `raise_equity` | 0.86 | adjusted equity probability | inherited unchanged from v1 |
| `open_size_bb` | 2.5 | raise-to big blinds | inherited unchanged from v1 |
| `reraise_multiple` | 3.0 | multiple of the current bet | inherited unchanged from v1 |
| `bet_pot_fraction` | 0.67 | fraction of the pot after calling | inherited unchanged from v1 |
| `premium_equity` | 0.92 | adjusted equity probability | inherited unchanged from v1 |
| `premium_pot_fraction` | 1.15 | fraction of the pot after calling | inherited unchanged from v1 |
| `raise_exclusion_fraction` | 0.28 | weakest-range fraction excluded per pressure unit | inherited unchanged from v1 |
| `prior_raise_weight` | 0.4 | current-street raise equivalents | inherited unchanged from v1 |
| `opening_raise_weight` | 0.4 | ordinary-raise equivalents for the initial opposing preflop raise | inherited unchanged from v1 |
| `realization_fraction` | 0.92 | fraction of equity realized with future betting | inherited unchanged from v1 |
| `call_margin_equity` | 0.025 | equity probability above pot odds | inherited unchanged from v1 |
| `bluff_frequency` | 0.35 | probability for eligible weak hands | inherited unchanged from v1 |
| `bluff_ceiling_equity` | 0.5 | equity probability versus random | inherited unchanged from v1 |
| `threebet_bluff_frequency` | 0.2 | probability for eligible preflop bluff hands | reasoned wider/polarized preflop trial; core seeds 0 and 2 |
| `threebet_bluff_ceiling_equity` | 0.58 | equity probability versus random | reasoned wider/polarized preflop trial; core seeds 0 and 2 |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (smoke only) | 0 | 2 × 20 | +876.25 | 431.25 |
| aggressor | 0 | 64 × 250 | -11.96 | 5.94 |
| bassander_lin_glm | 0 | 64 × 250 | +123.97 | 18.06 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +22.94 | 8.25 |
| valka_onderra_claude | 0 | 64 × 250 | +40.26 | 15.79 |
| v1:Bot | 1 | 64 × 250 | +16.02 | 10.65 |
| aggressor | 2 | 128 × 250 | -2.35 | 4.11 |
| bassander_lin_glm | 2 | 128 × 250 | +109.55 | 12.23 |
| hadrian_marlowe_gpt | 2 | 128 × 250 | +9.87 | 5.38 |
| v1:Bot | 2 | 128 × 250 | +12.79 | 7.78 |
| valka_onderra_claude | 2 | 128 × 250 | +57.30 | 9.43 |
| aggressor | 1000 | 256 × 250 | -3.82 | 2.97 |
| alan_wake_claude | 1000 | 256 × 250 | +89.42 | 5.51 |
| bassander_lin_glm | 1000 | 256 × 250 | +93.17 | 8.89 |
| hadrian_marlowe_gpt | 1000 | 256 × 250 | +1.10 | 4.06 |
| jack_joyce_gemini | 1000 | 256 × 250 | +282.04 | 10.08 |
| jesse_faden_glm | 1000 | 256 × 250 | +270.61 | 10.01 |
| max_payne_gpt | 1000 | 256 × 250 | +55.86 | 4.72 |
| random | 1000 | 256 × 250 | +393.95 | 7.12 |
| v1:Bot | 1000 | 256 × 250 | +7.63 | 5.40 |
| valka_onderra_claude | 1000 | 256 × 250 | +57.73 | 6.92 |

Seed 0 group means: core +43.81 bb/100.

Seed 2 group means: core +43.59 bb/100.

Seed 1000 group means: core +37.05 bb/100; legacy +174.48 bb/100.

**Verdict:** Kept as the parent of v7, v8 and v9, and as a held-out finalist. The core mean was similar on two seeds; both training parent matches were positive, though individually noisy. Not deployed; see the final-choice comparison above.

**Decision-code line count:** 67.

### v7 (parent: v6)

**What it does:** Keep v6’s hand selection and sizing but increase the frequency of its eligible preflop bluff re-raises. Other decisions remain the same.

**Why it exists:** Test a more polarized, aggressive response to wide opens.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | inherited unchanged from v6 |
| `open_equity` | 0.38 | equity probability versus random | inherited unchanged from v6 |
| `isolation_equity` | 0.56 | equity probability versus random | inherited unchanged from v6 |
| `preflop_raise_equity` | 0.6 | adjusted equity probability | inherited unchanged from v6 |
| `value_equity` | 0.7 | adjusted equity probability | inherited unchanged from v6 |
| `raise_equity` | 0.86 | adjusted equity probability | inherited unchanged from v6 |
| `open_size_bb` | 2.5 | raise-to big blinds | inherited unchanged from v6 |
| `reraise_multiple` | 3.0 | multiple of the current bet | inherited unchanged from v6 |
| `bet_pot_fraction` | 0.67 | fraction of the pot after calling | inherited unchanged from v6 |
| `premium_equity` | 0.92 | adjusted equity probability | inherited unchanged from v6 |
| `premium_pot_fraction` | 1.15 | fraction of the pot after calling | inherited unchanged from v6 |
| `raise_exclusion_fraction` | 0.28 | weakest-range fraction excluded per pressure unit | inherited unchanged from v6 |
| `prior_raise_weight` | 0.4 | current-street raise equivalents | inherited unchanged from v6 |
| `opening_raise_weight` | 0.4 | ordinary-raise equivalents for the initial opposing preflop raise | inherited unchanged from v6 |
| `realization_fraction` | 0.92 | fraction of equity realized with future betting | inherited unchanged from v6 |
| `call_margin_equity` | 0.025 | equity probability above pot odds | inherited unchanged from v6 |
| `bluff_frequency` | 0.35 | probability for eligible weak hands | inherited unchanged from v6 |
| `bluff_ceiling_equity` | 0.5 | equity probability versus random | inherited unchanged from v6 |
| `threebet_bluff_frequency` | 0.65 | probability for eligible preflop bluff hands | large frequency contrast; core seed 0, not fine grid tuning |
| `threebet_bluff_ceiling_equity` | 0.58 | equity probability versus random | inherited unchanged from v6 |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (smoke only) | 0 | 2 × 20 | +876.25 | 431.25 |
| aggressor | 0 | 64 × 250 | +4.33 | 6.87 |
| bassander_lin_glm | 0 | 64 × 250 | +93.50 | 18.80 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +26.29 | 8.68 |
| valka_onderra_claude | 0 | 64 × 250 | +27.19 | 13.24 |
| v6:Bot | 1 | 64 × 250 | +8.82 | 10.97 |

Seed 0 group means: core +37.83 bb/100.

Legacy: not run for this discarded candidate.

**Verdict:** Discarded as a finalist: improved aggressor balance on the first seed did not improve the overall core mean, and the parent result was uncertain. The mixed-checking branch offered a better-supported balance.

**Decision-code line count:** 67.

### v8 (parent: v6)

**What it does:** Keep v6’s betting ranges but size ordinary postflop bets at 40% of the pot. Retain the larger premium-hand sizing.

**Why it exists:** Test whether cheaper bets preserve sufficient fold equity to offset reduced value extraction.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | inherited unchanged from v6 |
| `open_equity` | 0.38 | equity probability versus random | inherited unchanged from v6 |
| `isolation_equity` | 0.56 | equity probability versus random | inherited unchanged from v6 |
| `preflop_raise_equity` | 0.6 | adjusted equity probability | inherited unchanged from v6 |
| `value_equity` | 0.7 | adjusted equity probability | inherited unchanged from v6 |
| `raise_equity` | 0.86 | adjusted equity probability | inherited unchanged from v6 |
| `open_size_bb` | 2.5 | raise-to big blinds | inherited unchanged from v6 |
| `reraise_multiple` | 3.0 | multiple of the current bet | inherited unchanged from v6 |
| `bet_pot_fraction` | 0.4 | fraction of the pot after calling | smaller-bet hypothesis; core seed 0 |
| `premium_equity` | 0.92 | adjusted equity probability | inherited unchanged from v6 |
| `premium_pot_fraction` | 1.15 | fraction of the pot after calling | inherited unchanged from v6 |
| `raise_exclusion_fraction` | 0.28 | weakest-range fraction excluded per pressure unit | inherited unchanged from v6 |
| `prior_raise_weight` | 0.4 | current-street raise equivalents | inherited unchanged from v6 |
| `opening_raise_weight` | 0.4 | ordinary-raise equivalents for the initial opposing preflop raise | inherited unchanged from v6 |
| `realization_fraction` | 0.92 | fraction of equity realized with future betting | inherited unchanged from v6 |
| `call_margin_equity` | 0.025 | equity probability above pot odds | inherited unchanged from v6 |
| `bluff_frequency` | 0.35 | probability for eligible weak hands | inherited unchanged from v6 |
| `bluff_ceiling_equity` | 0.5 | equity probability versus random | inherited unchanged from v6 |
| `threebet_bluff_frequency` | 0.2 | probability for eligible preflop bluff hands | inherited unchanged from v6 |
| `threebet_bluff_ceiling_equity` | 0.58 | equity probability versus random | inherited unchanged from v6 |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (smoke only) | 0 | 2 × 20 | +495.00 | 65.00 |
| aggressor | 0 | 64 × 250 | -12.01 | 5.09 |
| bassander_lin_glm | 0 | 64 × 250 | +77.60 | 18.56 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +24.88 | 8.22 |
| valka_onderra_claude | 0 | 64 × 250 | +33.75 | 12.36 |
| v6:Bot | 1 | 64 × 250 | -2.91 | 10.04 |

Seed 0 group means: core +31.06 bb/100.

Legacy: not run for this discarded candidate.

**Verdict:** Discarded: lower core mean and an essentially even parent match did not establish a benefit.

**Decision-code line count:** 67.

### v9 (parent: v6)

**What it does:** Keep v6’s wider preflop strategy. Before the river, occasionally check value hands out of position, so checking does not always announce a weak or middling hand. Continue normal value betting in position and on the river.

**Why it exists:** Protect the checking range and induce bets from frequent bettors, while allowing more than weak hands to reach a check-call line.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | inherited unchanged from v6 |
| `open_equity` | 0.38 | equity probability versus random | inherited unchanged from v6 |
| `isolation_equity` | 0.56 | equity probability versus random | inherited unchanged from v6 |
| `preflop_raise_equity` | 0.6 | adjusted equity probability | inherited unchanged from v6 |
| `value_equity` | 0.7 | adjusted equity probability | inherited unchanged from v6 |
| `raise_equity` | 0.86 | adjusted equity probability | inherited unchanged from v6 |
| `open_size_bb` | 2.5 | raise-to big blinds | inherited unchanged from v6 |
| `reraise_multiple` | 3.0 | multiple of the current bet | inherited unchanged from v6 |
| `bet_pot_fraction` | 0.67 | fraction of the pot after calling | inherited unchanged from v6 |
| `premium_equity` | 0.92 | adjusted equity probability | inherited unchanged from v6 |
| `premium_pot_fraction` | 1.15 | fraction of the pot after calling | inherited unchanged from v6 |
| `raise_exclusion_fraction` | 0.28 | weakest-range fraction excluded per pressure unit | inherited unchanged from v6 |
| `prior_raise_weight` | 0.4 | current-street raise equivalents | inherited unchanged from v6 |
| `opening_raise_weight` | 0.4 | ordinary-raise equivalents for the initial opposing preflop raise | inherited unchanged from v6 |
| `realization_fraction` | 0.92 | fraction of equity realized with future betting | inherited unchanged from v6 |
| `call_margin_equity` | 0.025 | equity probability above pot odds | inherited unchanged from v6 |
| `bluff_frequency` | 0.35 | probability for eligible weak hands | inherited unchanged from v6 |
| `bluff_ceiling_equity` | 0.5 | equity probability versus random | inherited unchanged from v6 |
| `threebet_bluff_frequency` | 0.2 | probability for eligible preflop bluff hands | inherited unchanged from v6 |
| `threebet_bluff_ceiling_equity` | 0.58 | equity probability versus random | inherited unchanged from v6 |
| `trap_frequency` | 0.35 | probability of checking a value hand out of position before the river | reasoned checking-range protection; core seeds 0 and 2 |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (smoke only) | 0 | 2 × 20 | +876.25 | 431.25 |
| aggressor | 0 | 64 × 250 | +11.59 | 6.12 |
| bassander_lin_glm | 0 | 64 × 250 | +116.83 | 19.02 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +19.89 | 8.24 |
| valka_onderra_claude | 0 | 64 × 250 | +22.05 | 12.03 |
| v6:Bot | 1 | 64 × 250 | +9.45 | 10.84 |
| aggressor | 2 | 128 × 250 | +11.06 | 4.73 |
| bassander_lin_glm | 2 | 128 × 250 | +95.17 | 12.73 |
| hadrian_marlowe_gpt | 2 | 128 × 250 | +3.32 | 5.46 |
| v6:Bot | 2 | 128 × 250 | +5.88 | 8.13 |
| valka_onderra_claude | 2 | 128 × 250 | +64.37 | 8.78 |
| aggressor | 1000 | 256 × 250 | +11.74 | 3.49 |
| alan_wake_claude | 1000 | 256 × 250 | +83.26 | 5.28 |
| bassander_lin_glm | 1000 | 256 × 250 | +97.26 | 8.52 |
| hadrian_marlowe_gpt | 1000 | 256 × 250 | +5.57 | 4.13 |
| jack_joyce_gemini | 1000 | 256 × 250 | +256.88 | 9.33 |
| jesse_faden_glm | 1000 | 256 × 250 | +243.83 | 10.14 |
| max_payne_gpt | 1000 | 256 × 250 | +53.36 | 4.66 |
| random | 1000 | 256 × 250 | +359.05 | 6.83 |
| v10:Bot | 1000 | 128 × 250 | +3.50 | 7.44 |
| v11:Bot | 1000 | 128 × 250 | -0.80 | 7.58 |
| v1:Bot | 1000 | 128 × 250 | +7.32 | 7.30 |
| v2:Bot | 1000 | 128 × 250 | +1.07 | 7.49 |
| v3:Bot | 1000 | 128 × 250 | +1.36 | 6.44 |
| v4:Bot | 1000 | 128 × 250 | +24.20 | 6.98 |
| v5:Bot | 1000 | 128 × 250 | +8.98 | 8.31 |
| v6:Bot | 1000 | 256 × 250 | +3.82 | 5.58 |
| v7:Bot | 1000 | 128 × 250 | +0.75 | 7.64 |
| v8:Bot | 1000 | 128 × 250 | +14.56 | 6.67 |
| valka_onderra_claude | 1000 | 256 × 250 | +49.55 | 6.93 |

Seed 0 group means: core +42.59 bb/100.

Seed 2 group means: core +43.48 bb/100.

Seed 1000 group means: core +41.03 bb/100; legacy +159.33 bb/100.

**Verdict:** Kept as the parent of v10 and v11 and as a held-out finalist. Its roughly +11 bb/100 training aggressor result repeated on a larger second seed without reducing the core mean materially. Deployed after held-out selection; see the final-choice rationale above.

**Decision-code line count:** 68.

### v10 (parent: v9)

**What it does:** Use v9’s postflop plan with cheaper two-big-blind preflop opens and an almost-all-hands button range. The common opening size also applies when isolating a limp.

**Why it exists:** Test a coherent wider, cheaper opening plan that risks fewer chips with the bottom of the range.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | inherited unchanged from v9 |
| `open_equity` | 0.3 | equity probability versus random | reasoned cheaper/wider opening plan; core seed 0 |
| `isolation_equity` | 0.56 | equity probability versus random | inherited unchanged from v9 |
| `preflop_raise_equity` | 0.6 | adjusted equity probability | inherited unchanged from v9 |
| `value_equity` | 0.7 | adjusted equity probability | inherited unchanged from v9 |
| `raise_equity` | 0.86 | adjusted equity probability | inherited unchanged from v9 |
| `open_size_bb` | 2.0 | raise-to big blinds | reasoned cheaper/wider opening plan; core seed 0 |
| `reraise_multiple` | 3.0 | multiple of the current bet | inherited unchanged from v9 |
| `bet_pot_fraction` | 0.67 | fraction of the pot after calling | inherited unchanged from v9 |
| `premium_equity` | 0.92 | adjusted equity probability | inherited unchanged from v9 |
| `premium_pot_fraction` | 1.15 | fraction of the pot after calling | inherited unchanged from v9 |
| `raise_exclusion_fraction` | 0.28 | weakest-range fraction excluded per pressure unit | inherited unchanged from v9 |
| `prior_raise_weight` | 0.4 | current-street raise equivalents | inherited unchanged from v9 |
| `opening_raise_weight` | 0.4 | ordinary-raise equivalents for the initial opposing preflop raise | inherited unchanged from v9 |
| `realization_fraction` | 0.92 | fraction of equity realized with future betting | inherited unchanged from v9 |
| `call_margin_equity` | 0.025 | equity probability above pot odds | inherited unchanged from v9 |
| `bluff_frequency` | 0.35 | probability for eligible weak hands | inherited unchanged from v9 |
| `bluff_ceiling_equity` | 0.5 | equity probability versus random | inherited unchanged from v9 |
| `threebet_bluff_frequency` | 0.2 | probability for eligible preflop bluff hands | inherited unchanged from v9 |
| `threebet_bluff_ceiling_equity` | 0.58 | equity probability versus random | inherited unchanged from v9 |
| `trap_frequency` | 0.35 | probability of checking a value hand out of position before the river | inherited unchanged from v9 |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (smoke only) | 0 | 2 × 20 | +535.00 | 202.50 |
| aggressor | 0 | 64 × 250 | +12.08 | 6.59 |
| bassander_lin_glm | 0 | 64 × 250 | +90.38 | 15.64 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +18.58 | 8.40 |
| valka_onderra_claude | 0 | 64 × 250 | +32.18 | 11.51 |
| v9:Bot | 1 | 64 × 250 | -9.64 | 10.88 |

Seed 0 group means: core +38.31 bb/100.

Legacy: not run for this discarded candidate.

**Verdict:** Discarded: lower initial core mean and no parent win did not justify replacing v9. This is not a claim that the small observed difference is statistically decisive.

**Decision-code line count:** 68.

### v11 (parent: v9)

**What it does:** Use the same decisions and parameters as v9, but estimate equity with 256 rather than 128 Monte Carlo deals. This uses the full allowed sampling budget to reduce decision noise.

**Why it exists:** Test whether more precise equity estimates help around call and raise thresholds.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 256 | Monte Carlo deals per decision | maximum legal sample budget; core seeds 0 and 2 |
| `open_equity` | 0.38 | equity probability versus random | inherited unchanged from v9 |
| `isolation_equity` | 0.56 | equity probability versus random | inherited unchanged from v9 |
| `preflop_raise_equity` | 0.6 | adjusted equity probability | inherited unchanged from v9 |
| `value_equity` | 0.7 | adjusted equity probability | inherited unchanged from v9 |
| `raise_equity` | 0.86 | adjusted equity probability | inherited unchanged from v9 |
| `open_size_bb` | 2.5 | raise-to big blinds | inherited unchanged from v9 |
| `reraise_multiple` | 3.0 | multiple of the current bet | inherited unchanged from v9 |
| `bet_pot_fraction` | 0.67 | fraction of the pot after calling | inherited unchanged from v9 |
| `premium_equity` | 0.92 | adjusted equity probability | inherited unchanged from v9 |
| `premium_pot_fraction` | 1.15 | fraction of the pot after calling | inherited unchanged from v9 |
| `raise_exclusion_fraction` | 0.28 | weakest-range fraction excluded per pressure unit | inherited unchanged from v9 |
| `prior_raise_weight` | 0.4 | current-street raise equivalents | inherited unchanged from v9 |
| `opening_raise_weight` | 0.4 | ordinary-raise equivalents for the initial opposing preflop raise | inherited unchanged from v9 |
| `realization_fraction` | 0.92 | fraction of equity realized with future betting | inherited unchanged from v9 |
| `call_margin_equity` | 0.025 | equity probability above pot odds | inherited unchanged from v9 |
| `bluff_frequency` | 0.35 | probability for eligible weak hands | inherited unchanged from v9 |
| `bluff_ceiling_equity` | 0.5 | equity probability versus random | inherited unchanged from v9 |
| `threebet_bluff_frequency` | 0.2 | probability for eligible preflop bluff hands | inherited unchanged from v9 |
| `threebet_bluff_ceiling_equity` | 0.58 | equity probability versus random | inherited unchanged from v9 |
| `trap_frequency` | 0.35 | probability of checking a value hand out of position before the river | inherited unchanged from v9 |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (smoke only) | 0 | 2 × 20 | +521.25 | 21.25 |
| aggressor | 0 | 64 × 250 | +10.15 | 6.61 |
| bassander_lin_glm | 0 | 64 × 250 | +100.71 | 16.63 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +17.53 | 10.08 |
| valka_onderra_claude | 0 | 64 × 250 | +57.01 | 12.71 |
| v9:Bot | 1 | 64 × 250 | +21.13 | 10.45 |
| aggressor | 2 | 128 × 250 | +1.69 | 4.69 |
| bassander_lin_glm | 2 | 128 × 250 | +95.45 | 13.60 |
| hadrian_marlowe_gpt | 2 | 128 × 250 | +16.69 | 6.41 |
| v9:Bot | 2 | 128 × 250 | -2.09 | 7.69 |
| valka_onderra_claude | 2 | 128 × 250 | +50.50 | 10.14 |
| aggressor | 1000 | 256 × 250 | +9.02 | 3.51 |
| alan_wake_claude | 1000 | 256 × 250 | +93.67 | 5.01 |
| bassander_lin_glm | 1000 | 256 × 250 | +97.06 | 8.57 |
| hadrian_marlowe_gpt | 1000 | 256 × 250 | +14.56 | 3.94 |
| jack_joyce_gemini | 1000 | 256 × 250 | +255.41 | 10.19 |
| jesse_faden_glm | 1000 | 256 × 250 | +254.29 | 9.64 |
| max_payne_gpt | 1000 | 256 × 250 | +53.62 | 4.92 |
| random | 1000 | 256 × 250 | +366.55 | 6.95 |
| v9:Bot | 1000 | 256 × 250 | +1.46 | 5.46 |
| valka_onderra_claude | 1000 | 256 × 250 | +46.66 | 6.49 |

Seed 0 group means: core +46.35 bb/100.

Seed 2 group means: core +41.08 bb/100.

Seed 1000 group means: core +41.83 bb/100; legacy +164.25 bb/100.

**Verdict:** Retained as a held-out finalist. Core training results were close to v9, so no small one-seed gain was treated as proof; the second seed and parent matches were also recorded. Not deployed; see the final-choice comparison above.

**Decision-code line count:** 68.

## Verification and saved files

`verification.log` records successful checks of all 11 versions: one `Bot` class, no `observe`, complete JSON-serializable parameter metadata, valid sample limits, immutable source fingerprints, `None` state, JIT/vmap execution, int32 scalar action fields, no folding a free check, no raising when prohibited, and bounded raise sizes. Each version also completed real engine smoke tests and core benchmarks.

`source_audit.json` contains decision-line counts and SHA-256 fingerprints. `check.log` preserves the successful environment check. `evaluation_plan.json` records the precommitted finalist list and freeze time. `deployment.json` records the final selection. `results.json` is generated from raw benchmark logs. `verify.py` and `build_readme.py` are local audit/report utilities and are not imported by any bot. All these scratch and audit files stay in the bot directory.

The one-line `__init__.py` is the deployment entry point. Version files are independent and never import another version. No version was edited after it was benchmarked.
