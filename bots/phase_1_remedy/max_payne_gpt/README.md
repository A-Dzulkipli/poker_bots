# max_payne_gpt

- Lineage: `phase_1_remedy/max_payne_gpt`.
- Model: GPT-6 (Codex; exact serving variant unavailable).
- Prompt version: `botgen-v1`.
- Engine commit: `02abc49` (checked against the local repository).
- Code executed: **yes**, on the machine holding the requested directory, using its existing Python 3.13.12 environment and JAX 0.11.2.
- Wall-clock time used, including final checks: **12.68 minutes (761 seconds)**. Started 2026-10-14 00:13:33 Australia/Adelaide (2026-10-13 13:43:33 UTC).
- Iteration stopped at about 8.1 minutes, before the 12-minute cutoff. Elapsed time was checked between steps; no persistent scheduled task was created.
- Deployed export: `from .v1 import Bot as DeployedBot`.

## Method and checks

Every version uses 64 Monte Carlo samples in one equity call per decision. `init_state` returns `None`, and `act` returns its input state unchanged. No runtime access to hidden cards, no cross-hand memory, no version-to-version imports, and no changes to benchmarked version files were made. Existing PROMPT.md and prompt.toml were preserved.

The supplied equity.py and bench.py were copied verbatim and checked against the attached prompt. Each version passed the required seed-0, 2-table × 20-hand qualification before full benchmarking. These tiny qualification results are included below for completeness, not used to rank versions. The full development runs used seed 0 only. Every newer version faced RandomBot and all earlier versions.

Versions 1 and 3 were declared finalists before reading held-out results. Each was run exactly once per ordered matchup on seed 1000 against RandomBot and every other version, including later experiments. Candidate/opponent seat roles differ between v1-v3 and v3-v1, so those are distinct ordered matchups, not repeat runs. No strategy or parameter was changed after held-out evaluation.

Full runs use 32 tables × 250 hands: 24,000 candidate seat-hands, six seats alternating three candidate and three opponent seats, 100bb starting stacks, 1/2 blinds, rebuys and cash-outs as specified. Raw output is retained in `vN_OPPONENT_seedS.txt`. The SE values are exactly those printed by the supplied script. Seat-hands within a table are correlated, so these are not cluster-adjusted confidence estimates; differences near two SE are treated cautiously.

Additional checks verified JSON-serializable metadata with every constructor parameter, exactly one Bot class per version, decision line counts, and sample caps. Mixed dispatch of all five versions compiled and simulated at both 2 and 9 seats using seed 9 (2 tables × 5 hands); these were contract checks, not selection benchmarks.

## Version 1

**Parent:** none.

**What it does:** Estimate showdown equity against uniform opponent cards, discount it before the river, and compare it with pot odds plus a margin. Enter preflop selectively and raise a three-quarter-pot amount with strong equity; unopened raises are at least 2.5bb.

**Why it exists:** Establish a simple, legible baseline with multiway equity, legal raise totals, and no free folds.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 64 | samples per decision | 64 balances sampling noise and runtime; reasoning default |
| `realization_fraction` | 0.85 | fraction of equity retained before river | 0.85 discounts incomplete equity realization; reasoning default |
| `call_margin_equity` | 0.04 | equity probability | 0.04 adds a four-percentage-point call cushion; reasoning default |
| `preflop_entry_multiplier` | 1.15 | multiple of equal pot share | 1.15 avoids entering with below-average holdings; reasoning default |
| `preflop_raise_multiplier` | 1.8 | multiple of equal pot share | 1.8 selects a strong initial raising range; reasoning default |
| `value_raise_equity` | 0.62 | equity probability | 0.62 is the postflop value threshold and preflop threshold cap; reasoning default |
| `bet_pot_fraction` | 0.75 | fraction of pot after calling | 0.75 balances value and price; reasoning default |
| `opening_size_bb` | 2.5 | big blinds, street total | 2.5 is the minimum unopened preflop raise total; reasoning default |

**Results:** The first row is the qualification run. Seed 1000 rows are held out.

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (qualification) | 0 | 2 × 20 | +604.58 | 289.73 |
| random | 0 | 32 × 250 | +377.99 | 24.26 |
| random | 1000 | 32 × 250 | +442.48 | 26.84 |
| v2 | 1000 | 32 × 250 | +2.93 | 4.29 |
| v3 | 1000 | 32 × 250 | -6.49 | 4.93 |
| v4 | 1000 | 32 × 250 | +0.49 | 4.99 |
| v5 | 1000 | 32 × 250 | +3.00 | 4.18 |

**Verdict:** Kept as parent of v2 and v3, and deployed. No later change established an advantage against it; it also had the strongest RandomBot results.

**Decision line count:** 20 nonblank, noncomment body lines; 21 including the `def act` line. No additional decision helpers are defined in this version.

## Version 2

**Parent:** 1.

**What it does:** Use the baseline strategy, but multiply equity by 0.82 for each opposing raise on the current street. Apply that adjusted equity to both calls and raises.

**Why it exists:** Test whether uniform-card equity is too optimistic against opponents who have raised. This is within-hand evidence only.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 64 | samples per decision | 64 balances sampling noise and runtime; reasoning default |
| `equity_retained_per_raise` | 0.82 | fraction of equity retained per opposing raise | 0.82 tests a moderate multiplicative range correction; reasoning, tested against v1 and RandomBot |
| `realization_fraction` | 0.85 | fraction of equity retained before river | 0.85 discounts incomplete equity realization; reasoning default |
| `call_margin_equity` | 0.04 | equity probability | 0.04 adds a four-percentage-point call cushion; reasoning default |
| `preflop_entry_multiplier` | 1.15 | multiple of equal pot share | 1.15 avoids entering with below-average holdings; reasoning default |
| `preflop_raise_multiplier` | 1.8 | multiple of equal pot share | 1.8 selects a strong initial raising range; reasoning default |
| `value_raise_equity` | 0.62 | equity probability | 0.62 is the postflop value threshold and preflop threshold cap; reasoning default |
| `bet_pot_fraction` | 0.75 | fraction of pot after calling | 0.75 balances value and price; reasoning default |
| `opening_size_bb` | 2.5 | big blinds, street total | 2.5 is the minimum unopened preflop raise total; reasoning default |

**Results:** The first row is the qualification run. Seed 1000 rows are held out.

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (qualification) | 0 | 2 × 20 | +260.00 | 170.77 |
| random | 0 | 32 × 250 | +196.20 | 16.08 |
| v1 | 0 | 32 × 250 | -4.28 | 2.75 |

**Verdict:** Discarded. The seed-0 loss to v1 is within about two reported SE, so it does not establish inferiority; equally, it provides no support for retaining the change. RandomBot performance fell markedly. The discount can also penalize near-certain winners too heavily.

**Decision line count:** 21 nonblank, noncomment body lines; 22 including the `def act` line. No additional decision helpers are defined in this version.

## Version 3

**Parent:** 1.

**What it does:** Use v1, but permit a value raise only when the hand also passes the call-equity test. All parameters and sizing remain unchanged.

**Why it exists:** Test a price guard against making a value raise when the baseline would reject a call at the current price.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 64 | samples per decision | 64 balances sampling noise and runtime; reasoning default |
| `realization_fraction` | 0.85 | fraction of equity retained before river | 0.85 discounts incomplete equity realization; reasoning default |
| `call_margin_equity` | 0.04 | equity probability | 0.04 adds a four-percentage-point call cushion; reasoning default |
| `preflop_entry_multiplier` | 1.15 | multiple of equal pot share | 1.15 avoids entering with below-average holdings; reasoning default |
| `preflop_raise_multiplier` | 1.8 | multiple of equal pot share | 1.8 selects a strong initial raising range; reasoning default |
| `value_raise_equity` | 0.62 | equity probability | 0.62 is the postflop value threshold and preflop threshold cap; reasoning default |
| `bet_pot_fraction` | 0.75 | fraction of pot after calling | 0.75 balances value and price; reasoning default |
| `opening_size_bb` | 2.5 | big blinds, street total | 2.5 is the minimum unopened preflop raise total; reasoning default |

**Results:** The first row is the qualification run. Seed 1000 rows are held out.

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (qualification) | 0 | 2 × 20 | +224.17 | 189.27 |
| random | 0 | 32 × 250 | +325.90 | 22.68 |
| v1 | 0 | 32 × 250 | -0.15 | 4.23 |
| v2 | 0 | 32 × 250 | -2.71 | 3.48 |
| random | 1000 | 32 × 250 | +318.52 | 22.49 |
| v1 | 1000 | 32 × 250 | -2.95 | 4.26 |
| v2 | 1000 | 32 × 250 | -3.88 | 3.82 |
| v4 | 1000 | 32 × 250 | +1.78 | 3.52 |
| v5 | 1000 | 32 × 250 | +0.81 | 3.84 |

**Verdict:** Kept as parent of v4 and v5 for its explicit price guard, not because of a statistically demonstrated win. Taken to held-out evaluation, then not deployed: no convincing self-play advantage and materially lower held-out RandomBot winnings.

**Decision line count:** 20 nonblank, noncomment body lines; 21 including the `def act` line. No additional decision helpers are defined in this version.

## Version 4

**Parent:** 3.

**What it does:** Use v3 with a lower preflop raise threshold: 1.4 times equal pot share instead of 1.8. This moves some playable preflop calls into raises.

**Why it exists:** Test whether the parent loses value and initiative by limping too many playable hands.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 64 | samples per decision | 64 balances sampling noise and runtime; reasoning default |
| `realization_fraction` | 0.85 | fraction of equity retained before river | 0.85 discounts incomplete equity realization; reasoning default |
| `call_margin_equity` | 0.04 | equity probability | 0.04 adds a four-percentage-point call cushion; reasoning default |
| `preflop_entry_multiplier` | 1.15 | multiple of equal pot share | 1.15 avoids entering with below-average holdings; reasoning default |
| `preflop_raise_multiplier` | 1.4 | multiple of equal pot share | 1.4 is a reasoning-led broader range experiment, tested against all earlier versions; not retained or further tuned |
| `value_raise_equity` | 0.62 | equity probability | 0.62 is the postflop value threshold and preflop threshold cap; reasoning default |
| `bet_pot_fraction` | 0.75 | fraction of pot after calling | 0.75 balances value and price; reasoning default |
| `opening_size_bb` | 2.5 | big blinds, street total | 2.5 is the minimum unopened preflop raise total; reasoning default |

**Results:** The first row is the qualification run. Seed 1000 rows are held out.

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (qualification) | 0 | 2 × 20 | +39.17 | 145.93 |
| random | 0 | 32 × 250 | +309.47 | 22.83 |
| v1 | 0 | 32 × 250 | -0.30 | 2.76 |
| v2 | 0 | 32 × 250 | -2.84 | 1.40 |
| v3 | 0 | 32 × 250 | +1.07 | 1.97 |

**Verdict:** Discarded. No clear gain against v1 or v3; the loss to v2 is approximately two reported SE. Broader aggression did not earn deployment, and the threshold was not tuned further.

**Decision line count:** 20 nonblank, noncomment body lines; 21 including the `def act` line. No additional decision helpers are defined in this version.

## Version 5

**Parent:** 3.

**What it does:** Use v3 with an odds-based correction for opposing raises on the current street. Each raise doubles the estimated loss odds; unlike v2, the transform preserves equity estimates of zero and one.

**Why it exists:** Test stronger-range caution without v2’s automatic discount of near-certain winning hands. This is one strategic addition to parent v3; two intermediate variables were inlined to stay within the line budget.

**Parameters:**

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 64 | samples per decision | 64 balances sampling noise and runtime; reasoning default |
| `range_strength_multiplier` | 2.0 | multiplier of loss odds per opposing raise | 2.0 doubles loss odds per raise; reasoning, tested against every earlier version and RandomBot |
| `realization_fraction` | 0.85 | fraction of equity retained before river | 0.85 discounts incomplete equity realization; reasoning default |
| `call_margin_equity` | 0.04 | equity probability | 0.04 adds a four-percentage-point call cushion; reasoning default |
| `preflop_entry_multiplier` | 1.15 | multiple of equal pot share | 1.15 avoids entering with below-average holdings; reasoning default |
| `preflop_raise_multiplier` | 1.8 | multiple of equal pot share | 1.8 selects a strong initial raising range; reasoning default |
| `value_raise_equity` | 0.62 | equity probability | 0.62 is the postflop value threshold and preflop threshold cap; reasoning default |
| `bet_pot_fraction` | 0.75 | fraction of pot after calling | 0.75 balances value and price; reasoning default |
| `opening_size_bb` | 2.5 | big blinds, street total | 2.5 is the minimum unopened preflop raise total; reasoning default |

**Results:** The first row is the qualification run. Seed 1000 rows are held out.

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random (qualification) | 0 | 2 × 20 | +160.42 | 137.53 |
| random | 0 | 32 × 250 | +116.13 | 11.99 |
| v1 | 0 | 32 × 250 | -2.59 | 2.18 |
| v2 | 0 | 32 × 250 | -2.48 | 1.75 |
| v3 | 0 | 32 × 250 | -0.63 | 2.03 |
| v4 | 0 | 32 × 250 | -0.59 | 1.76 |

**Verdict:** Discarded. All four seed-0 comparisons against earlier versions were negative but individually inconclusive. There was no demonstrated improvement, and RandomBot winnings fell substantially.

**Decision line count:** 21 nonblank, noncomment body lines; 22 including the `def act` line. No additional decision helpers are defined in this version.

## Final choice

**Deploy v1.** The later versions did not demonstrate a statistically clear gain against it. On held-out seed 1000, v1 scored +442.48 bb/100 (SE 26.84) against RandomBot, compared with v3’s +318.52 (SE 22.49). Against v2, v3, v4, and v5, v1 scored +2.93, −6.49, +0.49, and +3.00 bb/100 respectively; all are within two of their reported SE. The reverse v3-versus-v1 ordered run was also inconclusive: −2.95 (SE 4.26). These comparisons do not establish a winner among the closely related bots.

Earlier-version matchups were given priority: no later candidate earned retention through a clear advantage there. RandomBot performance and simplicity therefore break the tie in favor of v1. This is evidence for choosing the baseline within this experiment, not proof that it is optimal or robust against independent lineages. The best RandomBot-scoring finalist is also the deployed version.

**Known weaknesses:** Uniform opponent cards ignore selection into betting ranges; 64 samples are noisy; equity realization is a fixed discount rather than a positional or draw model. The bot has no positional opening ranges, explicit bluffing policy, side-pot eligibility correction, or board-texture-aware sizing. Preflop multiway equity includes all still-live seats even when some will fold. In v1, the value-raise threshold can override the call-price test, potentially overcommitting in expensive multiway pots; v3 tested that guard, but it did not establish a self-play benefit and won less against RandomBot. No cross-hand adaptation is possible by design.

## Reproduce

Run from this directory using the existing project environment:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/alik/Documents/Projects/poker_bots/poker_bots/.venv/bin/python -B bench.py v1:Bot random --seed 0
```

On this machine the verified interpreter is `/Users/alik/Documents/Projects/poker_bots/poker_bots/.venv/bin/python`. To avoid bytecode files anywhere outside this package, set `PYTHONDONTWRITEBYTECODE=1` when running. Held-out seed 1000 results have already been collected once per finalist/opponent and should not be used for further tuning.
