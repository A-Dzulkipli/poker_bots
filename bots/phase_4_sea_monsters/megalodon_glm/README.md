# megalodon_glm

- **Bot:** `megalodon_glm` — lineage `phase_4_sea_monsters/megalodon_glm`
- **Model:** GLM-5.3 (account:zai-individual-coding-plan/GLM-5.3)
- **Prompt version:** `botgen-v4`; **engine:** `decafb9`, engine_sha256 `295477eff7637056`
- **Wall clock used:** ~57 min (14:25–15:22 local); code executed: **yes**
- **`bench.py check`:**

```
python      /Users/alik/Documents/Projects/poker_bots/poker_bots/.venv/bin/python3 (3.13.12)
jax         0.11.2 (cpu)
poker_bots  /Users/alik/Documents/Projects/poker_bots/poker_bots/src/poker_bots
engine      sha256 295477eff7637056 (matches the contract)
smoke test  random vs aggressor, 40 hands in 2s: OK
```

All results below are 64 tables × 250 hands (16,000 hands per opponent per seed), candidate's
bb/100 with table-clustered SE, as printed by `bench.py`. Scratch logs are kept as
`v*_field_seed*.log` and `v*_vs_*.log` in this directory.

## v1 — sound stateless baseline

**What it does:** equity-driven heads-up strategy. Equity is measured against a random hand by
Monte Carlo (128 samples) and discounted by a factor of 0.58 per unit of opponent betting
pressure this street (0.5 per earlier-street bet). It opens every button to 2.5bb, 3-bets the
top of the range plus 15% light bluffs, raises limps with 0.55+ equity, and postflop bets
0.65-pot when checked to with value (equity ≥ 0.62), air (33% on the flop, decaying 0.55 per
street) and semi-bluffs (50%). Facing a bet it calls when discounted equity (times a 0.9
realization factor before the river) beats pot odds plus a 0.02 margin, and raises only at
equity ≥ 0.80.

**Why it exists:** a stateless baseline that is already strong, so the effect of tracking can
be measured against it. The architecture is borrowed from `windrunner_glm` v6, the strongest
phase-3 benchmark bot and my own earlier lineage; the parameter values are its deployed values.

**Parameters:** all constructor defaults above; taken from the windrunner v6 deployment (tuned
in phase 3 against the same field), not re-tuned here.

**Results (seed 0 / seed 1):**

| opponent | bb/100 (s0) | SE | bb/100 (s1) | SE |
|---|---|---|---|---|
| windrunner_glm | +6.42 | 5.82 | +6.60 | 6.50 |
| invoker_gpt | +8.74 | 10.50 | +13.17 | 10.11 |
| meepo_claude | +42.67 | 11.80 | +30.78 | 12.18 |
| hadrian_marlowe_gpt | +67.98 | 6.26 | +54.62 | 6.93 |
| valka_onderra_claude | +77.69 | 14.97 | +94.05 | 13.91 |
| aggressor | +40.83 | 5.05 | +33.53 | 5.38 |
| bassander_lin_glm | +153.37 | 15.21 | +153.82 | 15.08 |
| alan_wake_claude | +90.06 | 13.45 | +102.34 | 9.96 |
| max_payne_gpt | +51.19 | 4.92 | +49.42 | 5.54 |
| jesse_faden_glm | +189.11 | 17.76 | +216.94 | 20.81 |
| jack_joyce_gemini | +293.50 | 18.76 | +268.42 | 21.47 |
| random | +404.09 | 14.79 | +394.36 | 13.95 |

**Weighted score: +67.35 (seed 0), +66.99 (seed 1).** Line count: 36.

**Verdict:** kept as parent of v2 — a stable +67 baseline matching windrunner's own phase-3
score (+69.9), as expected for the same strategy.

## v2 — v1 plus opponent tracking

**What it does:** identical strategy to v1 plus two statistics accumulated in `observe` over
the session: the rate at which the opponent folds in hands where we put in any raise, and the
rate at which it bets or raises on hands that saw a flop. From a pseudo-count prior (10 hands
at 0.5), a high fold rate scales the bluff frequencies up (at most 1.6×, at least 0.6×) and a
low fold rate lowers the value-bet threshold by up to 0.06 (thin value against stations); a
high flop-bet rate relaxes the aggression discount by up to 0.2 (a near-100% bettor is betting
a near-random range, so our vs-random equity needs less discounting). The relief is one-sided,
so value-heavy infrequent bettors leave the discount untouched.

**Why it exists:** the phase-3 lesson that every benchmark opponent is stateless with fixed,
measurable tendencies (fold-to-bet rates 50–71%, bluff rates 21–50%). A session-long estimate
of those two tendencies is exactly what a bluff/value/call strategy wants to know.

**Parameters added:** `tracking_prior_hands=10` (pseudo-count hands), `prior_fold_rate=0.5`,
`prior_flop_bet_rate=0.5` (probabilities), `thin_value_slope=0.2` (value threshold drop per
unit fold-rate shortfall below the prior, probability units), `discount_relief_slope=0.4`
(discount relaxation per unit flop-bet-rate excess). Defaults chosen so adjustments stay
modest and capped; not tuned beyond that.

**Results (seed 0 / seed 1):**

| opponent | bb/100 (s0) | SE | bb/100 (s1) | SE |
|---|---|---|---|---|
| windrunner_glm | +2.06 | 5.69 | −4.37 | 7.00 |
| invoker_gpt | +18.41 | 10.37 | +15.86 | 11.84 |
| meepo_claude | +12.40 | 12.91 | +23.16 | 13.17 |
| hadrian_marlowe_gpt | +56.97 | 6.93 | +49.51 | 7.75 |
| valka_onderra_claude | +98.10 | 12.18 | +105.80 | 13.03 |
| aggressor | +51.63 | 5.07 | +60.13 | 4.74 |
| bassander_lin_glm | +144.75 | 15.61 | +155.80 | 17.11 |
| alan_wake_claude | +96.81 | 14.13 | +107.00 | 11.43 |
| max_payne_gpt | +46.47 | 5.15 | +59.33 | 4.48 |
| jesse_faden_glm | +205.24 | 18.92 | +202.30 | 21.20 |
| jack_joyce_gemini | +300.85 | 20.90 | +322.56 | 17.83 |
| random | +392.41 | 15.32 | +392.57 | 15.78 |

**Weighted score: +66.13 (seed 0), +70.08 (seed 1).** Line count: 48.

**Verdict:** kept as parent of v3. The two-seed weighted mean (+68.1 vs v1's +67.2) is within
noise, but v2 beats v1 head-to-head **+15.26 ± 7.46** (seed 2, 64×250), and the per-opponent
pattern is consistent where tracking should bite: aggressor +15/+27 (the discount relief),
valka +20/+12, jack_joyce +7/+54, invoker +10/+3. The seed-0 meepo dip (−30) did not repeat
(−7 on seed 1).

## v3 — postflop-scoped fold statistic

**What it does:** the same bot as v2 with one measurement change: the fold rate now counts
only hands where we made a **postflop** bet or raise, with the prior moved to 0.55 (the
phase-3 field folds 50–71% to postflop bets). In v2 the denominator included our button opens,
which happen 100% of the time, so a big-blind pot-odds caller (defends wide preflop, folds
postflop) read as a non-folder.

**Why it exists:** v2's fold estimate is diluted by preflop opens exactly where the bluff
scaling wants a postflop answer.

**Parameters changed:** `prior_fold_rate` 0.5 → 0.55; no others.

**Results (seed 0 / seed 1):**

| opponent | bb/100 (s0) | SE | bb/100 (s1) | SE |
|---|---|---|---|---|
| windrunner_glm | −1.05 | 6.11 | +4.76 | 6.14 |
| invoker_gpt | +14.52 | 9.66 | +10.43 | 10.71 |
| meepo_claude | +22.96 | 13.20 | +20.48 | 12.05 |
| hadrian_marlowe_gpt | +42.88 | 7.69 | +52.69 | 6.16 |
| valka_onderra_claude | +98.32 | 13.59 | +104.57 | 12.62 |
| aggressor | +46.04 | 5.13 | +52.88 | 5.07 |
| bassander_lin_glm | +170.31 | 15.30 | +159.65 | 18.15 |
| alan_wake_claude | +100.50 | 12.82 | +115.92 | 11.37 |
| max_payne_gpt | +48.84 | 5.14 | +62.00 | 4.35 |
| jesse_faden_glm | +195.30 | 20.02 | +229.92 | 20.71 |
| jack_joyce_gemini | +271.94 | 20.98 | +331.18 | 19.79 |
| random | +395.21 | 14.08 | +416.19 | 15.69 |

**Weighted score: +65.74 (seed 0), +71.94 (seed 1).** Line count: 49.

**Verdict:** kept and deployed. The two-seed weighted mean (+68.84) edges out v2 (+68.10) and
v1 (+67.17), and v3 beats v2 head-to-head **+15.52 ± 6.84** (seed 2, 64×250) — the same margin
by which v2 beat v1, so each tracking step earned its keep directly. Per-opponent differences
between v2 and v3 are all within noise; the head-to-head chain and the mean both point the
same way.

## Opponent tracking (deployed version, v3)

**What is tracked** (float32 counters in `state`, updated only in `observe`, one update per
finished hand, no Monte Carlo there):

1. `hands_i_bet_postflop` / `opponent_folded_those` — of hands where we made a postflop bet or
   raise, how many ended with the opponent folding. Prior: 10 pseudo-count hands at a fold
   rate of 0.55.
2. `flops_seen` / `opponent_bet_flops` — of hands that saw a flop, in how many the opponent
   bet or raised postflop. Prior: 10 pseudo-count hands at a rate of 0.5.

**Which decisions each statistic changes:**

- `fold_to_my_postflop_bet` scales both bluff frequencies by `clip(rate / 0.55, 0.6, 1.6)` —
  at most 1.6× bluffs against a frequent folder, at most 0.6× against a caller — and lowers
  the value-bet equity threshold by up to `0.3 × 0.2 = 0.06` when the rate sits 0.3 below the
  prior (thin value against a station).
- `flop_bet_rate` relaxes the aggression discount on called bets by
  `0.4 × (rate − 0.5)`, one-sided (the discount can rise from 0.58 to at most 1.0 but never
  fall): an opponent that bets nearly every flop is betting a near-random range, so our
  vs-random equity needs less discounting to justify a call. This is the aggressor adjustment.

**Evidence it helps:** with the strategy held fixed, v3 (tracking) beats v1 (no tracking)
head-to-head by the v2 intermediate: v2 beat v1 **+15.26 ± 7.46** and v3 beat v2
**+15.52 ± 6.84** (seed 2, 64×250 each). Against the field the biggest tracking-driven move is
the aggressor matchup, where the discount relief lifts the result from v1's +40.8/+33.5 to
+51.6/+60.1 (v2) and +46.0/+52.9 (v3). The early/late split (first 50 hands vs rest, averaged
over seeds 0/1) shows the session learning: vs valka +61/+103 (v3), vs jack_joyce +267/+310,
vs meepo +27/+20 — while the bot still plays soundly from hand 1 through the priors.

## Final choice

**Deployed: v3** (`__init__.py` → `from .v3 import Bot as DeployedBot`).

- Two-seed weighted score (seeds 0/1): v1 +67.17, v2 +68.10, v3 +68.84.
- Head-to-head chain: v2 > v1 (+15.3 ± 7.5), v3 > v2 (+15.5 ± 6.8) — each version beats its
  parent, so no warning signs there.
- v3 is also the clearest of the three to read (the fold statistic measures exactly the
  decision it feeds), and its one-line difference from v2 is easy to review.

**Held-out seed 1000 (run once per finalist, 64×250 per opponent):**

| opponent | v3 bb/100 | SE | v2 bb/100 | SE |
|---|---|---|---|---|
| windrunner_glm | +18.07 | 6.05 | +5.89 | 6.83 |
| invoker_gpt | +16.66 | 10.09 | +17.17 | 9.85 |
| meepo_claude | +27.08 | 11.34 | +15.55 | 12.23 |
| hadrian_marlowe_gpt | +42.11 | 8.10 | +52.39 | 6.95 |
| valka_onderra_claude | +59.58 | 14.04 | +76.21 | 14.39 |
| aggressor | +38.91 | 5.79 | +53.90 | 4.90 |
| bassander_lin_glm | +161.44 | 13.73 | +160.23 | 15.53 |
| alan_wake_claude | +104.80 | 13.69 | +89.38 | 14.31 |
| max_payne_gpt | +54.54 | 5.39 | +57.19 | 5.29 |
| jesse_faden_glm | +223.26 | 22.42 | +205.34 | 19.04 |
| jack_joyce_gemini | +320.68 | 20.15 | +325.42 | 21.52 |
| random | +381.90 | 14.51 | +385.18 | 14.78 |

**Held-out weighted score: v3 +66.70, v2 +66.19.** Both finalists clear every opponent on the
held-out seed; v3 edges v2, matching its seeds-0/1 lead and its head-to-head win, so the
deployment of v3 needed no tie-break. Across all three seeds v3's weighted scores are
+65.74 / +71.94 / +66.70 (mean +68.1) against v1's +67.35 / +66.99 — i.e. the tracking family
adds about +1 weighted point on average against this stateless field, with much larger
head-to-head gains (+15 per tracking step) that should matter more against the other phase-4
lineages' adaptive bots.

**Known weaknesses / how an adapting opponent could exploit the tracking:** the bluff scale is
a session-long estimate, so an opponent that calls down for 50 hands and then switches to
raising our stabs turns our own statistic against us (we keep bluffing into a raised fold
rate until the estimate decays). The discount relief rewards frequent betting, so a strong
opponent could bet every flop for value and fold to further aggression, making our relaxed
calls pay too much. Both adjustments are capped (1.6×, 0.06, +0.2 discount), so the worst
case is bounded, and the priors keep the first 50 hands of every session near the v1
strategy.
