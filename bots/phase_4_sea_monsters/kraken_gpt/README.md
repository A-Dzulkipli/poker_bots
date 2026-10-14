# kraken_gpt — v6 deployed

- Lineage: `phase_4_sea_monsters/kraken_gpt`.
- Model: GPT-6 (Codex; exact serving variant unavailable).
- Prompt: `botgen-v4`.
- Engine: `decafb9`, SHA-256 prefix `295477eff7637056`.
- Executed: **yes**. Every version passed its required 40-hand smoke test.
- Wall-clock development time: **46.6 minutes**, 2026-10-14T14:27:03+10:30 to 2026-10-14T15:13:40.698846+10:30 (Australia/Adelaide). Initial permission troubleshooting before this successful-check session is excluded.
- Total recorded elapsed time including initial permission troubleshooting: **49.2 minutes**.
- The one-shot reminder service rejected the schedule; elapsed time was checked between steps.
- Deployment: `__init__.py` imports `Bot` from `v6.py` as `DeployedBot`.

## Environment check

The first sandboxed attempts failed while opening the existing uv cache. The approved retry returned:

```text
python      /Users/alik/Documents/Projects/poker_bots/poker_bots/.venv/bin/python3 (3.13.12)
jax         0.11.2 (cpu)
poker_bots  /Users/alik/Documents/Projects/poker_bots/poker_bots/src/poker_bots
engine      sha256 295477eff7637056 (matches the contract)
smoke test  random vs aggressor, 40 hands in 2s: OK
```

## Protocol and evidence

All performance results below use 250-hand sessions with fresh state at each table, balanced candidate seats, 100bb initial stacks, and the supplied rebuy/cash-out rules. Smoke tests use 2 × 20 hands and are excluded from performance tables. Development uses seeds 0–5; seed 1000 was used exactly once for each of the three finalists. Standard errors are table-clustered, as printed by the supplied benchmark. Small differences under roughly two standard errors are not treated as conclusive.

The weighted score is the supplied twelve-opponent score, with rank-dependent weights that differ for positive and negative results. The first-50 and remaining-200 columns are bb/100, not cumulative winnings. Early/late differences alone do not prove learning: stack distributions and which hands reach later streets also change.

The first full-field run took approximately 233 seconds by summing the benchmark's printed timings. Some later runs overlapped, so their individual durations include contention.

| Version | Seed | Tables × hands per opponent | Weighted score |
|---|---:|---:|---:|
| v1 | 0 | 64 × 250 | +66.40 |
| v2 | 0 | 64 × 250 | +69.39 |
| v3 | 0 | 64 × 250 | +72.42 |
| v3 | 1 | 128 × 250 | +73.23 |
| v4 | 0 | 64 × 250 | +79.04 |
| v5 | 0 | 64 × 250 | +68.27 |
| v6 | 0 | 64 × 250 | +78.34 |
| v6 | 1000 | 128 × 250 | +79.64 |
| v7 | 0 | 64 × 250 | +83.54 |
| v7 | 1000 | 128 × 250 | +81.70 |
| v8 | 0 | 64 × 250 | +73.91 |
| v9 | 0 | 64 × 250 | +80.80 |
| v10 | 0 | 64 × 250 | +76.00 |
| v10 | 1000 | 128 × 250 | +77.01 |
| v11 | 0 | 64 × 250 | +66.30 |

## v1

**What it does.** Opens every button, isolates stronger hands, and mixes light three-bets. Postflop it bets value and some weaker hands, then uses betting pressure to narrow the opponent range before calling or raising.

**Why it exists.** Establish a compact stateless baseline before adding any learning.

**Decision line count:** 50. Imports, comments, blank lines, constructor, initialization and metadata are excluded.

**Parameters.** All probabilities are dimensionless. Constants for card/street indices, array layout, clipping to [0, 1], and floating-point numerical protection are structural rather than fitted parameters.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | Reasoned/source-informed default; no fitted table |
| `open_size_bb` | 2.5 | big blinds, raise-to | Reasoned/source-informed default; no fitted table |
| `reraise_multiple` | 3.0 | multiple of the current preflop bet | Reasoned/source-informed default; no fitted table |
| `threebet_equity` | 0.63 | equity probability | Reasoned/source-informed default; no fitted table |
| `light_threebet_frequency` | 0.15 | mixing probability | Reasoned/source-informed default; no fitted table |
| `isolate_equity` | 0.55 | equity probability | Reasoned/source-informed default; no fitted table |
| `value_equity` | 0.62 | equity probability | Reasoned/source-informed default; no fitted table |
| `raise_equity` | 0.85 | adjusted equity probability | Reasoned/source-informed default; no fitted table |
| `bluff_frequency` | 0.35 | mixing probability with air | Reasoned/source-informed default; no fitted table |
| `semibluff_frequency` | 0.5 | mixing probability with medium equity | Reasoned/source-informed default; no fitted table |
| `bluff_split_equity` | 0.45 | equity probability | Reasoned/source-informed default; no fitted table |
| `bluff_street_decay` | 0.65 | frequency multiplier per later street | Reasoned/source-informed default; no fitted table |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | Reasoned/source-informed default; no fitted table |
| `range_retention` | 0.58 | range fraction retained per ordinary pressure unit | Reasoned/source-informed default; no fitted table |
| `prior_street_weight` | 0.5 | current-street bet equivalents | Reasoned/source-informed default; no fitted table |
| `realization_fraction` | 0.9 | fraction of equity realized before river | Reasoned/source-informed default; no fitted table |
| `call_margin_equity` | 0.02 | equity probability above pot odds | Reasoned/source-informed default; no fitted table |

**Results.**

| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |
|---|---:|---:|---:|---:|---:|---:|
| aggressor | 0 | 64 × 250 | +44.48 | 4.62 | +62.5 | +40.0 |
| alan_wake_claude | 0 | 64 × 250 | +85.96 | 10.73 | +65.5 | +91.1 |
| bassander_lin_glm | 0 | 64 × 250 | +159.68 | 12.99 | +207.7 | +147.7 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +52.33 | 6.17 | +54.4 | +51.8 |
| invoker_gpt | 0 | 64 × 250 | +16.36 | 8.29 | +36.8 | +11.2 |
| jack_joyce_gemini | 0 | 64 × 250 | +233.34 | 16.16 | +219.9 | +236.7 |
| jesse_faden_glm | 0 | 64 × 250 | +238.26 | 14.67 | +222.7 | +242.1 |
| max_payne_gpt | 0 | 64 × 250 | +52.55 | 4.46 | +63.5 | +49.8 |
| meepo_claude | 0 | 64 × 250 | +37.87 | 10.22 | +48.6 | +35.2 |
| random | 0 | 64 × 250 | +286.79 | 12.87 | +317.4 | +279.1 |
| valka_onderra_claude | 0 | 64 × 250 | +72.24 | 11.85 | +114.6 | +61.6 |
| windrunner_glm | 0 | 64 × 250 | +16.15 | 7.17 | +17.2 | +15.9 |

Weighted field scores: seed 0: **+66.40** (64 × 250 per opponent).

**Verdict.** Kept as parent of v2. Final selection uses the held-out evidence and parent comparisons below.

## v2

**What it does.** Adds smoothed counts of free betting and folds to our bets or raises. Frequent bettors receive less range credit, and frequent folders face more bluffs. Its starting strategy is otherwise exactly v1.

**Why it exists.** Test whether observed behavior improves the same strategy without relying on opponent identity.

**Decision line count:** 72. Imports, comments, blank lines, constructor, initialization and metadata are excluded.

**Parameters.** All probabilities are dimensionless. Constants for card/street indices, array layout, clipping to [0, 1], and floating-point numerical protection are structural rather than fitted parameters.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | Inherited from v1 |
| `open_size_bb` | 2.5 | big blinds, raise-to | Inherited from v1 |
| `reraise_multiple` | 3.0 | multiple of the current preflop bet | Inherited from v1 |
| `threebet_equity` | 0.63 | equity probability | Inherited from v1 |
| `light_threebet_frequency` | 0.15 | mixing probability | Inherited from v1 |
| `isolate_equity` | 0.55 | equity probability | Inherited from v1 |
| `value_equity` | 0.62 | equity probability | Inherited from v1 |
| `raise_equity` | 0.85 | adjusted equity probability | Inherited from v1 |
| `bluff_frequency` | 0.35 | mixing probability with air | Inherited from v1 |
| `semibluff_frequency` | 0.5 | mixing probability with medium equity | Inherited from v1 |
| `bluff_split_equity` | 0.45 | equity probability | Inherited from v1 |
| `bluff_street_decay` | 0.65 | frequency multiplier per later street | Inherited from v1 |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | Inherited from v1 |
| `range_retention` | 0.58 | range fraction retained per ordinary pressure unit | Inherited from v1 |
| `prior_street_weight` | 0.5 | current-street bet equivalents | Inherited from v1 |
| `realization_fraction` | 0.9 | fraction of equity realized before river | Inherited from v1 |
| `call_margin_equity` | 0.02 | equity probability above pot odds | Inherited from v1 |
| `prior_events` | 20.0 | pseudo-opportunities per estimated rate | Reasoned/source-informed default; no fitted table |
| `prior_bet_rate` | 0.4 | probability of betting when checking is free | Reasoned/source-informed default; no fitted table |
| `prior_fold_rate` | 0.5 | probability of folding to our aggression | Reasoned/source-informed default; no fitted table |
| `betting_adjustment` | 0.65 | retention change per unit of observed frequency deviation | Reasoned/source-informed default; no fitted table |
| `bluff_adjustment` | 1.25 | bluff probability change per unit of fold-rate deviation | Reasoned/source-informed default; no fitted table |

**Results.**

| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |
|---|---:|---:|---:|---:|---:|---:|
| aggressor | 0 | 64 × 250 | +45.66 | 6.19 | +47.5 | +45.2 |
| alan_wake_claude | 0 | 64 × 250 | +124.27 | 9.02 | +96.9 | +131.1 |
| bassander_lin_glm | 0 | 64 × 250 | +158.96 | 13.57 | +151.8 | +160.7 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +60.78 | 6.83 | +70.7 | +58.3 |
| invoker_gpt | 0 | 64 × 250 | +34.94 | 8.73 | +30.1 | +36.2 |
| jack_joyce_gemini | 0 | 64 × 250 | +195.02 | 15.34 | +203.6 | +192.9 |
| jesse_faden_glm | 0 | 64 × 250 | +221.88 | 14.73 | +179.3 | +232.5 |
| max_payne_gpt | 0 | 64 × 250 | +52.29 | 4.92 | +57.2 | +51.1 |
| meepo_claude | 0 | 64 × 250 | +13.21 | 10.85 | +35.6 | +7.6 |
| random | 0 | 64 × 250 | +242.48 | 13.19 | +276.5 | +234.0 |
| valka_onderra_claude | 0 | 64 × 250 | +93.33 | 10.53 | +75.1 | +97.9 |
| windrunner_glm | 0 | 64 × 250 | +16.70 | 8.48 | +10.7 | +18.2 |
| v1:Bot | 1 | 128 × 250 | +23.92 | 4.70 | +19.4 | +25.0 |

Weighted field scores: seed 0: **+69.39** (64 × 250 per opponent).

**Verdict.** Kept as parent of v3. Final selection uses the held-out evidence and parent comparisons below.

## v3

**What it does.** Keeps v2’s two learned rates and maintains its bluff frequency through the turn and river. Bet sizes and all other parameters stay unchanged.

**Why it exists.** Test whether automatically reducing later-street bluffs gives up profitable opportunities against persistent folders.

**Decision line count:** 72. Imports, comments, blank lines, constructor, initialization and metadata are excluded.

**Parameters.** All probabilities are dimensionless. Constants for card/street indices, array layout, clipping to [0, 1], and floating-point numerical protection are structural rather than fitted parameters.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | Inherited from v2 |
| `open_size_bb` | 2.5 | big blinds, raise-to | Inherited from v2 |
| `reraise_multiple` | 3.0 | multiple of the current preflop bet | Inherited from v2 |
| `threebet_equity` | 0.63 | equity probability | Inherited from v2 |
| `light_threebet_frequency` | 0.15 | mixing probability | Inherited from v2 |
| `isolate_equity` | 0.55 | equity probability | Inherited from v2 |
| `value_equity` | 0.62 | equity probability | Inherited from v2 |
| `raise_equity` | 0.85 | adjusted equity probability | Inherited from v2 |
| `bluff_frequency` | 0.35 | mixing probability with air | Inherited from v2 |
| `semibluff_frequency` | 0.5 | mixing probability with medium equity | Inherited from v2 |
| `bluff_split_equity` | 0.45 | equity probability | Inherited from v2 |
| `bluff_street_decay` | 1.0 | frequency multiplier per later street | One-variable development experiment; v2 used 0.65 |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | Inherited from v2 |
| `range_retention` | 0.58 | range fraction retained per ordinary pressure unit | Inherited from v2 |
| `prior_street_weight` | 0.5 | current-street bet equivalents | Inherited from v2 |
| `realization_fraction` | 0.9 | fraction of equity realized before river | Inherited from v2 |
| `call_margin_equity` | 0.02 | equity probability above pot odds | Inherited from v2 |
| `prior_events` | 20.0 | pseudo-opportunities per estimated rate | Inherited from v2 |
| `prior_bet_rate` | 0.4 | probability of betting when checking is free | Inherited from v2 |
| `prior_fold_rate` | 0.5 | probability of folding to our aggression | Inherited from v2 |
| `betting_adjustment` | 0.65 | retention change per unit of observed frequency deviation | Inherited from v2 |
| `bluff_adjustment` | 1.25 | bluff probability change per unit of fold-rate deviation | Inherited from v2 |

**Results.**

| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |
|---|---:|---:|---:|---:|---:|---:|
| aggressor | 0 | 64 × 250 | +52.18 | 6.32 | +50.9 | +52.5 |
| alan_wake_claude | 0 | 64 × 250 | +128.19 | 8.65 | +110.5 | +132.6 |
| bassander_lin_glm | 0 | 64 × 250 | +131.42 | 12.52 | +116.7 | +135.1 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +65.05 | 6.57 | +56.8 | +67.1 |
| invoker_gpt | 0 | 64 × 250 | +30.23 | 8.76 | +29.6 | +30.4 |
| jack_joyce_gemini | 0 | 64 × 250 | +195.16 | 14.08 | +165.7 | +202.5 |
| jesse_faden_glm | 0 | 64 × 250 | +198.43 | 15.48 | +172.3 | +205.0 |
| max_payne_gpt | 0 | 64 × 250 | +51.62 | 4.95 | +54.9 | +50.8 |
| meepo_claude | 0 | 64 × 250 | +41.93 | 10.96 | +36.4 | +43.3 |
| random | 0 | 64 × 250 | +256.62 | 12.61 | +294.2 | +247.2 |
| valka_onderra_claude | 0 | 64 × 250 | +85.72 | 10.11 | +69.3 | +89.8 |
| windrunner_glm | 0 | 64 × 250 | +31.82 | 7.26 | +15.8 | +35.8 |
| aggressor | 1 | 128 × 250 | +35.86 | 4.66 | +42.2 | +34.3 |
| alan_wake_claude | 1 | 128 × 250 | +155.93 | 6.42 | +131.0 | +162.2 |
| bassander_lin_glm | 1 | 128 × 250 | +158.82 | 9.30 | +149.1 | +161.2 |
| hadrian_marlowe_gpt | 1 | 128 × 250 | +56.71 | 4.50 | +60.6 | +55.7 |
| invoker_gpt | 1 | 128 × 250 | +16.41 | 6.74 | +2.5 | +19.9 |
| jack_joyce_gemini | 1 | 128 × 250 | +238.11 | 12.07 | +216.5 | +243.5 |
| jesse_faden_glm | 1 | 128 × 250 | +235.30 | 10.64 | +199.9 | +244.2 |
| max_payne_gpt | 1 | 128 × 250 | +52.54 | 3.63 | +58.0 | +51.2 |
| meepo_claude | 1 | 128 × 250 | +28.05 | 6.82 | +32.2 | +27.0 |
| random | 1 | 128 × 250 | +226.01 | 9.07 | +202.4 | +231.9 |
| valka_onderra_claude | 1 | 128 × 250 | +96.09 | 7.88 | +73.6 | +101.7 |
| windrunner_glm | 1 | 128 × 250 | +30.83 | 5.54 | +33.2 | +30.2 |
| v1:Bot | 1 | 128 × 250 | +30.04 | 5.18 | +32.2 | +29.5 |
| v2:Bot | 1 | 128 × 250 | +5.87 | 5.46 | +0.9 | +7.1 |
| windrunner_glm | 2 | 128 × 250 | +36.59 | 5.71 | +25.6 | +39.3 |

Weighted field scores: seed 0: **+72.42** (64 × 250 per opponent); seed 1: **+73.23** (128 × 250 per opponent).

**Verdict.** Kept as parent of v4, v5, v6. Final selection uses the held-out evidence and parent comparisons below.

## v4

**What it does.** Uses v3’s decisions and learning, with postflop bets and raises sized at 90% instead of 65% of the pot after calling.

**Why it exists.** Test larger value extraction and greater fold pressure; check its robustness against its parent.

**Decision line count:** 72. Imports, comments, blank lines, constructor, initialization and metadata are excluded.

**Parameters.** All probabilities are dimensionless. Constants for card/street indices, array layout, clipping to [0, 1], and floating-point numerical protection are structural rather than fitted parameters.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | Inherited from v3 |
| `open_size_bb` | 2.5 | big blinds, raise-to | Inherited from v3 |
| `reraise_multiple` | 3.0 | multiple of the current preflop bet | Inherited from v3 |
| `threebet_equity` | 0.63 | equity probability | Inherited from v3 |
| `light_threebet_frequency` | 0.15 | mixing probability | Inherited from v3 |
| `isolate_equity` | 0.55 | equity probability | Inherited from v3 |
| `value_equity` | 0.62 | equity probability | Inherited from v3 |
| `raise_equity` | 0.85 | adjusted equity probability | Inherited from v3 |
| `bluff_frequency` | 0.35 | mixing probability with air | Inherited from v3 |
| `semibluff_frequency` | 0.5 | mixing probability with medium equity | Inherited from v3 |
| `bluff_split_equity` | 0.45 | equity probability | Inherited from v3 |
| `bluff_street_decay` | 1.0 | frequency multiplier per later street | Inherited from v3 |
| `bet_pot_fraction` | 0.9 | fraction of pot after calling | One-variable development experiment; v3 used 0.65 |
| `range_retention` | 0.58 | range fraction retained per ordinary pressure unit | Inherited from v3 |
| `prior_street_weight` | 0.5 | current-street bet equivalents | Inherited from v3 |
| `realization_fraction` | 0.9 | fraction of equity realized before river | Inherited from v3 |
| `call_margin_equity` | 0.02 | equity probability above pot odds | Inherited from v3 |
| `prior_events` | 20.0 | pseudo-opportunities per estimated rate | Inherited from v3 |
| `prior_bet_rate` | 0.4 | probability of betting when checking is free | Inherited from v3 |
| `prior_fold_rate` | 0.5 | probability of folding to our aggression | Inherited from v3 |
| `betting_adjustment` | 0.65 | retention change per unit of observed frequency deviation | Inherited from v3 |
| `bluff_adjustment` | 1.25 | bluff probability change per unit of fold-rate deviation | Inherited from v3 |

**Results.**

| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |
|---|---:|---:|---:|---:|---:|---:|
| aggressor | 0 | 64 × 250 | +18.51 | 8.07 | +35.9 | +14.2 |
| alan_wake_claude | 0 | 64 × 250 | +188.94 | 13.63 | +181.7 | +190.8 |
| bassander_lin_glm | 0 | 64 × 250 | +157.53 | 12.82 | +93.2 | +173.6 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +53.11 | 8.08 | +62.4 | +50.8 |
| invoker_gpt | 0 | 64 × 250 | +12.20 | 10.70 | -1.8 | +15.7 |
| jack_joyce_gemini | 0 | 64 × 250 | +270.03 | 21.82 | +279.3 | +267.7 |
| jesse_faden_glm | 0 | 64 × 250 | +267.18 | 17.97 | +258.3 | +269.4 |
| max_payne_gpt | 0 | 64 × 250 | +58.27 | 5.53 | +59.4 | +58.0 |
| meepo_claude | 0 | 64 × 250 | +28.13 | 12.13 | +50.0 | +22.7 |
| random | 0 | 64 × 250 | +333.68 | 12.77 | +345.8 | +330.7 |
| valka_onderra_claude | 0 | 64 × 250 | +112.61 | 13.81 | +102.2 | +115.2 |
| windrunner_glm | 0 | 64 × 250 | +36.93 | 8.06 | +3.4 | +45.3 |
| v3:Bot | 2 | 128 × 250 | -22.74 | 8.62 | +0.5 | -28.5 |
| windrunner_glm | 2 | 128 × 250 | +35.70 | 5.22 | +44.0 | +33.6 |
| v1:Bot | 3 | 128 × 250 | +15.08 | 5.88 | +7.1 | +17.1 |
| v2:Bot | 3 | 128 × 250 | -14.61 | 8.64 | -34.6 | -9.6 |

Weighted field scores: seed 0: **+79.04** (64 × 250 per opponent).

**Verdict.** Retained as an informative sizing experiment, but not deployed: its field gain came with a material loss to v3.

## v5

**What it does.** Adds occasional low-equity raises to v3 when facing the first opposing bet on a street. Their frequency is scaled by observed free-betting and fold-to-aggression rates.

**Why it exists.** Test whether fighting back with bluff raises improves on passive bluff catching and value raises.

**Decision line count:** 76. Imports, comments, blank lines, constructor, initialization and metadata are excluded.

**Parameters.** All probabilities are dimensionless. Constants for card/street indices, array layout, clipping to [0, 1], and floating-point numerical protection are structural rather than fitted parameters.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | Inherited from v3 |
| `open_size_bb` | 2.5 | big blinds, raise-to | Inherited from v3 |
| `reraise_multiple` | 3.0 | multiple of the current preflop bet | Inherited from v3 |
| `threebet_equity` | 0.63 | equity probability | Inherited from v3 |
| `light_threebet_frequency` | 0.15 | mixing probability | Inherited from v3 |
| `isolate_equity` | 0.55 | equity probability | Inherited from v3 |
| `value_equity` | 0.62 | equity probability | Inherited from v3 |
| `raise_equity` | 0.85 | adjusted equity probability | Inherited from v3 |
| `bluff_frequency` | 0.35 | mixing probability with air | Inherited from v3 |
| `semibluff_frequency` | 0.5 | mixing probability with medium equity | Inherited from v3 |
| `bluff_split_equity` | 0.45 | equity probability | Inherited from v3 |
| `bluff_street_decay` | 1.0 | frequency multiplier per later street | Inherited from v3 |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | Inherited from v3 |
| `range_retention` | 0.58 | range fraction retained per ordinary pressure unit | Inherited from v3 |
| `prior_street_weight` | 0.5 | current-street bet equivalents | Inherited from v3 |
| `realization_fraction` | 0.9 | fraction of equity realized before river | Inherited from v3 |
| `call_margin_equity` | 0.02 | equity probability above pot odds | Inherited from v3 |
| `prior_events` | 20.0 | pseudo-opportunities per estimated rate | Inherited from v3 |
| `prior_bet_rate` | 0.4 | probability of betting when checking is free | Inherited from v3 |
| `prior_fold_rate` | 0.5 | probability of folding to our aggression | Inherited from v3 |
| `betting_adjustment` | 0.65 | retention change per unit of observed frequency deviation | Inherited from v3 |
| `bluff_adjustment` | 1.25 | bluff probability change per unit of fold-rate deviation | Inherited from v3 |
| `bluff_raise_frequency` | 0.35 | maximum bluff-raise probability before rate scaling | Reasoned/source-informed default; no fitted table |

**Results.**

| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |
|---|---:|---:|---:|---:|---:|---:|
| aggressor | 0 | 64 × 250 | +43.01 | 6.70 | +53.0 | +40.5 |
| alan_wake_claude | 0 | 64 × 250 | +132.31 | 8.31 | +112.1 | +137.4 |
| bassander_lin_glm | 0 | 64 × 250 | +133.97 | 12.24 | +101.6 | +142.1 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +63.65 | 7.26 | +49.2 | +67.3 |
| invoker_gpt | 0 | 64 × 250 | +14.74 | 9.40 | +4.3 | +17.4 |
| jack_joyce_gemini | 0 | 64 × 250 | +194.14 | 14.09 | +171.4 | +199.8 |
| jesse_faden_glm | 0 | 64 × 250 | +201.90 | 15.54 | +176.5 | +208.3 |
| max_payne_gpt | 0 | 64 × 250 | +52.27 | 5.14 | +53.5 | +52.0 |
| meepo_claude | 0 | 64 × 250 | +33.20 | 10.83 | +30.5 | +33.9 |
| random | 0 | 64 × 250 | +247.12 | 12.22 | +297.7 | +234.5 |
| valka_onderra_claude | 0 | 64 × 250 | +73.65 | 11.73 | +58.4 | +77.5 |
| windrunner_glm | 0 | 64 × 250 | +37.66 | 7.77 | +22.5 | +41.5 |
| v3:Bot | 2 | 128 × 250 | -0.50 | 7.32 | -2.4 | -0.0 |
| v1:Bot | 3 | 128 × 250 | +20.60 | 6.16 | +22.9 | +20.0 |
| v2:Bot | 3 | 128 × 250 | -7.27 | 7.16 | +11.3 | -11.9 |
| v4:Bot | 3 | 128 × 250 | +1.75 | 7.94 | +10.7 | -0.5 |

Weighted field scores: seed 0: **+68.27** (64 × 250 per opponent).

**Verdict.** Discarded as a strategy branch: extra bluff-raise complexity did not establish an improvement over v3.

## v6

**What it does.** Separates bets made when checking is free from raises made when facing our aggression. Each gets its own observed frequency and range-strength prior, while v3’s bet sizes and bluff policy remain unchanged.

**Why it exists.** A player can bet a wide range but re-raise a narrow one. Avoid applying a loose free-betting estimate to every raise.

**Decision line count:** 83. Imports, comments, blank lines, constructor, initialization and metadata are excluded.

**Parameters.** All probabilities are dimensionless. Constants for card/street indices, array layout, clipping to [0, 1], and floating-point numerical protection are structural rather than fitted parameters.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | Inherited from v3 |
| `open_size_bb` | 2.5 | big blinds, raise-to | Inherited from v3 |
| `reraise_multiple` | 3.0 | multiple of the current preflop bet | Inherited from v3 |
| `threebet_equity` | 0.63 | equity probability | Inherited from v3 |
| `light_threebet_frequency` | 0.15 | mixing probability | Inherited from v3 |
| `isolate_equity` | 0.55 | equity probability | Inherited from v3 |
| `value_equity` | 0.62 | equity probability | Inherited from v3 |
| `raise_equity` | 0.85 | adjusted equity probability | Inherited from v3 |
| `bluff_frequency` | 0.35 | mixing probability with air | Inherited from v3 |
| `semibluff_frequency` | 0.5 | mixing probability with medium equity | Inherited from v3 |
| `bluff_split_equity` | 0.45 | equity probability | Inherited from v3 |
| `bluff_street_decay` | 1.0 | frequency multiplier per later street | Inherited from v3 |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | Inherited from v3 |
| `range_retention` | 0.58 | range fraction retained per ordinary pressure unit | Inherited from v3 |
| `prior_street_weight` | 0.5 | current-street bet equivalents | Inherited from v3 |
| `realization_fraction` | 0.9 | fraction of equity realized before river | Inherited from v3 |
| `call_margin_equity` | 0.02 | equity probability above pot odds | Inherited from v3 |
| `prior_events` | 20.0 | pseudo-opportunities per estimated rate | Inherited from v3 |
| `prior_bet_rate` | 0.4 | probability of betting when checking is free | Inherited from v3 |
| `prior_fold_rate` | 0.5 | probability of folding to our aggression | Inherited from v3 |
| `betting_adjustment` | 0.65 | retention change per unit of observed frequency deviation | Inherited from v3 |
| `bluff_adjustment` | 1.25 | bluff probability change per unit of fold-rate deviation | Inherited from v3 |
| `prior_raise_rate` | 0.15 | probability of raising when facing our aggression | Reasoned/source-informed default; no fitted table |
| `reraise_range_retention` | 0.4 | range fraction retained per re-raise pressure unit | Reasoned/source-informed default; no fitted table |

**Results.**

| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |
|---|---:|---:|---:|---:|---:|---:|
| aggressor | 0 | 64 × 250 | +52.18 | 6.32 | +50.9 | +52.5 |
| alan_wake_claude | 0 | 64 × 250 | +140.05 | 7.66 | +123.0 | +144.3 |
| bassander_lin_glm | 0 | 64 × 250 | +140.72 | 12.55 | +148.0 | +138.9 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +70.02 | 5.10 | +71.5 | +69.6 |
| invoker_gpt | 0 | 64 × 250 | +37.10 | 8.46 | +31.1 | +38.6 |
| jack_joyce_gemini | 0 | 64 × 250 | +170.37 | 14.30 | +181.5 | +167.6 |
| jesse_faden_glm | 0 | 64 × 250 | +181.26 | 12.42 | +126.9 | +194.9 |
| max_payne_gpt | 0 | 64 × 250 | +46.58 | 3.86 | +50.0 | +45.7 |
| meepo_claude | 0 | 64 × 250 | +52.42 | 8.90 | +58.1 | +51.0 |
| random | 0 | 64 × 250 | +211.57 | 13.89 | +204.9 | +213.2 |
| valka_onderra_claude | 0 | 64 × 250 | +92.20 | 10.15 | +82.9 | +94.5 |
| windrunner_glm | 0 | 64 × 250 | +51.51 | 5.60 | +27.7 | +57.5 |
| v3:Bot | 2 | 128 × 250 | +13.75 | 6.41 | +16.7 | +13.0 |
| v1:Bot | 3 | 128 × 250 | +33.31 | 4.81 | +29.8 | +34.2 |
| v2:Bot | 3 | 128 × 250 | +14.04 | 6.00 | +10.9 | +14.8 |
| v4:Bot | 3 | 128 × 250 | +36.26 | 7.12 | +9.5 | +43.0 |
| v5:Bot | 3 | 128 × 250 | -2.92 | 6.38 | -3.7 | -2.7 |
| v8:Bot | 4 | 128 × 250 | +31.51 | 4.45 | +15.1 | +35.6 |
| aggressor | 4 | 128 × 250 | +48.40 | 4.56 | +37.1 | +51.2 |
| aggressor | 1000 | 128 × 250 | +41.53 | 4.80 | +42.2 | +41.4 |
| alan_wake_claude | 1000 | 128 × 250 | +148.88 | 7.17 | +139.7 | +151.2 |
| bassander_lin_glm | 1000 | 128 × 250 | +154.69 | 8.22 | +171.6 | +150.5 |
| hadrian_marlowe_gpt | 1000 | 128 × 250 | +65.94 | 4.31 | +69.0 | +65.2 |
| invoker_gpt | 1000 | 128 × 250 | +38.52 | 6.06 | +56.5 | +34.0 |
| jack_joyce_gemini | 1000 | 128 × 250 | +198.46 | 10.16 | +212.5 | +194.9 |
| jesse_faden_glm | 1000 | 128 × 250 | +206.54 | 9.51 | +207.9 | +206.2 |
| max_payne_gpt | 1000 | 128 × 250 | +50.00 | 2.83 | +54.8 | +48.8 |
| meepo_claude | 1000 | 128 × 250 | +44.75 | 7.37 | +41.8 | +45.5 |
| random | 1000 | 128 × 250 | +217.05 | 9.25 | +227.1 | +214.5 |
| valka_onderra_claude | 1000 | 128 × 250 | +86.72 | 6.87 | +61.1 | +93.1 |
| windrunner_glm | 1000 | 128 × 250 | +54.69 | 4.19 | +43.5 | +57.5 |

Weighted field scores: seed 0: **+78.34** (64 × 250 per opponent); seed 1000: **+79.64** (128 × 250 per opponent).

**Verdict.** Deployed after the held-out comparison; see final choice below.

## v7

**What it does.** Uses v6’s separate betting and raising estimates with a 90%-pot postflop size. The tracking rules and action frequencies are unchanged.

**Why it exists.** Retest the larger-size idea after repairing the interpretation of re-raises.

**Decision line count:** 83. Imports, comments, blank lines, constructor, initialization and metadata are excluded.

**Parameters.** All probabilities are dimensionless. Constants for card/street indices, array layout, clipping to [0, 1], and floating-point numerical protection are structural rather than fitted parameters.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | Inherited from v6 |
| `open_size_bb` | 2.5 | big blinds, raise-to | Inherited from v6 |
| `reraise_multiple` | 3.0 | multiple of the current preflop bet | Inherited from v6 |
| `threebet_equity` | 0.63 | equity probability | Inherited from v6 |
| `light_threebet_frequency` | 0.15 | mixing probability | Inherited from v6 |
| `isolate_equity` | 0.55 | equity probability | Inherited from v6 |
| `value_equity` | 0.62 | equity probability | Inherited from v6 |
| `raise_equity` | 0.85 | adjusted equity probability | Inherited from v6 |
| `bluff_frequency` | 0.35 | mixing probability with air | Inherited from v6 |
| `semibluff_frequency` | 0.5 | mixing probability with medium equity | Inherited from v6 |
| `bluff_split_equity` | 0.45 | equity probability | Inherited from v6 |
| `bluff_street_decay` | 1.0 | frequency multiplier per later street | Inherited from v6 |
| `bet_pot_fraction` | 0.9 | fraction of pot after calling | One-variable development experiment; v6 used 0.65 |
| `range_retention` | 0.58 | range fraction retained per ordinary pressure unit | Inherited from v6 |
| `prior_street_weight` | 0.5 | current-street bet equivalents | Inherited from v6 |
| `realization_fraction` | 0.9 | fraction of equity realized before river | Inherited from v6 |
| `call_margin_equity` | 0.02 | equity probability above pot odds | Inherited from v6 |
| `prior_events` | 20.0 | pseudo-opportunities per estimated rate | Inherited from v6 |
| `prior_bet_rate` | 0.4 | probability of betting when checking is free | Inherited from v6 |
| `prior_fold_rate` | 0.5 | probability of folding to our aggression | Inherited from v6 |
| `betting_adjustment` | 0.65 | retention change per unit of observed frequency deviation | Inherited from v6 |
| `bluff_adjustment` | 1.25 | bluff probability change per unit of fold-rate deviation | Inherited from v6 |
| `prior_raise_rate` | 0.15 | probability of raising when facing our aggression | Inherited from v6 |
| `reraise_range_retention` | 0.4 | range fraction retained per re-raise pressure unit | Inherited from v6 |

**Results.**

| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |
|---|---:|---:|---:|---:|---:|---:|
| aggressor | 0 | 64 × 250 | +18.51 | 8.07 | +35.9 | +14.2 |
| alan_wake_claude | 0 | 64 × 250 | +198.62 | 13.02 | +206.5 | +196.6 |
| bassander_lin_glm | 0 | 64 × 250 | +163.81 | 13.11 | +88.0 | +182.8 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +58.37 | 7.22 | +73.2 | +54.7 |
| invoker_gpt | 0 | 64 × 250 | +21.06 | 11.13 | +5.0 | +25.1 |
| jack_joyce_gemini | 0 | 64 × 250 | +250.32 | 15.98 | +250.1 | +250.4 |
| jesse_faden_glm | 0 | 64 × 250 | +241.89 | 14.32 | +230.8 | +244.7 |
| max_payne_gpt | 0 | 64 × 250 | +56.94 | 5.08 | +58.0 | +56.7 |
| meepo_claude | 0 | 64 × 250 | +42.22 | 10.76 | +65.1 | +36.5 |
| random | 0 | 64 × 250 | +293.17 | 16.03 | +260.5 | +301.3 |
| valka_onderra_claude | 0 | 64 × 250 | +107.87 | 13.17 | +91.2 | +112.0 |
| windrunner_glm | 0 | 64 × 250 | +51.80 | 6.75 | +25.1 | +58.5 |
| v6:Bot | 2 | 128 × 250 | -11.90 | 6.85 | +0.5 | -15.0 |
| v1:Bot | 3 | 64 × 250 | +30.16 | 7.21 | +14.3 | +34.1 |
| v2:Bot | 3 | 64 × 250 | -7.28 | 11.02 | -8.8 | -6.9 |
| v3:Bot | 3 | 64 × 250 | -3.47 | 10.72 | -4.6 | -3.2 |
| v4:Bot | 3 | 64 × 250 | +24.66 | 10.85 | -9.9 | +33.3 |
| v5:Bot | 3 | 64 × 250 | -19.25 | 10.74 | -7.8 | -22.1 |
| v9:Bot | 4 | 128 × 250 | +27.50 | 5.98 | +22.6 | +28.7 |
| v6:Bot | 5 | 256 × 250 | -16.24 | 4.69 | -6.5 | -18.7 |
| aggressor | 1000 | 128 × 250 | +14.30 | 6.38 | +23.8 | +11.9 |
| alan_wake_claude | 1000 | 128 × 250 | +185.09 | 10.54 | +164.5 | +190.2 |
| bassander_lin_glm | 1000 | 128 × 250 | +157.79 | 9.67 | +194.2 | +148.7 |
| hadrian_marlowe_gpt | 1000 | 128 × 250 | +52.80 | 5.11 | +62.2 | +50.4 |
| invoker_gpt | 1000 | 128 × 250 | +21.76 | 8.07 | +54.6 | +13.5 |
| jack_joyce_gemini | 1000 | 128 × 250 | +258.41 | 11.41 | +258.5 | +258.4 |
| jesse_faden_glm | 1000 | 128 × 250 | +264.21 | 10.72 | +258.8 | +265.6 |
| max_payne_gpt | 1000 | 128 × 250 | +55.94 | 3.49 | +63.5 | +54.0 |
| meepo_claude | 1000 | 128 × 250 | +50.96 | 8.19 | +24.7 | +57.5 |
| random | 1000 | 128 × 250 | +301.15 | 11.01 | +312.1 | +298.4 |
| valka_onderra_claude | 1000 | 128 × 250 | +97.34 | 9.27 | +65.5 | +105.3 |
| windrunner_glm | 1000 | 128 × 250 | +46.93 | 3.72 | +41.2 | +48.4 |

Weighted field scores: seed 0: **+83.54** (64 × 250 per opponent); seed 1000: **+81.70** (128 × 250 per opponent).

**Verdict.** Discarded for deployment despite its development field lead: it lost to v6 on both tested seeds, approximately -14.79 bb/100 with pooled SE 3.87 over 96,000 hands.

## v8

**What it does.** A stateless control for v6: replaces learned rates with their exact prior values. All action logic, sampling, sizes and thresholds are retained.

**Why it exists.** Isolate the effect of learning in the finalist, without changing its underlying poker strategy.

**Decision line count:** 66. Imports, comments, blank lines, constructor, initialization and metadata are excluded.

**Parameters.** All probabilities are dimensionless. Constants for card/street indices, array layout, clipping to [0, 1], and floating-point numerical protection are structural rather than fitted parameters.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | Inherited from v6 |
| `open_size_bb` | 2.5 | big blinds, raise-to | Inherited from v6 |
| `reraise_multiple` | 3.0 | multiple of the current preflop bet | Inherited from v6 |
| `threebet_equity` | 0.63 | equity probability | Inherited from v6 |
| `light_threebet_frequency` | 0.15 | mixing probability | Inherited from v6 |
| `isolate_equity` | 0.55 | equity probability | Inherited from v6 |
| `value_equity` | 0.62 | equity probability | Inherited from v6 |
| `raise_equity` | 0.85 | adjusted equity probability | Inherited from v6 |
| `bluff_frequency` | 0.35 | mixing probability with air | Inherited from v6 |
| `semibluff_frequency` | 0.5 | mixing probability with medium equity | Inherited from v6 |
| `bluff_split_equity` | 0.45 | equity probability | Inherited from v6 |
| `bluff_street_decay` | 1.0 | frequency multiplier per later street | Inherited from v6 |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | Inherited from v6 |
| `range_retention` | 0.58 | range fraction retained per ordinary pressure unit | Inherited from v6 |
| `prior_street_weight` | 0.5 | current-street bet equivalents | Inherited from v6 |
| `realization_fraction` | 0.9 | fraction of equity realized before river | Inherited from v6 |
| `call_margin_equity` | 0.02 | equity probability above pot odds | Inherited from v6 |
| `prior_events` | 20.0 | pseudo-opportunities per estimated rate | Inherited from v6 |
| `prior_bet_rate` | 0.4 | probability of betting when checking is free | Inherited from v6 |
| `prior_fold_rate` | 0.5 | probability of folding to our aggression | Inherited from v6 |
| `betting_adjustment` | 0.65 | retention change per unit of observed frequency deviation | Inherited from v6 |
| `bluff_adjustment` | 1.25 | bluff probability change per unit of fold-rate deviation | Inherited from v6 |
| `prior_raise_rate` | 0.15 | probability of raising when facing our aggression | Inherited from v6 |
| `reraise_range_retention` | 0.4 | range fraction retained per re-raise pressure unit | Inherited from v6 |

**Results.**

| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |
|---|---:|---:|---:|---:|---:|---:|
| aggressor | 0 | 64 × 250 | +43.57 | 5.00 | +64.2 | +38.4 |
| alan_wake_claude | 0 | 64 × 250 | +117.51 | 10.58 | +99.7 | +122.0 |
| bassander_lin_glm | 0 | 64 × 250 | +170.56 | 11.88 | +177.4 | +168.8 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +70.68 | 5.47 | +74.3 | +69.8 |
| invoker_gpt | 0 | 64 × 250 | +37.13 | 10.48 | +53.9 | +32.9 |
| jack_joyce_gemini | 0 | 64 × 250 | +210.28 | 15.34 | +184.9 | +216.6 |
| jesse_faden_glm | 0 | 64 × 250 | +227.59 | 13.82 | +210.4 | +231.9 |
| max_payne_gpt | 0 | 64 × 250 | +48.83 | 3.50 | +49.4 | +48.7 |
| meepo_claude | 0 | 64 × 250 | +43.16 | 11.07 | +72.9 | +35.7 |
| random | 0 | 64 × 250 | +188.87 | 13.10 | +176.7 | +191.9 |
| valka_onderra_claude | 0 | 64 × 250 | +75.91 | 11.64 | +72.4 | +76.8 |
| windrunner_glm | 0 | 64 × 250 | +23.19 | 5.82 | +21.9 | +23.5 |

Weighted field scores: seed 0: **+73.91** (64 × 250 per opponent).

**Verdict.** Retained as a frozen no-learning control. Its adaptive parent v6 won the direct ablation match; see that parent’s results. Not selected for deployment.

## v9

**What it does.** A stateless control for v7: replaces learned rates with their exact prior values. All action logic, sampling, sizes and thresholds are retained.

**Why it exists.** Isolate the effect of learning in the finalist, without changing its underlying poker strategy.

**Decision line count:** 66. Imports, comments, blank lines, constructor, initialization and metadata are excluded.

**Parameters.** All probabilities are dimensionless. Constants for card/street indices, array layout, clipping to [0, 1], and floating-point numerical protection are structural rather than fitted parameters.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | Inherited from v7 |
| `open_size_bb` | 2.5 | big blinds, raise-to | Inherited from v7 |
| `reraise_multiple` | 3.0 | multiple of the current preflop bet | Inherited from v7 |
| `threebet_equity` | 0.63 | equity probability | Inherited from v7 |
| `light_threebet_frequency` | 0.15 | mixing probability | Inherited from v7 |
| `isolate_equity` | 0.55 | equity probability | Inherited from v7 |
| `value_equity` | 0.62 | equity probability | Inherited from v7 |
| `raise_equity` | 0.85 | adjusted equity probability | Inherited from v7 |
| `bluff_frequency` | 0.35 | mixing probability with air | Inherited from v7 |
| `semibluff_frequency` | 0.5 | mixing probability with medium equity | Inherited from v7 |
| `bluff_split_equity` | 0.45 | equity probability | Inherited from v7 |
| `bluff_street_decay` | 1.0 | frequency multiplier per later street | Inherited from v7 |
| `bet_pot_fraction` | 0.9 | fraction of pot after calling | Inherited from v7 |
| `range_retention` | 0.58 | range fraction retained per ordinary pressure unit | Inherited from v7 |
| `prior_street_weight` | 0.5 | current-street bet equivalents | Inherited from v7 |
| `realization_fraction` | 0.9 | fraction of equity realized before river | Inherited from v7 |
| `call_margin_equity` | 0.02 | equity probability above pot odds | Inherited from v7 |
| `prior_events` | 20.0 | pseudo-opportunities per estimated rate | Inherited from v7 |
| `prior_bet_rate` | 0.4 | probability of betting when checking is free | Inherited from v7 |
| `prior_fold_rate` | 0.5 | probability of folding to our aggression | Inherited from v7 |
| `betting_adjustment` | 0.65 | retention change per unit of observed frequency deviation | Inherited from v7 |
| `bluff_adjustment` | 1.25 | bluff probability change per unit of fold-rate deviation | Inherited from v7 |
| `prior_raise_rate` | 0.15 | probability of raising when facing our aggression | Inherited from v7 |
| `reraise_range_retention` | 0.4 | range fraction retained per re-raise pressure unit | Inherited from v7 |

**Results.**

| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |
|---|---:|---:|---:|---:|---:|---:|
| aggressor | 0 | 64 × 250 | +24.13 | 6.51 | +55.3 | +16.3 |
| alan_wake_claude | 0 | 64 × 250 | +144.73 | 13.68 | +122.7 | +150.2 |
| bassander_lin_glm | 0 | 64 × 250 | +164.32 | 12.75 | +175.5 | +161.5 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +56.65 | 6.91 | +75.4 | +52.0 |
| invoker_gpt | 0 | 64 × 250 | +37.73 | 12.98 | +49.2 | +34.9 |
| jack_joyce_gemini | 0 | 64 × 250 | +266.60 | 19.10 | +265.3 | +266.9 |
| jesse_faden_glm | 0 | 64 × 250 | +297.01 | 14.53 | +309.2 | +294.0 |
| max_payne_gpt | 0 | 64 × 250 | +49.79 | 4.22 | +47.7 | +50.3 |
| meepo_claude | 0 | 64 × 250 | +40.84 | 11.14 | +69.1 | +33.8 |
| random | 0 | 64 × 250 | +284.23 | 16.15 | +229.1 | +298.0 |
| valka_onderra_claude | 0 | 64 × 250 | +121.13 | 15.21 | +103.8 | +125.5 |
| windrunner_glm | 0 | 64 × 250 | +17.80 | 5.89 | +26.2 | +15.7 |

Weighted field scores: seed 0: **+80.80** (64 × 250 per opponent).

**Verdict.** Retained as a frozen no-learning control. Its adaptive parent v7 won the direct ablation match; see that parent’s results. Not selected for deployment.

## v10

**What it does.** Adds occasional out-of-position checks with value hands before the river. The check frequency rises with the opponent’s observed free-betting rate, inviting bluffs while preserving the rest of v6’s strategy.

**Why it exists.** Test whether inducing bets earns more than immediately betting value into frequent bettors.

**Decision line count:** 85. Imports, comments, blank lines, constructor, initialization and metadata are excluded.

**Parameters.** All probabilities are dimensionless. Constants for card/street indices, array layout, clipping to [0, 1], and floating-point numerical protection are structural rather than fitted parameters.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | Inherited from v6 |
| `open_size_bb` | 2.5 | big blinds, raise-to | Inherited from v6 |
| `reraise_multiple` | 3.0 | multiple of the current preflop bet | Inherited from v6 |
| `threebet_equity` | 0.63 | equity probability | Inherited from v6 |
| `light_threebet_frequency` | 0.15 | mixing probability | Inherited from v6 |
| `isolate_equity` | 0.55 | equity probability | Inherited from v6 |
| `value_equity` | 0.62 | equity probability | Inherited from v6 |
| `raise_equity` | 0.85 | adjusted equity probability | Inherited from v6 |
| `bluff_frequency` | 0.35 | mixing probability with air | Inherited from v6 |
| `semibluff_frequency` | 0.5 | mixing probability with medium equity | Inherited from v6 |
| `bluff_split_equity` | 0.45 | equity probability | Inherited from v6 |
| `bluff_street_decay` | 1.0 | frequency multiplier per later street | Inherited from v6 |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | Inherited from v6 |
| `range_retention` | 0.58 | range fraction retained per ordinary pressure unit | Inherited from v6 |
| `prior_street_weight` | 0.5 | current-street bet equivalents | Inherited from v6 |
| `realization_fraction` | 0.9 | fraction of equity realized before river | Inherited from v6 |
| `call_margin_equity` | 0.02 | equity probability above pot odds | Inherited from v6 |
| `prior_events` | 20.0 | pseudo-opportunities per estimated rate | Inherited from v6 |
| `prior_bet_rate` | 0.4 | probability of betting when checking is free | Inherited from v6 |
| `prior_fold_rate` | 0.5 | probability of folding to our aggression | Inherited from v6 |
| `betting_adjustment` | 0.65 | retention change per unit of observed frequency deviation | Inherited from v6 |
| `bluff_adjustment` | 1.25 | bluff probability change per unit of fold-rate deviation | Inherited from v6 |
| `prior_raise_rate` | 0.15 | probability of raising when facing our aggression | Inherited from v6 |
| `reraise_range_retention` | 0.4 | range fraction retained per re-raise pressure unit | Inherited from v6 |
| `trap_frequency` | 0.65 | maximum value-check probability, scaled by free-betting rate | Reasoned/source-informed default; no fitted table |

**Results.**

| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |
|---|---:|---:|---:|---:|---:|---:|
| aggressor | 0 | 64 × 250 | +72.45 | 7.10 | +49.0 | +78.3 |
| alan_wake_claude | 0 | 64 × 250 | +136.12 | 7.88 | +131.2 | +137.3 |
| bassander_lin_glm | 0 | 64 × 250 | +132.62 | 9.74 | +109.9 | +138.3 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +72.70 | 5.79 | +82.1 | +70.3 |
| invoker_gpt | 0 | 64 × 250 | +37.94 | 7.29 | +32.8 | +39.2 |
| jack_joyce_gemini | 0 | 64 × 250 | +165.37 | 14.74 | +174.7 | +163.1 |
| jesse_faden_glm | 0 | 64 × 250 | +172.82 | 12.42 | +142.1 | +180.5 |
| max_payne_gpt | 0 | 64 × 250 | +45.43 | 4.14 | +46.8 | +45.1 |
| meepo_claude | 0 | 64 × 250 | +45.35 | 9.50 | +63.1 | +40.9 |
| random | 0 | 64 × 250 | +189.26 | 11.99 | +169.7 | +194.1 |
| valka_onderra_claude | 0 | 64 × 250 | +70.31 | 9.09 | +75.2 | +69.1 |
| windrunner_glm | 0 | 64 × 250 | +53.07 | 6.59 | +51.6 | +53.4 |
| v1:Bot | 3 | 64 × 250 | +32.49 | 7.11 | +26.6 | +34.0 |
| v2:Bot | 3 | 64 × 250 | +19.80 | 9.55 | +34.6 | +16.1 |
| v3:Bot | 3 | 64 × 250 | +37.18 | 8.43 | +23.5 | +40.6 |
| v4:Bot | 3 | 64 × 250 | +41.14 | 11.84 | +35.5 | +42.6 |
| v5:Bot | 3 | 64 × 250 | +20.87 | 8.02 | +26.8 | +19.4 |
| v7:Bot | 3 | 64 × 250 | +10.00 | 9.89 | +32.9 | +4.3 |
| v8:Bot | 3 | 64 × 250 | +34.35 | 5.63 | +22.7 | +37.3 |
| v9:Bot | 3 | 64 × 250 | +47.67 | 7.87 | +32.3 | +51.5 |
| v11:Bot | 4 | 128 × 250 | +25.52 | 4.46 | +8.4 | +29.8 |
| v6:Bot | 4 | 128 × 250 | +6.38 | 5.08 | -0.2 | +8.0 |
| aggressor | 4 | 128 × 250 | +58.70 | 5.23 | +35.5 | +64.5 |
| v6:Bot | 5 | 128 × 250 | +5.78 | 4.89 | +7.3 | +5.4 |
| aggressor | 1000 | 128 × 250 | +60.26 | 5.21 | +52.9 | +62.1 |
| alan_wake_claude | 1000 | 128 × 250 | +136.57 | 6.69 | +136.6 | +136.6 |
| bassander_lin_glm | 1000 | 128 × 250 | +134.05 | 7.80 | +159.3 | +127.7 |
| hadrian_marlowe_gpt | 1000 | 128 × 250 | +65.26 | 4.50 | +69.7 | +64.1 |
| invoker_gpt | 1000 | 128 × 250 | +36.00 | 6.12 | +51.7 | +32.1 |
| jack_joyce_gemini | 1000 | 128 × 250 | +184.36 | 9.73 | +191.6 | +182.5 |
| jesse_faden_glm | 1000 | 128 × 250 | +177.58 | 9.36 | +190.3 | +174.4 |
| max_payne_gpt | 1000 | 128 × 250 | +51.41 | 2.96 | +53.1 | +51.0 |
| meepo_claude | 1000 | 128 × 250 | +46.73 | 6.78 | +53.5 | +45.0 |
| random | 1000 | 128 × 250 | +191.09 | 8.93 | +186.2 | +192.3 |
| valka_onderra_claude | 1000 | 128 × 250 | +81.97 | 6.77 | +58.8 | +87.8 |
| windrunner_glm | 1000 | 128 × 250 | +57.80 | 4.90 | +43.7 | +61.3 |

Weighted field scores: seed 0: **+76.00** (64 × 250 per opponent); seed 1000: **+77.01** (128 × 250 per opponent).

**Verdict.** Not deployed: the value-checking idea did not earn the final choice on held-out weighted score; its uncertain parent advantage is reported rather than treated as decisive.

## v11

**What it does.** A stateless control for v10: replaces learned rates with their exact prior values. All action logic, sampling, sizes and thresholds are retained.

**Why it exists.** Isolate the effect of learning in the finalist, without changing its underlying poker strategy.

**Decision line count:** 68. Imports, comments, blank lines, constructor, initialization and metadata are excluded.

**Parameters.** All probabilities are dimensionless. Constants for card/street indices, array layout, clipping to [0, 1], and floating-point numerical protection are structural rather than fitted parameters.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo deals per decision | Inherited from v10 |
| `open_size_bb` | 2.5 | big blinds, raise-to | Inherited from v10 |
| `reraise_multiple` | 3.0 | multiple of the current preflop bet | Inherited from v10 |
| `threebet_equity` | 0.63 | equity probability | Inherited from v10 |
| `light_threebet_frequency` | 0.15 | mixing probability | Inherited from v10 |
| `isolate_equity` | 0.55 | equity probability | Inherited from v10 |
| `value_equity` | 0.62 | equity probability | Inherited from v10 |
| `raise_equity` | 0.85 | adjusted equity probability | Inherited from v10 |
| `bluff_frequency` | 0.35 | mixing probability with air | Inherited from v10 |
| `semibluff_frequency` | 0.5 | mixing probability with medium equity | Inherited from v10 |
| `bluff_split_equity` | 0.45 | equity probability | Inherited from v10 |
| `bluff_street_decay` | 1.0 | frequency multiplier per later street | Inherited from v10 |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | Inherited from v10 |
| `range_retention` | 0.58 | range fraction retained per ordinary pressure unit | Inherited from v10 |
| `prior_street_weight` | 0.5 | current-street bet equivalents | Inherited from v10 |
| `realization_fraction` | 0.9 | fraction of equity realized before river | Inherited from v10 |
| `call_margin_equity` | 0.02 | equity probability above pot odds | Inherited from v10 |
| `prior_events` | 20.0 | pseudo-opportunities per estimated rate | Inherited from v10 |
| `prior_bet_rate` | 0.4 | probability of betting when checking is free | Inherited from v10 |
| `prior_fold_rate` | 0.5 | probability of folding to our aggression | Inherited from v10 |
| `betting_adjustment` | 0.65 | retention change per unit of observed frequency deviation | Inherited from v10 |
| `bluff_adjustment` | 1.25 | bluff probability change per unit of fold-rate deviation | Inherited from v10 |
| `prior_raise_rate` | 0.15 | probability of raising when facing our aggression | Inherited from v10 |
| `reraise_range_retention` | 0.4 | range fraction retained per re-raise pressure unit | Inherited from v10 |
| `trap_frequency` | 0.65 | maximum value-check probability, scaled by free-betting rate | Inherited from v10 |

**Results.**

| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |
|---|---:|---:|---:|---:|---:|---:|
| aggressor | 0 | 64 × 250 | +47.73 | 5.32 | +55.6 | +45.8 |
| alan_wake_claude | 0 | 64 × 250 | +105.02 | 10.82 | +102.8 | +105.6 |
| bassander_lin_glm | 0 | 64 × 250 | +152.46 | 11.94 | +175.3 | +146.8 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +61.50 | 6.15 | +82.4 | +56.3 |
| invoker_gpt | 0 | 64 × 250 | +22.60 | 9.10 | +46.4 | +16.7 |
| jack_joyce_gemini | 0 | 64 × 250 | +191.32 | 13.09 | +176.2 | +195.1 |
| jesse_faden_glm | 0 | 64 × 250 | +212.64 | 11.90 | +198.5 | +216.2 |
| max_payne_gpt | 0 | 64 × 250 | +48.71 | 3.27 | +50.1 | +48.4 |
| meepo_claude | 0 | 64 × 250 | +42.73 | 10.41 | +75.9 | +34.4 |
| random | 0 | 64 × 250 | +162.18 | 12.76 | +164.7 | +161.6 |
| valka_onderra_claude | 0 | 64 × 250 | +57.10 | 10.57 | +43.6 | +60.5 |
| windrunner_glm | 0 | 64 × 250 | +27.80 | 5.81 | +24.2 | +28.7 |

Weighted field scores: seed 0: **+66.30** (64 × 250 per opponent).

**Verdict.** Retained as a frozen no-learning control. Its adaptive parent v10 won the direct ablation match; see that parent’s results. Not selected for deployment.

## Opponent tracking

Only named float32 summary counts persist across hands. No action logs, identities, random keys, hidden cards or external memory are stored. `observe` counts valid postflop opponent actions. An action immediately after our bet/raise on the same street is a facing-bet opportunity; all other voluntary postflop actions are free-check opportunities. Fold and raise counts use these denominators. Facing-bet matching requires both the current and previous history entries to be valid.

The deployed tracker starts with 20 pseudo-opportunities: 8 free bets (40%), 10 folds facing aggression (50%), and 3 raises facing aggression (15%). The raise and fold estimates share the same facing-bet denominator. Priors shrink early estimates toward a sound opening strategy.

- Ordinary-bet range retention = `clip(0.58 + 0.65 × (free_bet_rate − 0.40), epsilon, 1)`.
- Re-raise range retention = `clip(0.40 + 0.65 × (raise_rate − 0.15), epsilon, 1)`.
- Each current-street event contributes one pressure unit; prior-street events contribute 0.5. Their retained fractions multiply. Preflop keeps the fixed 0.58 model.
- Calling uses `max(raw_equity + remaining_range − 1, 0) / remaining_range`, pot odds, a 0.02 equity margin, and 90% realization when future betting is possible. Value raises use 0.85 adjusted equity.
- Air bluffs start at 35%; medium-equity bluffs at 50%. Add `1.25 × (fold_rate − 0.50)` and clip to [0, 1]. A ten-percentage-point increase in folding therefore adds 12.5 percentage points of bluff probability. No automatic later-street decay remains.

For example, a 70% free-betting rate gives 0.775 retention for ordinary bets, while a 15% re-raising rate still gives 0.40 retention for re-raises. Thus frequent betting does not automatically justify loose calls against re-raises. These are interpretable range heuristics, not exact conditional-equity estimates.

**Exact ablation:** `v8.py` is the deployed strategy with rates fixed to the same priors and no persistent state. Its head-to-head results against v6 and field comparison quantify learning separately from the underlying strategy.

| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |
|---|---:|---:|---:|---:|---:|---:|
| v8:Bot | 4 | 128 × 250 | +31.51 | 4.45 | +15.1 | +35.6 |

v6 weighted scores: seed 0: +78.34; seed 1000: +79.64.

v8 weighted scores: seed 0: +73.91.

## Final choice and held-out results

**Deploy v6.** Held-out weighted score: **+79.64**, seed 1000, 128 × 250 hands against each of twelve opponents.

Finalist held-out scores: v6: +79.64; v7: +81.70; v10: +77.01.

| Opponent | Seed | Tables × hands | bb/100 | SE | First 50 | Remaining 200 |
|---|---:|---:|---:|---:|---:|---:|
| windrunner_glm | 1000 | 128 × 250 | +54.69 | 4.19 | +43.5 | +57.5 |
| invoker_gpt | 1000 | 128 × 250 | +38.52 | 6.06 | +56.5 | +34.0 |
| meepo_claude | 1000 | 128 × 250 | +44.75 | 7.37 | +41.8 | +45.5 |
| hadrian_marlowe_gpt | 1000 | 128 × 250 | +65.94 | 4.31 | +69.0 | +65.2 |
| valka_onderra_claude | 1000 | 128 × 250 | +86.72 | 6.87 | +61.1 | +93.1 |
| aggressor | 1000 | 128 × 250 | +41.53 | 4.80 | +42.2 | +41.4 |
| bassander_lin_glm | 1000 | 128 × 250 | +154.69 | 8.22 | +171.6 | +150.5 |
| alan_wake_claude | 1000 | 128 × 250 | +148.88 | 7.17 | +139.7 | +151.2 |
| max_payne_gpt | 1000 | 128 × 250 | +50.00 | 2.83 | +54.8 | +48.8 |
| jesse_faden_glm | 1000 | 128 × 250 | +206.54 | 9.51 | +207.9 | +206.2 |
| jack_joyce_gemini | 1000 | 128 × 250 | +198.46 | 10.16 | +212.5 | +194.9 |
| random | 1000 | 128 × 250 | +217.05 | 9.25 | +227.1 | +214.5 |

V6 is deployed as the most defensible final choice: its held-out weighted score is +79.64, it won every field matchup, and it beat its strategic parent v3 by +13.75 bb/100 (SE 6.41) over 32,000 hands. Its exact no-learning control v8 lost to it by 31.51 bb/100 (SE 4.45), with the adaptive advantage increasing from +15.1 during the first 50 hands to +35.6 afterward. On development seed 0, learning increased the weighted field score from v8's +73.91 to v6's +78.34; that field gain is modest and should not be read as a precise estimate of future benefit.

V7 has the highest observed held-out field score, +81.70, but is not deployed. Its 2.06-point lead over v6 is small compared with the noisy individual matchups, whereas its direct weakness against its own adaptive parent was replicated: -11.90 bb/100 (SE 6.85) on 128 tables with seed 2 and -16.24 (SE 4.69) on 256 tables with seed 5. Pooling the independent table samples gives approximately -14.79 bb/100 (SE 3.87) over 96,000 hands, using the rounded printed statistics. That reproducible vulnerability matters for the unseen adaptive phase-4 field, so this selection deliberately sacrifices a small measured benchmark lead for stronger evidence of robustness. It does not claim that v6 maximizes the measured fixed-field score.

V10's value-checking idea scored +77.01 on the held-out field. Its two parent comparisons against v6 pooled to about +6.08 bb/100 (SE 3.52) over 64,000 hands, an uncertain advantage that did not outweigh its lower field score and added parameter. V4 and v5 were rejected earlier for a parent-loss warning or a lack of demonstrated improvement. The three stateless controls were retained for attribution; each lost convincingly to its adaptive counterpart. All finalist code was frozen before any seed-1000 results were examined, and each finalist ran the held-out field exactly once.

The final choice also considers direct parent performance and clarity. The highest development score alone is insufficient evidence when an experiment loses to its own parent. All version files were frozen before benchmarking; no held-out result was used to change a strategy.

### Known weaknesses

Equity is sampled against random cards and transformed using a coarse range-strength model. It does not reconstruct opponent card combinations, account fully for prior calls, or model different bet sizes. Monte Carlo noise can change borderline decisions. The bot opens every button and learns only postflop behavior, leaving preflop adaptation limited.

Rates pool all postflop streets and are conditional on opportunities created by our own strategy. Twenty-event priors delay learning rare re-raises. An adapting opponent can establish a misleading early pattern, then change behavior; cumulative counts have no forgetting and may react slowly. Aggressive bluff increases can be exploited by an opponent who begins calling or check-raising more. Tests against our own adaptive versions help, but do not establish strength against the unseen phase-4 finals.

### Validation, provenance and files

`audit.py` checks allowed bot imports, one Bot class per version, constructor/metadata consistency, decision-line and sample budgets, immutable version hashes, exact JIT tracking counts, float32 state shape/dtype preservation, and persistence across hands. Full simulator benchmarks exercise action tracing and play. `protected_files.sha256` records the supplied helpers and frozen opponents; final verification checks that they remain unchanged.

Ideas borrowed from the supplied frozen windrunner_glm include wide heads-up opening, pressure counts, light three-bets and mixed postflop aggression. The supplied invoker_gpt informed range truncation and excluding unmatched all-in chips from pot odds. The frozen meepo_claude strategy was also read for comparison. No identity recognition, fingerprints or opponent-specific branches are used.

Scope disclosure: before reading the original pasted directory restriction, an initial repository-wide filename listing was made. No source contents outside this bot directory were inspected. Subsequent source inspection and all created/modified project files were confined to this directory; the supplied benchmark used the authorized Python environment. No commits were made.

Scratch evidence is retained here: `smoke_*.log`, `field_*.log`, `heldout_*.log`, `parent_*.log`, `probe_*.log`, `audit.log`, `audit_results.json`, `protected_files.sha256`, `protected_files_check.log`, `session_notes.txt`, `audit.py`, and this report’s generator `build_readme.py`, `benchmark_results.json`, and `selection_note.txt`. These utilities are not imported by the bot. Every performance table is parsed from recorded benchmark output. Pooled parent statistics are explicitly derived from the reported means, errors and table counts. No benchmark results were invented.
