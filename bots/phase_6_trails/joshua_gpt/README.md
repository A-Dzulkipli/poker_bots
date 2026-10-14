# joshua_gpt — deployed v3

- **Lineage:** `phase_6_trails/joshua_gpt`
- **Model:** GPT-6 (Codex), as identified by this session; a more precise model identifier was not supplied.
- **Prompt:** `botgen-v6`
- **Engine:** `decafb9`, SHA-256 prefix `295477eff7637056`
- **Executed:** yes; existing repository environment, Python 3.13.12, JAX 0.11.2 CPU. No packages installed or environment changes.
- **Access:** execution and requested file writes used sandbox escalation because the assigned directory and existing uv cache are outside the desktop chat's default writable roots.

- **Wall-clock:** 55.74 minutes, 2026-10-14T07:28:02+00:00 to 2026-10-14T08:23:46.515708+00:00. Iteration cutoff: 08:18:02 UTC; deadline: 08:28:02 UTC.

- **Sandbox deviation:** before reading the full pasted task, an initial `rg --files` listing accidentally ranged across the repository and exposed filenames outside this directory, including opponent, engine, and earlier-phase filenames. No opponent source, engine source, earlier result contents, or git state were opened. Subsequent inspection and all authored files stayed inside this directory. The explicitly supplied task attachment was read to obtain the request.

## Environment check

The first attempt hit a desktop sandbox restriction on the existing uv cache. The retry with sandbox escalation succeeded:

```text
Initial sandbox attempt:
error: Failed to initialize cache at `/Users/alik/.cache/uv`
  Caused by: failed to open file `/Users/alik/.cache/uv/sdists-v9/.git`: Operation not permitted (os error 1)

Successful retry with existing environment and sandbox escalation:
python      /Users/alik/Documents/Projects/poker_bots/poker_bots/.venv/bin/python3 (3.13.12)
jax         0.11.2 (cpu)
poker_bots  /Users/alik/Documents/Projects/poker_bots/poker_bots/src/poker_bots
engine      sha256 295477eff7637056 (matches the contract)
smoke test  random vs aggressor, 40 hands in 2s: OK
```

## Design

The deployed v3 builds a weighted opponent range from the complete public action history, with preflop rank/pair/suited merit, historical made-hand percentiles, draw bonuses, and distinct limp/check/call/raise likelihoods. It uses range-weighted showdown equity to price calls and combines value betting with moderate opening and bluff frequencies. It was selected for the highest held-out weighted score among the three finalists; the later pressure and raise-size variants did not demonstrate a convincing improvement.

The bot uses at most 128 candidate holdings and legal board completions per decision, sharing samples between inference and showdown evaluation. It reconstructs the posterior from this hand's history each action, without persistent learning or opponent identification. Versions are independent and benchmarked sources were frozen.

## Benchmark protocol

Performance tables use 250 hands per table, balanced candidate seats, and table-clustered SE in bb/100. Separate 2 × 20 smoke tests establish execution and are excluded from performance tables. Development seeds are 0–9; held-out field evaluation uses seed 1000 once per finalist. Results smaller than roughly twice their SE are inconclusive. Small score differences are treated cautiously.

The first field run used 64 tables per opponent; later screening used 32 to preserve time for final checks. The v8 uniform control used 16 tables per opponent; held-out field and earlier-version checks use 32. Exact sizes appear below. Some runs overlapped, so timing includes resource contention.

The first field run's sum of printed opponent timings was 393 seconds.

## v1

**What it does:** Baseline importance-sampled range equity. Every opponent action updates each holding. Calls, value bets and sparse bluffs use the resulting equity.

**Why it exists:** Establish an executable, compact, stateless baseline.

**Decision-code lines:** 71; includes helper and act definitions, excludes imports, constructor, init_state, metadata, comments, blanks and shared helpers.

### Parameters

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `samples` | 128 | holdings/decision | 128 shared range and showdown samples; reasoning |
| `range_power` | 1.0 | dimensionless likelihood exponent | 1.0 baseline; 0.35 tests discounted evidence; 0.0 disables ranges for controls |
| `likelihood_floor` | 0.12 | probability | 0.12 contamination floor; reasoning |
| `likelihood_width` | 0.12 | percentile fraction | 0.12 smooth transition; reasoning |
| `open_percentile` | 0.38 | starting-hand percentile | 0.38 first preflop raise cutoff; reasoning |
| `reraise_percentile` | 0.82 | holding percentile | 0.82 second/later raise cutoff; reasoning |
| `call_percentile` | 0.45 | holding percentile | 0.45 call cutoff; reasoning |
| `bet_percentile` | 0.64 | made-hand percentile | 0.64 first postflop bet cutoff; reasoning |
| `check_discount` | 0.65 | probability fraction | 0.65 discounts strong hands after checks/limps; reasoning |
| `pair_bonus_ranks` | 13.0 | summed rank points | 13-point pocket-pair bonus; reasoning |
| `suited_bonus_ranks` | 3.0 | summed rank points | 3-point suited bonus; reasoning |
| `call_margin_equity` | 0.04 | equity fraction | 0.04 preflop margin, declines linearly to zero on river; reasoning |
| `open_equity` | 0.53 | equity fraction | 0.53 baseline; 0.49 in v3; 0.44 in v5, hypothesis-tested on seed 0 |
| `enter_equity` | 0.44 | equity fraction | 0.44 minimum unopened button entry and maximum bluff equity; reasoning |
| `value_equity` | 0.68 | equity fraction | 0.68 baseline free-action value threshold; 0.64 in betting experiment |
| `raise_equity` | 0.8 | equity fraction | 0.80 value threshold facing a wager; reasoning |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | 0.65 bet/raise increment; reasoning |
| `open_size_bb` | 2.5 | big blinds, raise-to | 2.5 opening/limp-isolation size; reasoning |
| `bluff_frequency` | 0.07 | probability per eligible free action | 0.07 baseline; 0.25 in v3; 0.50 in v5, hypothesis-tested on seed 0 |

### Results

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| kraken_gpt | 0 | 64 × 250 | -59.75 | 6.10 |
| leviathan_claude | 0 | 64 × 250 | -14.33 | 8.24 |
| megalodon_glm | 0 | 64 × 250 | -13.25 | 10.74 |
| spike_claude | 0 | 64 × 250 | -18.91 | 7.72 |
| faye_gpt | 0 | 64 × 250 | -65.07 | 5.60 |
| windrunner_glm | 0 | 64 × 250 | -15.41 | 9.33 |
| invoker_gpt | 0 | 64 × 250 | -0.42 | 7.82 |
| meepo_claude | 0 | 64 × 250 | +12.35 | 10.22 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +11.81 | 7.42 |
| vicious_glm | 0 | 64 × 250 | +73.80 | 12.94 |
| valka_onderra_claude | 0 | 64 × 250 | +68.92 | 10.33 |
| aggressor | 0 | 64 × 250 | -79.88 | 6.10 |
| alan_wake_claude | 0 | 64 × 250 | +95.17 | 9.41 |
| max_payne_gpt | 0 | 64 × 250 | +74.09 | 8.31 |
| bassander_lin_glm | 0 | 64 × 250 | +121.14 | 13.76 |
| jesse_faden_glm | 0 | 64 × 250 | +286.11 | 16.87 |
| jack_joyce_gemini | 0 | 64 × 250 | +276.09 | 12.57 |
| random | 0 | 64 × 250 | +320.96 | 13.28 |

Weighted score from `v1_field_s0.log`: **+17.80**.

**Verdict:** Kept as parent of v2. The executable baseline scored +17.80, but lost substantially to several stronger opponents and the aggressor probe.

## v2

**What it does:** Adds four-card straight/flush draws to flop and turn range quality. Button limps favor weaker holdings rather than using the calling-an-open model. Decision thresholds remain unchanged.

**Why it exists:** Test a more realistic passive-action and drawing-hand model.

**Decision-code lines:** 83; includes helper and act definitions, excludes imports, constructor, init_state, metadata, comments, blanks and shared helpers.

### Parameters

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `samples` | 128 | holdings/decision | 128 shared range and showdown samples; reasoning |
| `range_power` | 1.0 | dimensionless likelihood exponent | 1.0 baseline; 0.35 tests discounted evidence; 0.0 disables ranges for controls |
| `likelihood_floor` | 0.12 | probability | 0.12 contamination floor; reasoning |
| `likelihood_width` | 0.12 | percentile fraction | 0.12 smooth transition; reasoning |
| `open_percentile` | 0.38 | starting-hand percentile | 0.38 first preflop raise cutoff; reasoning |
| `reraise_percentile` | 0.82 | holding percentile | 0.82 second/later raise cutoff; reasoning |
| `call_percentile` | 0.45 | holding percentile | 0.45 call cutoff; reasoning |
| `bet_percentile` | 0.64 | made-hand percentile | 0.64 first postflop bet cutoff; reasoning |
| `check_discount` | 0.65 | probability fraction | 0.65 discounts strong hands after checks/limps; reasoning |
| `pair_bonus_ranks` | 13.0 | summed rank points | 13-point pocket-pair bonus; reasoning |
| `suited_bonus_ranks` | 3.0 | summed rank points | 3-point suited bonus; reasoning |
| `call_margin_equity` | 0.04 | equity fraction | 0.04 preflop margin, declines linearly to zero on river; reasoning |
| `open_equity` | 0.53 | equity fraction | 0.53 baseline; 0.49 in v3; 0.44 in v5, hypothesis-tested on seed 0 |
| `enter_equity` | 0.44 | equity fraction | 0.44 minimum unopened button entry and maximum bluff equity; reasoning |
| `value_equity` | 0.68 | equity fraction | 0.68 baseline free-action value threshold; 0.64 in betting experiment |
| `raise_equity` | 0.8 | equity fraction | 0.80 value threshold facing a wager; reasoning |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | 0.65 bet/raise increment; reasoning |
| `open_size_bb` | 2.5 | big blinds, raise-to | 2.5 opening/limp-isolation size; reasoning |
| `bluff_frequency` | 0.07 | probability per eligible free action | 0.07 baseline; 0.25 in v3; 0.50 in v5, hypothesis-tested on seed 0 |
| `draw_bonus_percentile` | 0.18 | percentile fraction | 0.18 for four-card flush/straight draws on flop/turn; reasoning |

### Results

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| kraken_gpt | 0 | 64 × 250 | -51.54 | 7.73 |
| leviathan_claude | 0 | 64 × 250 | -14.44 | 9.65 |
| megalodon_glm | 0 | 64 × 250 | -13.42 | 10.81 |
| spike_claude | 0 | 64 × 250 | -11.83 | 8.68 |
| faye_gpt | 0 | 64 × 250 | -51.20 | 5.47 |
| windrunner_glm | 0 | 64 × 250 | -28.76 | 8.22 |
| invoker_gpt | 0 | 64 × 250 | -6.78 | 9.16 |
| meepo_claude | 0 | 64 × 250 | +3.52 | 10.10 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +9.90 | 7.42 |
| vicious_glm | 0 | 64 × 250 | +83.02 | 13.53 |
| valka_onderra_claude | 0 | 64 × 250 | +54.08 | 10.79 |
| aggressor | 0 | 64 × 250 | -73.20 | 6.56 |
| alan_wake_claude | 0 | 64 × 250 | +85.12 | 11.37 |
| max_payne_gpt | 0 | 64 × 250 | +66.90 | 8.44 |
| bassander_lin_glm | 0 | 64 × 250 | +135.57 | 12.88 |
| jesse_faden_glm | 0 | 64 × 250 | +263.78 | 18.21 |
| jack_joyce_gemini | 0 | 64 × 250 | +302.73 | 18.79 |
| random | 0 | 64 × 250 | +347.43 | 15.89 |

Weighted score from `v2_field_s0.log`: **+17.12**.

**Verdict:** Kept as the modeling foundation for v3 and v4. Its +17.12 score did not improve on v1 within sampling uncertainty; the draw and limp changes were retained for their explicit poker interpretation, not claimed as a measured gain.

## v3

**What it does:** Keeps the v2 range model. Lowers opening equity from 0.53 to 0.49 and free-action value equity from 0.68 to 0.64; raises eligible bluff frequency from 0.07 to 0.25.

**Why it exists:** Test a related increase in betting frequency without changing inference or bet sizes.

**Decision-code lines:** 83; includes helper and act definitions, excludes imports, constructor, init_state, metadata, comments, blanks and shared helpers.

### Parameters

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `samples` | 128 | holdings/decision | 128 shared range and showdown samples; reasoning |
| `range_power` | 1.0 | dimensionless likelihood exponent | 1.0 baseline; 0.35 tests discounted evidence; 0.0 disables ranges for controls |
| `likelihood_floor` | 0.12 | probability | 0.12 contamination floor; reasoning |
| `likelihood_width` | 0.12 | percentile fraction | 0.12 smooth transition; reasoning |
| `open_percentile` | 0.38 | starting-hand percentile | 0.38 first preflop raise cutoff; reasoning |
| `reraise_percentile` | 0.82 | holding percentile | 0.82 second/later raise cutoff; reasoning |
| `call_percentile` | 0.45 | holding percentile | 0.45 call cutoff; reasoning |
| `bet_percentile` | 0.64 | made-hand percentile | 0.64 first postflop bet cutoff; reasoning |
| `check_discount` | 0.65 | probability fraction | 0.65 discounts strong hands after checks/limps; reasoning |
| `pair_bonus_ranks` | 13.0 | summed rank points | 13-point pocket-pair bonus; reasoning |
| `suited_bonus_ranks` | 3.0 | summed rank points | 3-point suited bonus; reasoning |
| `call_margin_equity` | 0.04 | equity fraction | 0.04 preflop margin, declines linearly to zero on river; reasoning |
| `open_equity` | 0.49 | equity fraction | 0.53 baseline; 0.49 in v3; 0.44 in v5, hypothesis-tested on seed 0 |
| `enter_equity` | 0.44 | equity fraction | 0.44 minimum unopened button entry and maximum bluff equity; reasoning |
| `value_equity` | 0.64 | equity fraction | 0.68 baseline free-action value threshold; 0.64 in betting experiment |
| `raise_equity` | 0.8 | equity fraction | 0.80 value threshold facing a wager; reasoning |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | 0.65 bet/raise increment; reasoning |
| `open_size_bb` | 2.5 | big blinds, raise-to | 2.5 opening/limp-isolation size; reasoning |
| `bluff_frequency` | 0.25 | probability per eligible free action | 0.07 baseline; 0.25 in v3; 0.50 in v5, hypothesis-tested on seed 0 |
| `draw_bonus_percentile` | 0.18 | percentile fraction | 0.18 for four-card flush/straight draws on flop/turn; reasoning |

### Results

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| kraken_gpt | 0 | 32 × 250 | -44.41 | 11.67 |
| leviathan_claude | 0 | 32 × 250 | -7.15 | 13.28 |
| megalodon_glm | 0 | 32 × 250 | +2.76 | 13.62 |
| spike_claude | 0 | 32 × 250 | +24.94 | 14.21 |
| faye_gpt | 0 | 32 × 250 | -19.80 | 6.00 |
| windrunner_glm | 0 | 32 × 250 | -5.96 | 16.45 |
| invoker_gpt | 0 | 32 × 250 | -9.11 | 13.83 |
| meepo_claude | 0 | 32 × 250 | +29.06 | 14.02 |
| hadrian_marlowe_gpt | 0 | 32 × 250 | +2.66 | 13.78 |
| vicious_glm | 0 | 32 × 250 | +69.54 | 22.75 |
| valka_onderra_claude | 0 | 32 × 250 | +56.48 | 19.20 |
| aggressor | 0 | 32 × 250 | -49.39 | 8.94 |
| alan_wake_claude | 0 | 32 × 250 | +66.61 | 15.75 |
| max_payne_gpt | 0 | 32 × 250 | +82.19 | 12.54 |
| bassander_lin_glm | 0 | 32 × 250 | +133.05 | 20.81 |
| jesse_faden_glm | 0 | 32 × 250 | +211.30 | 26.17 |
| jack_joyce_gemini | 0 | 32 × 250 | +324.72 | 23.62 |
| random | 0 | 32 × 250 | +335.67 | 17.69 |
| kraken_gpt | 1000 | 32 × 250 | -18.81 | 9.65 |
| leviathan_claude | 1000 | 32 × 250 | +36.57 | 12.11 |
| megalodon_glm | 1000 | 32 × 250 | +18.07 | 17.11 |
| spike_claude | 1000 | 32 × 250 | +8.14 | 11.13 |
| faye_gpt | 1000 | 32 × 250 | -33.77 | 7.72 |
| windrunner_glm | 1000 | 32 × 250 | +14.47 | 13.04 |
| invoker_gpt | 1000 | 32 × 250 | -5.62 | 11.29 |
| meepo_claude | 1000 | 32 × 250 | -4.70 | 12.29 |
| hadrian_marlowe_gpt | 1000 | 32 × 250 | -6.62 | 10.94 |
| vicious_glm | 1000 | 32 × 250 | +66.86 | 19.38 |
| valka_onderra_claude | 1000 | 32 × 250 | +37.24 | 15.93 |
| aggressor | 1000 | 32 × 250 | -42.64 | 6.05 |
| alan_wake_claude | 1000 | 32 × 250 | +110.81 | 14.35 |
| max_payne_gpt | 1000 | 32 × 250 | +60.46 | 12.01 |
| bassander_lin_glm | 1000 | 32 × 250 | +106.06 | 18.27 |
| jesse_faden_glm | 1000 | 32 × 250 | +260.89 | 28.74 |
| jack_joyce_gemini | 1000 | 32 × 250 | +234.08 | 26.58 |
| random | 1000 | 32 × 250 | +359.92 | 16.07 |
| v1:Bot | 1000 | 32 × 250 | +1.61 | 8.17 |
| v2:Bot | 1 | 64 × 250 | -3.02 | 8.29 |
| v2:Bot | 1000 | 32 × 250 | -2.74 | 9.90 |
| v8:Bot | 1000 | 32 × 250 | +140.97 | 16.64 |

Weighted score from `v3_field_s0.log`: **+23.12**.

Weighted score from `v3_field_s1000.log`: **+25.07**.

**Verdict:** Kept as parent of v5 and selected for deployment. Development +23.12 and held-out +25.07 were consistent; +25.07 was the highest finalist held-out score. Its slightly negative parent matches are an explicit warning, not evidence of a clear loss or win.

## v4

**What it does:** Keeps v2 decisions and features, but raises the full action likelihood to power 0.35 instead of 1.0. Each action is less conclusive, while the whole line still matters.

**Why it exists:** Test whether model uncertainty makes the baseline posterior too concentrated.

**Decision-code lines:** 83; includes helper and act definitions, excludes imports, constructor, init_state, metadata, comments, blanks and shared helpers.

### Parameters

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `samples` | 128 | holdings/decision | 128 shared range and showdown samples; reasoning |
| `range_power` | 0.35 | dimensionless likelihood exponent | 1.0 baseline; 0.35 tests discounted evidence; 0.0 disables ranges for controls |
| `likelihood_floor` | 0.12 | probability | 0.12 contamination floor; reasoning |
| `likelihood_width` | 0.12 | percentile fraction | 0.12 smooth transition; reasoning |
| `open_percentile` | 0.38 | starting-hand percentile | 0.38 first preflop raise cutoff; reasoning |
| `reraise_percentile` | 0.82 | holding percentile | 0.82 second/later raise cutoff; reasoning |
| `call_percentile` | 0.45 | holding percentile | 0.45 call cutoff; reasoning |
| `bet_percentile` | 0.64 | made-hand percentile | 0.64 first postflop bet cutoff; reasoning |
| `check_discount` | 0.65 | probability fraction | 0.65 discounts strong hands after checks/limps; reasoning |
| `pair_bonus_ranks` | 13.0 | summed rank points | 13-point pocket-pair bonus; reasoning |
| `suited_bonus_ranks` | 3.0 | summed rank points | 3-point suited bonus; reasoning |
| `call_margin_equity` | 0.04 | equity fraction | 0.04 preflop margin, declines linearly to zero on river; reasoning |
| `open_equity` | 0.53 | equity fraction | 0.53 baseline; 0.49 in v3; 0.44 in v5, hypothesis-tested on seed 0 |
| `enter_equity` | 0.44 | equity fraction | 0.44 minimum unopened button entry and maximum bluff equity; reasoning |
| `value_equity` | 0.68 | equity fraction | 0.68 baseline free-action value threshold; 0.64 in betting experiment |
| `raise_equity` | 0.8 | equity fraction | 0.80 value threshold facing a wager; reasoning |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | 0.65 bet/raise increment; reasoning |
| `open_size_bb` | 2.5 | big blinds, raise-to | 2.5 opening/limp-isolation size; reasoning |
| `bluff_frequency` | 0.07 | probability per eligible free action | 0.07 baseline; 0.25 in v3; 0.50 in v5, hypothesis-tested on seed 0 |
| `draw_bonus_percentile` | 0.18 | percentile fraction | 0.18 for four-card flush/straight draws on flop/turn; reasoning |

### Results

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| kraken_gpt | 0 | 32 × 250 | -85.96 | 15.37 |
| leviathan_claude | 0 | 32 × 250 | -63.47 | 17.32 |
| megalodon_glm | 0 | 32 × 250 | -67.65 | 19.84 |
| spike_claude | 0 | 32 × 250 | -56.20 | 12.16 |
| faye_gpt | 0 | 32 × 250 | -44.19 | 8.41 |
| windrunner_glm | 0 | 32 × 250 | -0.07 | 16.47 |
| invoker_gpt | 0 | 32 × 250 | -45.65 | 17.63 |
| meepo_claude | 0 | 32 × 250 | -1.54 | 21.44 |
| hadrian_marlowe_gpt | 0 | 32 × 250 | -70.83 | 15.09 |
| vicious_glm | 0 | 32 × 250 | -15.53 | 26.16 |
| valka_onderra_claude | 0 | 32 × 250 | +7.33 | 22.67 |
| aggressor | 0 | 32 × 250 | -45.49 | 15.12 |
| alan_wake_claude | 0 | 32 × 250 | +45.64 | 14.01 |
| max_payne_gpt | 0 | 32 × 250 | +69.83 | 14.77 |
| bassander_lin_glm | 0 | 32 × 250 | +36.68 | 30.16 |
| jesse_faden_glm | 0 | 32 × 250 | +257.03 | 34.05 |
| jack_joyce_gemini | 0 | 32 × 250 | +319.96 | 28.98 |
| random | 0 | 32 × 250 | +425.97 | 23.51 |

Weighted score from `v4_field_s0.log`: **+0.78**.

**Verdict:** Discarded. Discounting all range evidence reduced the weighted score to +0.78 and caused large losses against much of the stronger field.

## v5

**What it does:** Keeps v3 inference, sizing, and value thresholds. Lowers opening equity to 0.44, matching minimum button entry, and raises eligible bluff frequency to 0.50.

**Why it exists:** Test raising all playable unopened button hands and applying more pressure.

**Decision-code lines:** 83; includes helper and act definitions, excludes imports, constructor, init_state, metadata, comments, blanks and shared helpers.

### Parameters

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `samples` | 128 | holdings/decision | 128 shared range and showdown samples; reasoning |
| `range_power` | 1.0 | dimensionless likelihood exponent | 1.0 baseline; 0.35 tests discounted evidence; 0.0 disables ranges for controls |
| `likelihood_floor` | 0.12 | probability | 0.12 contamination floor; reasoning |
| `likelihood_width` | 0.12 | percentile fraction | 0.12 smooth transition; reasoning |
| `open_percentile` | 0.38 | starting-hand percentile | 0.38 first preflop raise cutoff; reasoning |
| `reraise_percentile` | 0.82 | holding percentile | 0.82 second/later raise cutoff; reasoning |
| `call_percentile` | 0.45 | holding percentile | 0.45 call cutoff; reasoning |
| `bet_percentile` | 0.64 | made-hand percentile | 0.64 first postflop bet cutoff; reasoning |
| `check_discount` | 0.65 | probability fraction | 0.65 discounts strong hands after checks/limps; reasoning |
| `pair_bonus_ranks` | 13.0 | summed rank points | 13-point pocket-pair bonus; reasoning |
| `suited_bonus_ranks` | 3.0 | summed rank points | 3-point suited bonus; reasoning |
| `call_margin_equity` | 0.04 | equity fraction | 0.04 preflop margin, declines linearly to zero on river; reasoning |
| `open_equity` | 0.44 | equity fraction | 0.53 baseline; 0.49 in v3; 0.44 in v5, hypothesis-tested on seed 0 |
| `enter_equity` | 0.44 | equity fraction | 0.44 minimum unopened button entry and maximum bluff equity; reasoning |
| `value_equity` | 0.64 | equity fraction | 0.68 baseline free-action value threshold; 0.64 in betting experiment |
| `raise_equity` | 0.8 | equity fraction | 0.80 value threshold facing a wager; reasoning |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | 0.65 bet/raise increment; reasoning |
| `open_size_bb` | 2.5 | big blinds, raise-to | 2.5 opening/limp-isolation size; reasoning |
| `bluff_frequency` | 0.5 | probability per eligible free action | 0.07 baseline; 0.25 in v3; 0.50 in v5, hypothesis-tested on seed 0 |
| `draw_bonus_percentile` | 0.18 | percentile fraction | 0.18 for four-card flush/straight draws on flop/turn; reasoning |

### Results

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| kraken_gpt | 0 | 32 × 250 | -9.03 | 11.85 |
| leviathan_claude | 0 | 32 × 250 | +13.98 | 13.64 |
| megalodon_glm | 0 | 32 × 250 | +9.58 | 18.32 |
| spike_claude | 0 | 32 × 250 | -1.49 | 14.34 |
| faye_gpt | 0 | 32 × 250 | -13.38 | 11.15 |
| windrunner_glm | 0 | 32 × 250 | +16.19 | 12.69 |
| invoker_gpt | 0 | 32 × 250 | -11.31 | 15.65 |
| meepo_claude | 0 | 32 × 250 | +29.15 | 14.14 |
| hadrian_marlowe_gpt | 0 | 32 × 250 | -3.21 | 11.42 |
| vicious_glm | 0 | 32 × 250 | +87.35 | 19.94 |
| valka_onderra_claude | 0 | 32 × 250 | +31.16 | 18.65 |
| aggressor | 0 | 32 × 250 | -25.46 | 8.08 |
| alan_wake_claude | 0 | 32 × 250 | +54.24 | 17.87 |
| max_payne_gpt | 0 | 32 × 250 | +45.47 | 15.04 |
| bassander_lin_glm | 0 | 32 × 250 | +109.50 | 23.43 |
| jesse_faden_glm | 0 | 32 × 250 | +227.71 | 32.17 |
| jack_joyce_gemini | 0 | 32 × 250 | +239.07 | 26.94 |
| random | 0 | 32 × 250 | +344.30 | 24.18 |
| kraken_gpt | 1000 | 32 × 250 | +3.74 | 13.67 |
| leviathan_claude | 1000 | 32 × 250 | +25.73 | 15.19 |
| megalodon_glm | 1000 | 32 × 250 | +30.71 | 16.61 |
| spike_claude | 1000 | 32 × 250 | -4.27 | 9.27 |
| faye_gpt | 1000 | 32 × 250 | -15.86 | 9.90 |
| windrunner_glm | 1000 | 32 × 250 | +6.19 | 13.40 |
| invoker_gpt | 1000 | 32 × 250 | -23.18 | 14.75 |
| meepo_claude | 1000 | 32 × 250 | +16.75 | 13.21 |
| hadrian_marlowe_gpt | 1000 | 32 × 250 | -4.59 | 12.00 |
| vicious_glm | 1000 | 32 × 250 | +45.98 | 20.04 |
| valka_onderra_claude | 1000 | 32 × 250 | +23.48 | 19.55 |
| aggressor | 1000 | 32 × 250 | -9.39 | 7.20 |
| alan_wake_claude | 1000 | 32 × 250 | +66.94 | 11.81 |
| max_payne_gpt | 1000 | 32 × 250 | +55.16 | 12.85 |
| bassander_lin_glm | 1000 | 32 × 250 | +79.40 | 26.68 |
| jesse_faden_glm | 1000 | 32 × 250 | +259.37 | 31.12 |
| jack_joyce_gemini | 1000 | 32 × 250 | +289.32 | 29.32 |
| random | 1000 | 32 × 250 | +331.14 | 21.20 |
| v1:Bot | 1000 | 32 × 250 | +1.58 | 9.77 |
| v2:Bot | 1000 | 32 × 250 | +13.14 | 10.39 |
| v3:Bot | 1000 | 32 × 250 | +9.37 | 11.53 |
| v3:Bot | 2 | 64 × 250 | +2.49 | 7.85 |
| v4:Bot | 1000 | 32 × 250 | +37.20 | 11.34 |
| v6:Bot | 1000 | 32 × 250 | +123.13 | 21.59 |

Weighted score from `v5_field_s0.log`: **+23.15**.

Weighted score from `v5_field_s1000.log`: **+24.17**.

**Verdict:** Discarded as a deployment candidate. Development +23.15 was only 0.03 above v3; held-out +24.17 was below v3. Parent results +2.49 (SE 7.85, seed 2) and +9.37 (SE 11.53, seed 1000) were inconclusive. The extra bluffing and wider openings were not established improvements.

## v6

**What it does:** Identical strategy to its metadata parent, with range_power=0.0 so every legal sampled opponent holding has unit weight. This uniform-hand control is not eligible for phase-6 deployment.

**Why it exists:** Measure the effect of range conditioning while keeping other strategy and sampling settings fixed.

**Decision-code lines:** 83; includes helper and act definitions, excludes imports, constructor, init_state, metadata, comments, blanks and shared helpers.

### Parameters

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `samples` | 128 | holdings/decision | 128 shared range and showdown samples; reasoning |
| `range_power` | 0.0 | dimensionless likelihood exponent | 1.0 baseline; 0.35 tests discounted evidence; 0.0 disables ranges for controls |
| `likelihood_floor` | 0.12 | probability | 0.12 contamination floor; reasoning |
| `likelihood_width` | 0.12 | percentile fraction | 0.12 smooth transition; reasoning |
| `open_percentile` | 0.38 | starting-hand percentile | 0.38 first preflop raise cutoff; reasoning |
| `reraise_percentile` | 0.82 | holding percentile | 0.82 second/later raise cutoff; reasoning |
| `call_percentile` | 0.45 | holding percentile | 0.45 call cutoff; reasoning |
| `bet_percentile` | 0.64 | made-hand percentile | 0.64 first postflop bet cutoff; reasoning |
| `check_discount` | 0.65 | probability fraction | 0.65 discounts strong hands after checks/limps; reasoning |
| `pair_bonus_ranks` | 13.0 | summed rank points | 13-point pocket-pair bonus; reasoning |
| `suited_bonus_ranks` | 3.0 | summed rank points | 3-point suited bonus; reasoning |
| `call_margin_equity` | 0.04 | equity fraction | 0.04 preflop margin, declines linearly to zero on river; reasoning |
| `open_equity` | 0.44 | equity fraction | 0.53 baseline; 0.49 in v3; 0.44 in v5, hypothesis-tested on seed 0 |
| `enter_equity` | 0.44 | equity fraction | 0.44 minimum unopened button entry and maximum bluff equity; reasoning |
| `value_equity` | 0.64 | equity fraction | 0.68 baseline free-action value threshold; 0.64 in betting experiment |
| `raise_equity` | 0.8 | equity fraction | 0.80 value threshold facing a wager; reasoning |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | 0.65 bet/raise increment; reasoning |
| `open_size_bb` | 2.5 | big blinds, raise-to | 2.5 opening/limp-isolation size; reasoning |
| `bluff_frequency` | 0.5 | probability per eligible free action | 0.07 baseline; 0.25 in v3; 0.50 in v5, hypothesis-tested on seed 0 |
| `draw_bonus_percentile` | 0.18 | percentile fraction | 0.18 for four-card flush/straight draws on flop/turn; reasoning |

### Results

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| kraken_gpt | 0 | 32 × 250 | -216.85 | 15.17 |
| leviathan_claude | 0 | 32 × 250 | -267.20 | 24.60 |
| megalodon_glm | 0 | 32 × 250 | -234.84 | 26.07 |
| spike_claude | 0 | 32 × 250 | -186.21 | 22.53 |
| faye_gpt | 0 | 32 × 250 | -68.38 | 10.89 |
| windrunner_glm | 0 | 32 × 250 | -189.90 | 25.27 |
| invoker_gpt | 0 | 32 × 250 | -167.00 | 24.22 |
| meepo_claude | 0 | 32 × 250 | -134.87 | 23.12 |
| hadrian_marlowe_gpt | 0 | 32 × 250 | -172.34 | 19.53 |
| vicious_glm | 0 | 32 × 250 | -302.58 | 38.71 |
| valka_onderra_claude | 0 | 32 × 250 | -215.16 | 33.07 |
| aggressor | 0 | 32 × 250 | -33.90 | 16.50 |
| alan_wake_claude | 0 | 32 × 250 | -130.63 | 22.12 |
| max_payne_gpt | 0 | 32 × 250 | -95.11 | 21.79 |
| bassander_lin_glm | 0 | 32 × 250 | -291.02 | 39.44 |
| jesse_faden_glm | 0 | 32 × 250 | -33.34 | 34.29 |
| jack_joyce_gemini | 0 | 32 × 250 | +11.18 | 42.95 |
| random | 0 | 32 × 250 | +537.46 | 28.36 |

Weighted score from `v6_field_s0.log`: **-120.75**.

**Verdict:** Uniform-hand control of v5 only; not eligible for deployment. Its -120.75 field score, compared with v5 +23.15 at the same seed and sample size, demonstrated the value of action-based range conditioning for that policy.

## v7

**What it does:** Keeps v5 decisions and range features. Reconstructs prior contributions and pot sizes from history; a raise increment below half the prior pot supplies proportionally less strength evidence.

**Why it exists:** Test whether distinguishing inexpensive raises from substantial wagers makes action-based ranges more robust.

**Decision-code lines:** 94; includes helper and act definitions, excludes imports, constructor, init_state, metadata, comments, blanks and shared helpers.

### Parameters

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `samples` | 128 | holdings/decision | 128 shared range and showdown samples; reasoning |
| `range_power` | 1.0 | dimensionless likelihood exponent | 1.0 baseline; 0.35 tests discounted evidence; 0.0 disables ranges for controls |
| `likelihood_floor` | 0.12 | probability | 0.12 contamination floor; reasoning |
| `likelihood_width` | 0.12 | percentile fraction | 0.12 smooth transition; reasoning |
| `open_percentile` | 0.38 | starting-hand percentile | 0.38 first preflop raise cutoff; reasoning |
| `reraise_percentile` | 0.82 | holding percentile | 0.82 second/later raise cutoff; reasoning |
| `call_percentile` | 0.45 | holding percentile | 0.45 call cutoff; reasoning |
| `bet_percentile` | 0.64 | made-hand percentile | 0.64 first postflop bet cutoff; reasoning |
| `check_discount` | 0.65 | probability fraction | 0.65 discounts strong hands after checks/limps; reasoning |
| `pair_bonus_ranks` | 13.0 | summed rank points | 13-point pocket-pair bonus; reasoning |
| `suited_bonus_ranks` | 3.0 | summed rank points | 3-point suited bonus; reasoning |
| `call_margin_equity` | 0.04 | equity fraction | 0.04 preflop margin, declines linearly to zero on river; reasoning |
| `open_equity` | 0.44 | equity fraction | 0.53 baseline; 0.49 in v3; 0.44 in v5, hypothesis-tested on seed 0 |
| `enter_equity` | 0.44 | equity fraction | 0.44 minimum unopened button entry and maximum bluff equity; reasoning |
| `value_equity` | 0.64 | equity fraction | 0.68 baseline free-action value threshold; 0.64 in betting experiment |
| `raise_equity` | 0.8 | equity fraction | 0.80 value threshold facing a wager; reasoning |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | 0.65 bet/raise increment; reasoning |
| `open_size_bb` | 2.5 | big blinds, raise-to | 2.5 opening/limp-isolation size; reasoning |
| `bluff_frequency` | 0.5 | probability per eligible free action | 0.07 baseline; 0.25 in v3; 0.50 in v5, hypothesis-tested on seed 0 |
| `draw_bonus_percentile` | 0.18 | percentile fraction | 0.18 for four-card flush/straight draws on flop/turn; reasoning |
| `informative_raise_pot_fraction` | 0.5 | raise increment / pot before action | 0.50: half-pot raises carry full evidence; smaller raises carry proportional evidence, reasoning |

### Results

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| kraken_gpt | 0 | 32 × 250 | -9.74 | 11.56 |
| leviathan_claude | 0 | 32 × 250 | +11.07 | 13.05 |
| megalodon_glm | 0 | 32 × 250 | +6.69 | 18.54 |
| spike_claude | 0 | 32 × 250 | -2.22 | 14.28 |
| faye_gpt | 0 | 32 × 250 | -11.42 | 7.90 |
| windrunner_glm | 0 | 32 × 250 | +19.31 | 13.33 |
| invoker_gpt | 0 | 32 × 250 | -11.59 | 15.70 |
| meepo_claude | 0 | 32 × 250 | +27.38 | 14.29 |
| hadrian_marlowe_gpt | 0 | 32 × 250 | -1.73 | 11.14 |
| vicious_glm | 0 | 32 × 250 | +89.68 | 20.22 |
| valka_onderra_claude | 0 | 32 × 250 | +31.16 | 18.65 |
| aggressor | 0 | 32 × 250 | -25.46 | 8.08 |
| alan_wake_claude | 0 | 32 × 250 | +54.24 | 17.87 |
| max_payne_gpt | 0 | 32 × 250 | +45.47 | 15.04 |
| bassander_lin_glm | 0 | 32 × 250 | +109.50 | 23.43 |
| jesse_faden_glm | 0 | 32 × 250 | +226.46 | 31.82 |
| jack_joyce_gemini | 0 | 32 × 250 | +243.59 | 27.38 |
| random | 0 | 32 × 250 | +332.27 | 26.38 |
| kraken_gpt | 1000 | 32 × 250 | +3.74 | 13.67 |
| leviathan_claude | 1000 | 32 × 250 | +27.26 | 15.20 |
| megalodon_glm | 1000 | 32 × 250 | +33.88 | 17.13 |
| spike_claude | 1000 | 32 × 250 | -6.17 | 9.82 |
| faye_gpt | 1000 | 32 × 250 | -11.41 | 11.10 |
| windrunner_glm | 1000 | 32 × 250 | +5.62 | 13.38 |
| invoker_gpt | 1000 | 32 × 250 | -26.24 | 14.23 |
| meepo_claude | 1000 | 32 × 250 | +15.37 | 12.91 |
| hadrian_marlowe_gpt | 1000 | 32 × 250 | -4.58 | 12.04 |
| vicious_glm | 1000 | 32 × 250 | +45.81 | 20.01 |
| valka_onderra_claude | 1000 | 32 × 250 | +22.96 | 19.54 |
| aggressor | 1000 | 32 × 250 | -9.39 | 7.20 |
| alan_wake_claude | 1000 | 32 × 250 | +66.14 | 11.54 |
| max_payne_gpt | 1000 | 32 × 250 | +55.16 | 12.85 |
| bassander_lin_glm | 1000 | 32 × 250 | +79.40 | 26.68 |
| jesse_faden_glm | 1000 | 32 × 250 | +259.90 | 31.10 |
| jack_joyce_gemini | 1000 | 32 × 250 | +285.32 | 28.82 |
| random | 1000 | 32 × 250 | +367.26 | 24.66 |
| v1:Bot | 1000 | 32 × 250 | +2.17 | 9.88 |
| v2:Bot | 1000 | 32 × 250 | +12.99 | 10.39 |
| v3:Bot | 1000 | 32 × 250 | +10.32 | 11.69 |
| v4:Bot | 1000 | 32 × 250 | +37.64 | 11.43 |
| v5:Bot | 2 | 64 × 250 | -5.56 | 7.20 |

Weighted score from `v7_field_s0.log`: **+22.90**.

Weighted score from `v7_field_s1000.log`: **+24.56**.

**Verdict:** Discarded. The raise-size model changed the intended diagnostic range but did not improve the primary score: +22.90 development and +24.56 held out. It adds 11 decision lines, and its parent result -5.56 (SE 7.20) was inconclusive. It supplied no compelling reason to replace v3.

## v8

**What it does:** Identical strategy to its metadata parent, with range_power=0.0 so every legal sampled opponent holding has unit weight. This uniform-hand control is not eligible for phase-6 deployment.

**Why it exists:** Measure the effect of range conditioning while keeping other strategy and sampling settings fixed.

**Decision-code lines:** 83; includes helper and act definitions, excludes imports, constructor, init_state, metadata, comments, blanks and shared helpers.

### Parameters

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `samples` | 128 | holdings/decision | 128 shared range and showdown samples; reasoning |
| `range_power` | 0.0 | dimensionless likelihood exponent | 1.0 baseline; 0.35 tests discounted evidence; 0.0 disables ranges for controls |
| `likelihood_floor` | 0.12 | probability | 0.12 contamination floor; reasoning |
| `likelihood_width` | 0.12 | percentile fraction | 0.12 smooth transition; reasoning |
| `open_percentile` | 0.38 | starting-hand percentile | 0.38 first preflop raise cutoff; reasoning |
| `reraise_percentile` | 0.82 | holding percentile | 0.82 second/later raise cutoff; reasoning |
| `call_percentile` | 0.45 | holding percentile | 0.45 call cutoff; reasoning |
| `bet_percentile` | 0.64 | made-hand percentile | 0.64 first postflop bet cutoff; reasoning |
| `check_discount` | 0.65 | probability fraction | 0.65 discounts strong hands after checks/limps; reasoning |
| `pair_bonus_ranks` | 13.0 | summed rank points | 13-point pocket-pair bonus; reasoning |
| `suited_bonus_ranks` | 3.0 | summed rank points | 3-point suited bonus; reasoning |
| `call_margin_equity` | 0.04 | equity fraction | 0.04 preflop margin, declines linearly to zero on river; reasoning |
| `open_equity` | 0.49 | equity fraction | 0.53 baseline; 0.49 in v3; 0.44 in v5, hypothesis-tested on seed 0 |
| `enter_equity` | 0.44 | equity fraction | 0.44 minimum unopened button entry and maximum bluff equity; reasoning |
| `value_equity` | 0.64 | equity fraction | 0.68 baseline free-action value threshold; 0.64 in betting experiment |
| `raise_equity` | 0.8 | equity fraction | 0.80 value threshold facing a wager; reasoning |
| `bet_pot_fraction` | 0.65 | fraction of pot after calling | 0.65 bet/raise increment; reasoning |
| `open_size_bb` | 2.5 | big blinds, raise-to | 2.5 opening/limp-isolation size; reasoning |
| `bluff_frequency` | 0.25 | probability per eligible free action | 0.07 baseline; 0.25 in v3; 0.50 in v5, hypothesis-tested on seed 0 |
| `draw_bonus_percentile` | 0.18 | percentile fraction | 0.18 for four-card flush/straight draws on flop/turn; reasoning |

### Results

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| kraken_gpt | 0 | 16 × 250 | -219.31 | 29.13 |
| leviathan_claude | 0 | 16 × 250 | -255.85 | 35.41 |
| megalodon_glm | 0 | 16 × 250 | -167.50 | 44.28 |
| spike_claude | 0 | 16 × 250 | -205.05 | 25.41 |
| faye_gpt | 0 | 16 × 250 | -85.25 | 20.79 |
| windrunner_glm | 0 | 16 × 250 | -168.99 | 33.13 |
| invoker_gpt | 0 | 16 × 250 | -139.99 | 36.48 |
| meepo_claude | 0 | 16 × 250 | -102.72 | 31.46 |
| hadrian_marlowe_gpt | 0 | 16 × 250 | -110.81 | 33.38 |
| vicious_glm | 0 | 16 × 250 | -230.72 | 48.46 |
| valka_onderra_claude | 0 | 16 × 250 | -150.89 | 43.06 |
| aggressor | 0 | 16 × 250 | +9.95 | 27.94 |
| alan_wake_claude | 0 | 16 × 250 | -61.42 | 33.32 |
| max_payne_gpt | 0 | 16 × 250 | +7.54 | 36.01 |
| bassander_lin_glm | 0 | 16 × 250 | -230.26 | 46.26 |
| jesse_faden_glm | 0 | 16 × 250 | +117.92 | 70.57 |
| jack_joyce_gemini | 0 | 16 × 250 | +173.82 | 70.04 |
| random | 0 | 16 × 250 | +551.89 | 29.44 |

Weighted score from `v8_field_s0.log`: **-76.91**.

**Verdict:** Uniform-hand control of v3 only; not eligible for deployment. Its -76.91 field score and large direct loss to v3 support retaining the range model.

## Deployed range model

The prior is uniform over legal opponent holdings after removing our cards and visible board cards. Preflop merit sums both ranks, adding named pair and suited bonuses. Later streets use evaluated made-hand percentiles on each historical board prefix; ties receive shared midranks. Four-card straight/flush draws get a named percentile bonus on flop and turn.

Each opponent action contributes a smooth likelihood. First preflop raises use the open threshold; first postflop bets use the betting threshold; second or later raises on a street use the tighter reraise threshold. Calls favor the upper holding distribution. Checks discount strong hands. Position distinguishes a big blind's free check from a button completion; button limps use the weaker passive model. Every valid opponent bet, raise, call and check contributes; blind posts do not. Folds terminate the relevant heads-up sequence.

Likelihoods have a nonzero floor for unusual play. In v7, a raise increment smaller than half the pot before the action contributes proportionally less log-likelihood evidence; calls and checks retain full weight. Prior pots are reconstructed from actual contribution increments, including blinds. Their product raised to range_power gives the holding weight. New board cards remove impossible holdings, while earlier actions retain their historical board context. Range-weighted showdown share drives entry, folds, calls, value bets, raises and bluff eligibility. Pot odds plus a declining future-street margin price calls. Sizing uses legal raise-to totals.

### Measured range table

`measure_ranges.py` calls the deployed range function on constructed histories. Flop: **K♠ 9♥ 4♠**; turn: **7♣**; river: **2♦**. Examples use a 2.5bb open, 9bb 3-bet, 3.5bb flop bet, 11bb flop check-raise, 8bb turn bet and 17.5bb river bet. Versions 1–6 distinguish action order but ignore nonblind bet sizes in inference. Version 7 additionally scales raise evidence by the raise increment divided by the prior pot.

Each row averages 10 batches of 128 sampled legal holdings, seeds 0–9, with no fixed hero hand. Retained mass is mean unnormalised weight multiplied by the fraction of all 1,326 starting combinations compatible with the board. This is a **soft weighted share of all starting hands**, not a hard exclusion: every legal holding retains positive weight. These are Monte Carlo measurements, not exact enumeration. Pocket-pair and two-broadway columns describe normalized posterior composition.

Effective sample size in the deployed model averages 105.3 of 128 after the button open and 38.1 after the three-street line. The extra small-turn-re-raise diagnostic is retained in the JSON measurements.

| Opponent line | Weighted starting-hand mass | Posterior pocket pairs | Posterior two broadways | Description |
|---|---:|---:|---:|---|
| Button opens | 66.20% | 8.33% | 22.34% | Broad; favors pairs, higher ranks and suited hands. |
| Button limps | 64.77% | 4.09% | 10.13% | Weaker completions; premiums discounted, still possible. |
| Big blind calls an open | 60.27% | 8.92% | 24.27% | Medium/high preflop merit, pairs and suited high cards. |
| Big blind 3-bets | 29.92% | 12.55% | 34.48% | High pairs and broadways favored; weak-hand tail retained. |
| Button c-bets flop after opening | 30.76% | 12.00% | 31.20% | Opening range that connects, strong draws, some bluffs. |
| Big blind check-raises flop | 11.72% | 11.36% | 30.64% | Preflop caller that checked, then represented strength or a semibluff. |
| Opponent bets turn after flop bet/call | 19.18% | 13.63% | 34.54% | Whole opening and flop line, narrowed again on the turn. |
| Opponent bets all three streets | 13.60% | 14.78% | 33.68% | Complete betting line; strong made hands favored, missed draws reduced. |

### What range conditioning is worth

The deployed v3 uses the same policy and sampling settings as v8, except that v8 sets range_power=0.0, giving every sampled legal holding weight 1. On development seed 0, v3 scored **+23.12** with 32 × 250 hands per opponent; v8 scored **−76.91** with 16 × 250, a **100.03-point** weighted-score difference. The sample sizes differ, so this is not a paired equal-size estimate. In the direct held-out match, v3 beat v8 by **+140.97 bb/100, SE 16.64**, at 32 × 250 hands, seed 1000. Together these support range conditioning for this policy on the tested opponents; they do not establish that the inferred probabilities are calibrated. As an additional equal-size check, v5 scored +23.15 against its uniform control v6 at −120.75 on the same seed-0 field and 32 × 250 size; v5 beat v6 head to head by +123.13 bb/100, SE 21.59, on seed 1000.

## Final choice

The deployment export is **from .v3 import Bot as DeployedBot**. Each finalist received one field run on held-out seed 1000 at 32 × 250 hands per opponent. Results were **v3 +25.07**, **v7 +24.56**, and **v5 +24.17**. v3 had the highest held-out weighted score. The margins are small relative to the per-opponent uncertainty; this selection is not a claim that v3 is statistically superior to both alternatives. The later variants offered no convincing benefit, so the moderate-pressure v3 was retained. v5 had the highest development score (+23.15 versus v3 +23.12); that 0.03-point difference is noise and was not used to choose it.

The parent warning remains: v3 scored **−2.74 bb/100, SE 9.90**, against v2 on held-out seed 1000, and **−3.02, SE 8.29**, on development seed 1. Against v1, its held-out result was **+1.61, SE 8.17**. These are close to even, so the field advantage is not a demonstrated advantage over related strategies. In the held-out field, material weaknesses included **aggressor −42.64, SE 6.05**, and **faye_gpt −33.77, SE 7.72**. No claim is made to match the prompt’s strongest-bot calibration or to dominate unseen phase-6 lineages.

### Held-out deployed results against the field

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| kraken_gpt | 1000 | 32 × 250 | -18.81 | 9.65 |
| leviathan_claude | 1000 | 32 × 250 | +36.57 | 12.11 |
| megalodon_glm | 1000 | 32 × 250 | +18.07 | 17.11 |
| spike_claude | 1000 | 32 × 250 | +8.14 | 11.13 |
| faye_gpt | 1000 | 32 × 250 | -33.77 | 7.72 |
| windrunner_glm | 1000 | 32 × 250 | +14.47 | 13.04 |
| invoker_gpt | 1000 | 32 × 250 | -5.62 | 11.29 |
| meepo_claude | 1000 | 32 × 250 | -4.70 | 12.29 |
| hadrian_marlowe_gpt | 1000 | 32 × 250 | -6.62 | 10.94 |
| vicious_glm | 1000 | 32 × 250 | +66.86 | 19.38 |
| valka_onderra_claude | 1000 | 32 × 250 | +37.24 | 15.93 |
| aggressor | 1000 | 32 × 250 | -42.64 | 6.05 |
| alan_wake_claude | 1000 | 32 × 250 | +110.81 | 14.35 |
| max_payne_gpt | 1000 | 32 × 250 | +60.46 | 12.01 |
| bassander_lin_glm | 1000 | 32 × 250 | +106.06 | 18.27 |
| jesse_faden_glm | 1000 | 32 × 250 | +260.89 | 28.74 |
| jack_joyce_gemini | 1000 | 32 × 250 | +234.08 | 26.58 |
| random | 1000 | 32 × 250 | +359.92 | 16.07 |

### Results against every earlier version

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| v1:Bot | 1000 | 32 × 250 | +1.61 | 8.17 |
| v2:Bot | 1 | 64 × 250 | -3.02 | 8.29 |
| v2:Bot | 1000 | 32 × 250 | -2.74 | 9.90 |
| v8:Bot | 1000 | 32 × 250 | +140.97 | 16.64 |

### Known weaknesses

This is an action-likelihood model, not an equilibrium solution. It does not fit calibrated bet-size distributions or learn between hands. Preflop merit omits explicit connectivity. Draw bonuses are approximate and include some shared-board draws. Repeated evidence can be overconfident, and 128 importance samples can leave few effective samples on narrow lines. Value sizing and future-street margins are coarse. Call pricing uses the displayed pot and capped call cost without separately modeling unmatched-bet refunds. One held-out seed cannot establish universal superiority or results against unseen phase-6 lineages.

## Verification and scratch files

`verify_delivery.py` checks complete field coverage, the deployment export, earlier-version comparisons and the matching uniform control. `audit.py` checks statelessness, unchanged state return, sample limits, decision lines, serializable metadata, one Bot class per version, and source hashes. Smoke tests and full benchmarks exercise engine compilation and action compatibility. `bench.py` and `equity.py` were not edited. No commits were made.

Reproducibility files retained: `check.log`, `vN_smoke.log`, field/head-to-head logs, measured `vN_ranges.json` and `.log` files, `measure_ranges.py`, `report_tools.py`, `write_report.py`, `audit.py`, `audit.log`, `source_hashes.json`, `selection.json`, `run_notes.txt`, `run_series.py`, `evaluate_finalist.py`, `evaluate_pairs.py`, `verify_delivery.py`, and `delivery_checks.log`. The bot only needs its deployed version, the one-line package export, and the provided helper at runtime.

