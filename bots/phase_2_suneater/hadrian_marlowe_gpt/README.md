# hadrian_marlowe_gpt

- **Lineage:** `phase_2_suneater/hadrian_marlowe_gpt`
- **Model:** GPT-6 (Codex; exact serving variant unavailable)
- **Prompt:** `botgen-v2`
- **Engine:** `02abc49` + working tree, `engine_sha256 295477eff7637056`
- **Execution:** All nine versions compiled and were benchmarked on the machine holding this directory, using the existing project Python environment. No packages were installed.
- **Recorded wall-clock time:** 30.3 minutes, 2026-10-14 02:12:06 to 02:42:26 Australia/Adelaide. The attachment and initial directory listing preceded the recorded start by approximately one minute.
- **Iteration stopped:** 16.8 minutes after the recorded start, before the 50-minute deadline; a local 50-minute reminder was started. Only evaluation, documentation and verification followed the freeze.
- **Deployment:** `from .v6 import Bot as DeployedBot`

## Protocol, evidence and limitations

Every benchmark number below comes from a saved local log. Development used seeds 0–2; seed 1000 was reserved for frozen evaluation. Each finalist was run exactly once against each phase-1 final and RandomBot on seed 1000 at 512 tables × 250 hands, and once against every earlier version at 128 × 250. The primary score is the unweighted mean against the four phase-1 finals; RandomBot never enters that score.

The provided benchmark balances button position and reports table-clustered standard errors. Individual SEs are reported as printed. A mean across opponents is not assigned an independence-based SE because the seed/table streams are shared. For selection, the average marginal SE bounds each candidate mean SE by the triangle inequality; the sum of those bounds conservatively bounds uncertainty in their difference. This is an uncertainty heuristic using estimated SEs, not a multiple-comparison-adjusted significance test. The selection rule was saved in `evaluation_lock.json` before reading held-out results.

**Disclosed deviations:** The first discovery command listed filenames outside this bot directory while the task attachment was being read. No engine source, other-lineage source, or git state was opened or used. All subsequent project reads and writes stayed in this directory, apart from invoking the explicitly allowed Python environment. Also, v1 has 43 decision lines, two below the nominal 45–55 range; it was preserved once benchmarked instead of silently edited. Versions v2–v9 are within range. The initial plain `uv run` hit a cache permission error; later invocations used an offline, no-sync cache inside this directory.

All versions are standalone, stateless classes: `init_state` returns `None`, `act` returns the supplied state unchanged, and none defines `observe`. They import only JAX, the documented `poker_bots` contract and the shared equity helpers. Constructor defaults are reasoning choices, not fitted lookup tables. Versioned changes are frozen; their hashes are in `version_hashes.json`.

## Ideas borrowed from the supplied opponents

I read only the four frozen opponents supplied here. Alan suggested discounting equity after opponent aggression; Max supplied the general realization-discount and explicit-opening-size ideas; Jesse illustrated pot-odds/value-betting structure; Jack suggested mixed bluffing and stronger-hand sizing. The new policy uses ordinary within-hand raises, position, pot odds and effective stacks. It never identifies a bot, matches a particular opponent bet size, or remembers prior hands.

## v1

**What it does:** Opens a broad range on the button and isolates limps more selectively from the big blind. Calls according to pot odds after a multiplicative aggression discount, and uses conservative value raises. Caps raises at the effective stack.

**Why it exists:** Establish a simple heads-up baseline with position-aware preflop play.

**Parent:** None. **Decision line count:** 43 including the `act` declaration (42 body lines); no extra decision helpers.

**Parameters:** All unchanged defaults are inherited reasoning choices. No grid search or automated fitting was performed.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo samples per decision | Reasoning default tested in this version |
| `button_open_equity` | 0.43 | raw equity probability | Reasoning default tested in this version |
| `isolation_equity` | 0.6 | raw equity probability | Reasoning default tested in this version |
| `preflop_value_equity` | 0.69 | adjusted equity probability | Reasoning default tested in this version |
| `postflop_value_equity` | 0.68 | adjusted equity probability | Reasoning default tested in this version |
| `reraise_margin_equity` | 0.12 | equity probability increment when facing a bet | Reasoning default tested in this version |
| `raise_discount_fraction` | 0.15 | equity fraction removed per current-street opposing raise | Reasoning default tested in this version |
| `call_margin_equity` | 0.03 | equity probability above break-even call odds | Reasoning default tested in this version |
| `realization_fraction` | 0.9 | fraction of adjusted equity before possible future betting | Reasoning default tested in this version |
| `opening_size_bb` | 2.5 | big blinds (street raise-to total) | Reasoning default tested in this version |
| `bet_pot_fraction` | 0.66 | fraction of contestable pot after calling | Reasoning default tested in this version |
| `preflop_reraise_multiple` | 3 | multiple of the current street bet | Reasoning default tested in this version |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| random | 0 | 2 × 20 | +501.25 | 281.25 |
| alan_wake_claude | 0 | 64 × 250 | +20.54 | 10.56 |
| jack_joyce_gemini | 0 | 64 × 250 | +174.31 | 21.45 |
| jesse_faden_glm | 0 | 64 × 250 | +176.98 | 23.48 |
| max_payne_gpt | 0 | 64 × 250 | +31.82 | 10.55 |

Mean against finals, seed 0, 64 tables: **+100.91 bb/100**.

**Verdict:** Retained as parent of v2. Its range discount forgets earlier streets and over-discounts near-nut hands; its 43-line baseline also falls below the nominal 45-line lower end.

## v2

**What it does:** Replaces the multiplicative equity haircut with an approximate truncated-range adjustment. Earlier opponent raises retain half their weight on later streets, while equity near one remains near one.

**Why it exists:** Fix the baseline's disappearing range information and excessive caution with very strong hands as one related range-model change.

**Parent:** 1. **Decision line count:** 47 including the `act` declaration (46 body lines); no extra decision helpers.

**Parameters:** All unchanged defaults are inherited reasoning choices. No grid search or automated fitting was performed.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo samples per decision | Inherited reasoning default |
| `button_open_equity` | 0.43 | raw equity probability | Inherited reasoning default |
| `isolation_equity` | 0.6 | raw equity probability | Inherited reasoning default |
| `preflop_value_equity` | 0.69 | adjusted equity probability | Inherited reasoning default |
| `postflop_value_equity` | 0.68 | adjusted equity probability | Inherited reasoning default |
| `reraise_margin_equity` | 0.12 | equity probability increment when facing a bet | Inherited reasoning default |
| `range_exclusion_fraction` | 0.3 | weak-range fraction removed per raise-equivalent | Reasoning default tested in this version |
| `prior_raise_weight` | 0.5 | current-street raise-equivalents per earlier raise | Reasoning default tested in this version |
| `call_margin_equity` | 0.03 | equity probability above break-even call odds | Inherited reasoning default |
| `realization_fraction` | 0.9 | fraction of adjusted equity before possible future betting | Inherited reasoning default |
| `opening_size_bb` | 2.5 | big blinds (street raise-to total) | Inherited reasoning default |
| `bet_pot_fraction` | 0.66 | fraction of contestable pot after calling | Inherited reasoning default |
| `preflop_reraise_multiple` | 3 | multiple of the current street bet | Inherited reasoning default |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| alan_wake_claude | 0 | 64 × 250 | +77.07 | 9.45 |
| jack_joyce_gemini | 0 | 64 × 250 | +270.85 | 18.49 |
| jesse_faden_glm | 0 | 64 × 250 | +280.92 | 17.78 |
| max_payne_gpt | 0 | 64 × 250 | +66.53 | 6.16 |
| alan_wake_claude | 1 | 64 × 250 | +101.70 | 9.62 |
| jack_joyce_gemini | 1 | 64 × 250 | +294.61 | 18.98 |
| jesse_faden_glm | 1 | 64 × 250 | +254.69 | 18.51 |
| max_payne_gpt | 1 | 64 × 250 | +64.46 | 7.29 |
| v1:Bot | 1 | 64 × 250 | +51.93 | 8.84 |

Mean against finals, seed 0, 64 tables: **+173.84 bb/100**.

Mean against finals, seed 1, 64 tables: **+178.86 bb/100**.

**Verdict:** Retained as parent of v3. The large benchmark improvement repeated on seed 1, and it beat v1 directly by +51.93 bb/100 (SE 8.84).

## v3

**What it does:** Keeps v2's range model. Bets 1.25 pots when adjusted equity is at least 0.90, and otherwise retains the 0.66-pot sizing.

**Why it exists:** Test whether very strong hands earn more with larger value bets rather than using one size for every hand.

**Parent:** 2. **Decision line count:** 49 including the `act` declaration (48 body lines); no extra decision helpers.

**Parameters:** All unchanged defaults are inherited reasoning choices. No grid search or automated fitting was performed.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo samples per decision | Inherited reasoning default |
| `button_open_equity` | 0.43 | raw equity probability | Inherited reasoning default |
| `isolation_equity` | 0.6 | raw equity probability | Inherited reasoning default |
| `preflop_value_equity` | 0.69 | adjusted equity probability | Inherited reasoning default |
| `postflop_value_equity` | 0.68 | adjusted equity probability | Inherited reasoning default |
| `reraise_margin_equity` | 0.12 | equity probability increment when facing a bet | Inherited reasoning default |
| `range_exclusion_fraction` | 0.3 | weak-range fraction removed per raise-equivalent | Inherited reasoning default |
| `prior_raise_weight` | 0.5 | current-street raise-equivalents per earlier raise | Inherited reasoning default |
| `call_margin_equity` | 0.03 | equity probability above break-even call odds | Inherited reasoning default |
| `realization_fraction` | 0.9 | fraction of adjusted equity before possible future betting | Inherited reasoning default |
| `opening_size_bb` | 2.5 | big blinds (street raise-to total) | Inherited reasoning default |
| `premium_equity` | 0.9 | adjusted equity probability | Reasoning default tested in this version |
| `premium_pot_fraction` | 1.25 | fraction of contestable pot after calling | Reasoning default tested in this version |
| `bet_pot_fraction` | 0.66 | fraction of contestable pot after calling | Inherited reasoning default |
| `preflop_reraise_multiple` | 3 | multiple of the current street bet | Inherited reasoning default |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| alan_wake_claude | 0 | 64 × 250 | +117.60 | 9.20 |
| jack_joyce_gemini | 0 | 64 × 250 | +285.15 | 17.34 |
| jesse_faden_glm | 0 | 64 × 250 | +279.09 | 16.95 |
| max_payne_gpt | 0 | 64 × 250 | +68.67 | 6.08 |
| v2:Bot | 1 | 64 × 250 | +0.96 | 6.49 |
| alan_wake_claude | 1 | 128 × 250 | +116.43 | 7.46 |
| jack_joyce_gemini | 1 | 128 × 250 | +305.53 | 13.58 |
| jesse_faden_glm | 1 | 128 × 250 | +275.33 | 12.09 |
| max_payne_gpt | 1 | 128 × 250 | +66.05 | 4.62 |

Mean against finals, seed 0, 64 tables: **+187.63 bb/100**.

Mean against finals, seed 1, 128 tables: **+190.84 bb/100**.

**Verdict:** Retained as parent of v4. Improvements against Alan repeated on the larger second-seed run; the direct result against v2 was a tie within error, not evidence of dominance.

## v4

**What it does:** Removes an opponent's unmatched overbet from the pot used for calling odds. Otherwise uses v3's decisions and sizing.

**Why it exists:** Fix a chip-accounting issue when the hero has fewer chips than the opponent's wager.

**Parent:** 3. **Decision line count:** 52 including the `act` declaration (51 body lines); no extra decision helpers.

**Parameters:** All unchanged defaults are inherited reasoning choices. No grid search or automated fitting was performed.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo samples per decision | Inherited reasoning default |
| `button_open_equity` | 0.43 | raw equity probability | Inherited reasoning default |
| `isolation_equity` | 0.6 | raw equity probability | Inherited reasoning default |
| `preflop_value_equity` | 0.69 | adjusted equity probability | Inherited reasoning default |
| `postflop_value_equity` | 0.68 | adjusted equity probability | Inherited reasoning default |
| `reraise_margin_equity` | 0.12 | equity probability increment when facing a bet | Inherited reasoning default |
| `range_exclusion_fraction` | 0.3 | weak-range fraction removed per raise-equivalent | Inherited reasoning default |
| `prior_raise_weight` | 0.5 | current-street raise-equivalents per earlier raise | Inherited reasoning default |
| `call_margin_equity` | 0.03 | equity probability above break-even call odds | Inherited reasoning default |
| `realization_fraction` | 0.9 | fraction of adjusted equity before possible future betting | Inherited reasoning default |
| `opening_size_bb` | 2.5 | big blinds (street raise-to total) | Inherited reasoning default |
| `premium_equity` | 0.9 | adjusted equity probability | Inherited reasoning default |
| `premium_pot_fraction` | 1.25 | fraction of contestable pot after calling | Inherited reasoning default |
| `bet_pot_fraction` | 0.66 | fraction of contestable pot after calling | Inherited reasoning default |
| `preflop_reraise_multiple` | 3 | multiple of the current street bet | Inherited reasoning default |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| alan_wake_claude | 0 | 64 × 250 | +117.60 | 9.20 |
| jack_joyce_gemini | 0 | 64 × 250 | +286.07 | 17.34 |
| jesse_faden_glm | 0 | 64 × 250 | +279.09 | 16.95 |
| max_payne_gpt | 0 | 64 × 250 | +68.67 | 6.08 |
| v3:Bot | 1 | 64 × 250 | -5.44 | 6.43 |

Mean against finals, seed 0, 64 tables: **+187.86 bb/100**.

**Verdict:** Retained as parent of v5 and v6 for the accounting correction. Score differences were small and are not claimed as statistically meaningful.

## v5

**What it does:** Adds a 15% bluff chance with less than 0.35 raw equity when checked to on the button. Medium-strength showdown hands remain in the checking range; value betting is unchanged.

**Why it exists:** Test a small positional bluff component against the value-only parent, without recognizing opponent identities or exact bet sizes.

**Parent:** 4. **Decision line count:** 55 including the `act` declaration (54 body lines); no extra decision helpers.

**Parameters:** All unchanged defaults are inherited reasoning choices. No grid search or automated fitting was performed.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo samples per decision | Inherited reasoning default |
| `button_open_equity` | 0.43 | raw equity probability | Inherited reasoning default |
| `isolation_equity` | 0.6 | raw equity probability | Inherited reasoning default |
| `preflop_value_equity` | 0.69 | adjusted equity probability | Inherited reasoning default |
| `postflop_value_equity` | 0.68 | adjusted equity probability | Inherited reasoning default |
| `reraise_margin_equity` | 0.12 | equity probability increment when facing a bet | Inherited reasoning default |
| `range_exclusion_fraction` | 0.3 | weak-range fraction removed per raise-equivalent | Inherited reasoning default |
| `prior_raise_weight` | 0.5 | current-street raise-equivalents per earlier raise | Inherited reasoning default |
| `call_margin_equity` | 0.03 | equity probability above break-even call odds | Inherited reasoning default |
| `realization_fraction` | 0.9 | fraction of adjusted equity before possible future betting | Inherited reasoning default |
| `opening_size_bb` | 2.5 | big blinds (street raise-to total) | Inherited reasoning default |
| `bluff_frequency` | 0.15 | probability per eligible positional bluff opportunity | Reasoning default tested in this version |
| `bluff_max_equity` | 0.35 | maximum raw equity probability eligible to bluff | Reasoning default tested in this version |
| `premium_equity` | 0.9 | adjusted equity probability | Inherited reasoning default |
| `premium_pot_fraction` | 1.25 | fraction of contestable pot after calling | Inherited reasoning default |
| `bet_pot_fraction` | 0.66 | fraction of contestable pot after calling | Inherited reasoning default |
| `preflop_reraise_multiple` | 3 | multiple of the current street bet | Inherited reasoning default |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| alan_wake_claude | 0 | 64 × 250 | +134.22 | 11.34 |
| jack_joyce_gemini | 0 | 64 × 250 | +310.12 | 19.59 |
| jesse_faden_glm | 0 | 64 × 250 | +312.31 | 18.11 |
| max_payne_gpt | 0 | 64 × 250 | +58.04 | 6.89 |
| v4:Bot | 1 | 64 × 250 | +2.86 | 5.78 |
| alan_wake_claude | 1 | 128 × 250 | +128.38 | 7.92 |
| jack_joyce_gemini | 1 | 128 × 250 | +285.74 | 11.45 |
| jesse_faden_glm | 1 | 128 × 250 | +292.43 | 12.86 |
| max_payne_gpt | 1 | 128 × 250 | +66.46 | 4.83 |
| v4:Bot | 2 | 128 × 250 | +7.40 | 4.82 |

Mean against finals, seed 0, 64 tables: **+203.67 bb/100**.

Mean against finals, seed 1, 128 tables: **+193.25 bb/100**.

**Verdict:** Retained as parent of v9. The development gain was small and did not establish a clear advantage; v9 applies the all-in correction before final evaluation.

## v6

**What it does:** Uses v4's value-only strategy, but removes the realization penalty if calling leaves no future betting. Keeps the unmatched-overbet correction and effective-stack sizing.

**Why it exists:** Do not discount showdown equity for future betting decisions that cannot occur after an all-in call.

**Parent:** 4. **Decision line count:** 53 including the `act` declaration (52 body lines); no extra decision helpers.

**Parameters:** All unchanged defaults are inherited reasoning choices. No grid search or automated fitting was performed.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo samples per decision | Inherited reasoning default |
| `button_open_equity` | 0.43 | raw equity probability | Inherited reasoning default |
| `isolation_equity` | 0.6 | raw equity probability | Inherited reasoning default |
| `preflop_value_equity` | 0.69 | adjusted equity probability | Inherited reasoning default |
| `postflop_value_equity` | 0.68 | adjusted equity probability | Inherited reasoning default |
| `reraise_margin_equity` | 0.12 | equity probability increment when facing a bet | Inherited reasoning default |
| `range_exclusion_fraction` | 0.3 | weak-range fraction removed per raise-equivalent | Inherited reasoning default |
| `prior_raise_weight` | 0.5 | current-street raise-equivalents per earlier raise | Inherited reasoning default |
| `call_margin_equity` | 0.03 | equity probability above break-even call odds | Inherited reasoning default |
| `realization_fraction` | 0.9 | fraction of adjusted equity before possible future betting | Inherited reasoning default |
| `opening_size_bb` | 2.5 | big blinds (street raise-to total) | Inherited reasoning default |
| `premium_equity` | 0.9 | adjusted equity probability | Inherited reasoning default |
| `premium_pot_fraction` | 1.25 | fraction of contestable pot after calling | Inherited reasoning default |
| `bet_pot_fraction` | 0.66 | fraction of contestable pot after calling | Inherited reasoning default |
| `preflop_reraise_multiple` | 3 | multiple of the current street bet | Inherited reasoning default |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| alan_wake_claude | 0 | 64 × 250 | +117.37 | 9.16 |
| jack_joyce_gemini | 0 | 64 × 250 | +292.38 | 16.92 |
| jesse_faden_glm | 0 | 64 × 250 | +278.48 | 16.98 |
| max_payne_gpt | 0 | 64 × 250 | +68.68 | 6.09 |
| random | 0 | 64 × 250 | +343.33 | 14.05 |
| v4:Bot | 1 | 64 × 250 | -5.37 | 6.44 |
| v1:Bot | 1000 | 128 × 250 | +63.63 | 5.86 |
| v2:Bot | 1000 | 128 × 250 | -10.28 | 4.60 |
| v3:Bot | 1000 | 128 × 250 | -13.48 | 4.60 |
| v4:Bot | 1000 | 128 × 250 | -13.48 | 4.60 |
| v5:Bot | 1000 | 128 × 250 | -2.80 | 4.55 |
| v6:Bot | 1000 | 128 × 250 | -13.48 | 4.60 |
| alan_wake_claude | 1000 | 512 × 250 | +118.67 | 4.16 |
| jack_joyce_gemini | 1000 | 512 × 250 | +296.40 | 6.39 |
| jesse_faden_glm | 1000 | 512 × 250 | +291.08 | 6.01 |
| max_payne_gpt | 1000 | 512 × 250 | +62.23 | 2.43 |
| random | 1000 | 512 × 250 | +340.26 | 4.57 |

Mean against finals, seed 0, 64 tables: **+189.23 bb/100**.

Mean against finals, seed 1000, 512 tables: **+192.10 bb/100**.

**Verdict:** Finalist; retained as parent of v7 and v8. The change is a strategic correction, with little measured development impact. Selected for deployment by the frozen held-out rule.

## v7

**What it does:** Gives opponent overbets additional range-strength weight according to the effective calling odds. All other decisions match v6.

**Why it exists:** Test generic bet-size information rather than treating a small raise and a large overbet as equally informative.

**Parent:** 6. **Decision line count:** 54 including the `act` declaration (53 body lines); no extra decision helpers.

**Parameters:** All unchanged defaults are inherited reasoning choices. No grid search or automated fitting was performed.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo samples per decision | Inherited reasoning default |
| `button_open_equity` | 0.43 | raw equity probability | Inherited reasoning default |
| `isolation_equity` | 0.6 | raw equity probability | Inherited reasoning default |
| `preflop_value_equity` | 0.69 | adjusted equity probability | Inherited reasoning default |
| `postflop_value_equity` | 0.68 | adjusted equity probability | Inherited reasoning default |
| `reraise_margin_equity` | 0.12 | equity probability increment when facing a bet | Inherited reasoning default |
| `range_exclusion_fraction` | 0.3 | weak-range fraction removed per raise-equivalent | Inherited reasoning default |
| `prior_raise_weight` | 0.5 | current-street raise-equivalents per earlier raise | Inherited reasoning default |
| `call_margin_equity` | 0.03 | equity probability above break-even call odds | Inherited reasoning default |
| `realization_fraction` | 0.9 | fraction of adjusted equity before possible future betting | Inherited reasoning default |
| `opening_size_bb` | 2.5 | big blinds (street raise-to total) | Inherited reasoning default |
| `pressure_reference_odds` | 0.333333 | call equity probability; 1/3 corresponds to a pot bet | Reasoning default tested in this version |
| `premium_equity` | 0.9 | adjusted equity probability | Inherited reasoning default |
| `premium_pot_fraction` | 1.25 | fraction of contestable pot after calling | Inherited reasoning default |
| `bet_pot_fraction` | 0.66 | fraction of contestable pot after calling | Inherited reasoning default |
| `preflop_reraise_multiple` | 3 | multiple of the current street bet | Inherited reasoning default |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| alan_wake_claude | 0 | 64 × 250 | +117.37 | 9.16 |
| jack_joyce_gemini | 0 | 64 × 250 | +292.38 | 16.92 |
| jesse_faden_glm | 0 | 64 × 250 | +278.48 | 16.98 |
| max_payne_gpt | 0 | 64 × 250 | +68.68 | 6.09 |
| random | 0 | 64 × 250 | +324.66 | 13.70 |
| v6:Bot | 1 | 64 × 250 | -3.31 | 6.40 |

Mean against finals, seed 0, 64 tables: **+189.23 bb/100**.

**Verdict:** Discarded. Its four seed-0 finals results exactly matched v6, and its parent match showed no advantage. The extra parameter had no demonstrated benefit.

## v8

**What it does:** Uses exactly v6's strategy with 256 Monte Carlo samples instead of 128. This consumes the full allowed sampling budget.

**Why it exists:** Test whether lower equity-estimation noise improves strength enough to justify twice the candidate sampling work.

**Parent:** 6. **Decision line count:** 53 including the `act` declaration (52 body lines); no extra decision helpers.

**Parameters:** All unchanged defaults are inherited reasoning choices. No grid search or automated fitting was performed.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 256 | Monte Carlo samples per decision | Compute-cap experiment against v6; not retained |
| `button_open_equity` | 0.43 | raw equity probability | Inherited reasoning default |
| `isolation_equity` | 0.6 | raw equity probability | Inherited reasoning default |
| `preflop_value_equity` | 0.69 | adjusted equity probability | Inherited reasoning default |
| `postflop_value_equity` | 0.68 | adjusted equity probability | Inherited reasoning default |
| `reraise_margin_equity` | 0.12 | equity probability increment when facing a bet | Inherited reasoning default |
| `range_exclusion_fraction` | 0.3 | weak-range fraction removed per raise-equivalent | Inherited reasoning default |
| `prior_raise_weight` | 0.5 | current-street raise-equivalents per earlier raise | Inherited reasoning default |
| `call_margin_equity` | 0.03 | equity probability above break-even call odds | Inherited reasoning default |
| `realization_fraction` | 0.9 | fraction of adjusted equity before possible future betting | Inherited reasoning default |
| `opening_size_bb` | 2.5 | big blinds (street raise-to total) | Inherited reasoning default |
| `premium_equity` | 0.9 | adjusted equity probability | Inherited reasoning default |
| `premium_pot_fraction` | 1.25 | fraction of contestable pot after calling | Inherited reasoning default |
| `bet_pot_fraction` | 0.66 | fraction of contestable pot after calling | Inherited reasoning default |
| `preflop_reraise_multiple` | 3 | multiple of the current street bet | Inherited reasoning default |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| alan_wake_claude | 0 | 64 × 250 | +109.58 | 11.05 |
| jack_joyce_gemini | 0 | 64 × 250 | +283.99 | 19.30 |
| jesse_faden_glm | 0 | 64 × 250 | +279.05 | 17.01 |
| max_payne_gpt | 0 | 64 × 250 | +55.27 | 6.39 |
| v6:Bot | 2 | 64 × 250 | +1.61 | 6.43 |

Mean against finals, seed 0, 64 tables: **+181.97 bb/100**.

**Verdict:** Discarded before held-out evaluation. The development mean did not improve, and the parent comparison was within error; 128 samples is the simpler compute choice.

## v9

**What it does:** Combines v5's positional bluffing with full equity realization when calling ends betting. Inlines the final legality conjunction to keep the decision method at 55 readable lines without changing that check.

**Why it exists:** Apply the separately tested all-in correction to the bluffing branch, then compare frozen candidates.

**Parent:** 5. **Decision line count:** 55 including the `act` declaration (54 body lines); no extra decision helpers.

**Parameters:** All unchanged defaults are inherited reasoning choices. No grid search or automated fitting was performed.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `num_samples` | 128 | Monte Carlo samples per decision | Inherited reasoning default |
| `button_open_equity` | 0.43 | raw equity probability | Inherited reasoning default |
| `isolation_equity` | 0.6 | raw equity probability | Inherited reasoning default |
| `preflop_value_equity` | 0.69 | adjusted equity probability | Inherited reasoning default |
| `postflop_value_equity` | 0.68 | adjusted equity probability | Inherited reasoning default |
| `reraise_margin_equity` | 0.12 | equity probability increment when facing a bet | Inherited reasoning default |
| `range_exclusion_fraction` | 0.3 | weak-range fraction removed per raise-equivalent | Inherited reasoning default |
| `prior_raise_weight` | 0.5 | current-street raise-equivalents per earlier raise | Inherited reasoning default |
| `call_margin_equity` | 0.03 | equity probability above break-even call odds | Inherited reasoning default |
| `realization_fraction` | 0.9 | fraction of adjusted equity before possible future betting | Inherited reasoning default |
| `opening_size_bb` | 2.5 | big blinds (street raise-to total) | Inherited reasoning default |
| `bluff_frequency` | 0.15 | probability per eligible positional bluff opportunity | Inherited reasoning default |
| `bluff_max_equity` | 0.35 | maximum raw equity probability eligible to bluff | Inherited reasoning default |
| `premium_equity` | 0.9 | adjusted equity probability | Inherited reasoning default |
| `premium_pot_fraction` | 1.25 | fraction of contestable pot after calling | Inherited reasoning default |
| `bet_pot_fraction` | 0.66 | fraction of contestable pot after calling | Inherited reasoning default |
| `preflop_reraise_multiple` | 3 | multiple of the current street bet | Inherited reasoning default |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| alan_wake_claude | 0 | 64 × 250 | +134.22 | 11.34 |
| jack_joyce_gemini | 0 | 64 × 250 | +309.52 | 19.57 |
| jesse_faden_glm | 0 | 64 × 250 | +314.18 | 17.97 |
| max_payne_gpt | 0 | 64 × 250 | +58.83 | 6.93 |
| v5:Bot | 2 | 64 × 250 | -3.75 | 6.41 |
| v1:Bot | 1000 | 128 × 250 | +79.33 | 7.19 |
| v2:Bot | 1000 | 128 × 250 | -5.36 | 4.30 |
| v3:Bot | 1000 | 128 × 250 | -7.03 | 4.83 |
| v4:Bot | 1000 | 128 × 250 | -7.03 | 4.83 |
| v5:Bot | 1000 | 128 × 250 | -4.71 | 4.69 |
| v6:Bot | 1000 | 128 × 250 | -7.03 | 4.83 |
| v7:Bot | 1000 | 128 × 250 | -6.67 | 4.83 |
| v8:Bot | 1000 | 128 × 250 | -0.48 | 4.09 |
| v9:Bot | 1000 | 128 × 250 | -4.58 | 4.63 |
| alan_wake_claude | 1000 | 512 × 250 | +117.33 | 4.35 |
| jack_joyce_gemini | 1000 | 512 × 250 | +290.39 | 6.23 |
| jesse_faden_glm | 1000 | 512 × 250 | +290.64 | 6.62 |
| max_payne_gpt | 1000 | 512 × 250 | +62.33 | 2.51 |
| random | 1000 | 512 × 250 | +335.32 | 4.57 |

Mean against finals, seed 0, 64 tables: **+204.19 bb/100**.

Mean against finals, seed 1000, 512 tables: **+190.17 bb/100**.

**Verdict:** Finalist; no descendant and no strategy changes after the evaluation lock. Not deployed under the frozen held-out rule.

## Final choice

**Deploy v6.** Held-out mean: **+192.10 bb/100** against the four finals.

The held-out means were v6 **+192.10** and v9 **+190.17** bb/100. Their difference (v9 − v6) was **-1.93**. The predeclared conservative two-SE difference threshold was **19.35 bb/100**.

The difference was within that uncertainty threshold, so the predeclared rule chose the value-only policy with two fewer parameters. V6 also had the higher held-out point estimate. V9 led the first-seed development results, but its apparent advantage did not repeat on held-out evaluation. No parameters were changed after seeing seed 1000.

### Deployed version: all held-out benchmarks

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| v1:Bot | 1000 | 128 × 250 | +63.63 | 5.86 |
| v2:Bot | 1000 | 128 × 250 | -10.28 | 4.60 |
| v3:Bot | 1000 | 128 × 250 | -13.48 | 4.60 |
| v4:Bot | 1000 | 128 × 250 | -13.48 | 4.60 |
| v5:Bot | 1000 | 128 × 250 | -2.80 | 4.55 |
| v6:Bot | 1000 | 128 × 250 | -13.48 | 4.60 |
| alan_wake_claude | 1000 | 512 × 250 | +118.67 | 4.16 |
| jack_joyce_gemini | 1000 | 512 × 250 | +296.40 | 6.39 |
| jesse_faden_glm | 1000 | 512 × 250 | +291.08 | 6.01 |
| max_payne_gpt | 1000 | 512 × 250 | +62.23 | 2.43 |
| random | 1000 | 512 × 250 | +340.26 | 4.57 |

### Self-play control for the parent-match warning

| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---:|---:|---:|---:|
| v6:Bot | 1000 | 128 × 250 | -13.48 | 4.60 |
| v9:Bot | 1000 | 128 × 250 | -4.58 | 4.63 |

The deployed policy scored -13.48 bb/100 (SE 4.60) against an identical copy on the same seed and table count. This control has zero true policy advantage by symmetry. Compare it with the parent result below: a similar negative value suggests shared-deal noise rather than evidence that the change caused a regression. These controls were added after noticing the repeated losses, are diagnostic only, and did not change the frozen selection rule. They are not used to subtract a correction from any published score.

Direct held-out result against parent v4: **-13.48 bb/100 (SE 4.60)**. Interpret small differences as noise; a positive finals score does not establish strength against unseen phase-2 bots.

### Known weaknesses

- The range correction treats raw showdown equity approximately like a hand-strength percentile; it is not a calibrated posterior over opponent holdings. It can misjudge shared-board ties, draws, blockers and unusual betting ranges.
- Preflop opens and re-raises use noisy random-hand equity rather than solved ranges. There is no opponent learning, no equilibrium solver and no action-history model beyond aggregate aggression.
- v6 has no deliberate postflop bluffs. Its sizing can reveal strength to an adaptive opponent; v9 adds only a small heuristic positional bluff component.
- The conservative range correction can overfold against frequent bluffers. Realization and betting thresholds are simple fixed heuristics, and prior raises are not distinguished by size or detailed sequence.
- Success against four phase-1 bots and close relatives is limited evidence about independent phase-2 opponents. All tests are heads-up; the three-seat contract smoke check is not strategic multiway validation.

## Verification and reproduction

The contract checker exercises every street under `jit` and `vmap`, scalar int32 action fields, legal raise bounds, free checks, disabled raising, unchanged state, JSON metadata and a non-hard-coded seat dimension. Versions v4 onward also pass the unmatched-overbet calling-invariance regression. The protected reference hashes match the originals, and all version hashes match the pre-held-out freeze. Simulation smoke tests and all logged benchmarks completed without an exception.

Decision-line counts include the one-line method declaration and exclude comments/blank lines. No version defines another decision helper. Numeric 0/1 identities, card/street identifiers and the 1e-6 division guard are structural or numerical safeguards rather than strategy knobs. Monte Carlo counts are 128 in all deployed candidates and 256 only in v8, with constructor validation rejecting counts outside 1–256.

Run from this directory using the existing environment:

```sh
uv run --offline --no-sync --cache-dir .uv-cache python bench.py v6:Bot finals --seed 0
uv run --offline --no-sync --cache-dir .uv-cache python check_contract.py v1 v2 v3 v4 v5 v6 v7 v8 v9
```

### Retained scratch artifacts

- `benchmark_*.log`: unedited development and held-out benchmark output; names encode candidate, opponent group, seed, and nondefault table count. Default runs use 64 × 250.
- `smoke_v1.log`: initial 2 × 20 import/compilation check, too small for any strength inference.
- `check_contract.py` and `contract_*.log`: reproducible contract checks and their outputs.
- `reference_hashes.json`, `version_hashes.json`, `evaluation_lock.json`: reference integrity, frozen source integrity and selection protocol.
- `write_readme.py`: generates this report from logs and each version metadata; it does not modify version code.
- `final_static_audit.log`: import, purity, single-class and frozen-source audit; a comment-punctuation false positive in the first checker was corrected without editing bot code.
- `experiment_summary.json`: machine-readable selected version, held-out scores, runtime and benchmark totals.
- `.uv-cache/` and `__pycache__/`: local runtime caches.

The source references `equity.py`, `bench.py` and `opponents/` were not edited. No repository commits were made.
