# faye_gpt — phase_5_cowboy_bebop/faye_gpt

- **Model:** GPT-6 (Codex); no more specific runtime model suffix is available to this session.
- **Prompt:** botgen-v5.
- **Engine contract:** decafb9, engine_sha256 `295477eff7637056`; hash verified by the supplied check.
- **Executed:** yes, on the existing project Python 3.13.12 / JAX 0.11.2 CPU environment, invoked through `uv run python` from this directory. No packages were installed; `.venv`, `uv.lock`, `bench.py` and `equity.py` were not edited.
- **Start:** 2026-10-14 05:36:28 UTC / 16:06:28 Australia/Adelaide, after the successful check.
- **Finish:** 2026-10-14T06:21:17+00:00. **Wall-clock used:** 44.82 minutes. A wrap-up reminder was scheduled; the iteration deadline was 06:26:28 UTC and the overall deadline 06:36:28 UTC.
- **Deployment:** `from .v10 import Bot as DeployedBot`.

The first check encountered a sandbox restriction on the existing uv cache. The identical command then succeeded with approved execution access:

```text
python      /Users/alik/Documents/Projects/poker_bots/poker_bots/.venv/bin/python3 (3.13.12)
jax         0.11.2 (cpu)
poker_bots  /Users/alik/Documents/Projects/poker_bots/poker_bots/src/poker_bots
engine      sha256 295477eff7637056 (matches the contract)
smoke test  random vs aggressor, 40 hands in 2s: OK
```

## Access and historical audit

Before reading the pasted directory-only restriction, the agent read the requester-provided attachment and issued `pwd` plus a search for AGENTS.md under the wider named project. The search returned no file paths or contents. No opponent source, engine source, other bot source, experiment results outside this directory, or git state was read by the agent. Subsequent reads and all authored project files stayed inside this directory; the supplied benchmark loaded the engine and black-box opponents as authorized. The initial cache error and exact audit are preserved in [run_notes.txt](run_notes.txt).

Versions were frozen after their first benchmark. **Historical parameter-exposure limitation:** v1–v3 left the 0.15 semibluff draw threshold, 0.40 semibluff equity threshold, and 0.50 pressure cap as literals. v4 exposed them without changing their values; v1–v3 remain unchanged for honest provenance and are not deployment candidates. The later versions expose the betting, calling, pressure and semibluff thresholds as constructor parameters. Structural card counts, rank definitions and poker arithmetic remain constants.

## Design and experimental protocol

The main lineage combines small value bets and steals with conservative calls against selected betting ranges. Fixed-shape float32 counters track how often the opponent raises preflop and postflop and folds to a postflop bet. The strategy adapts to those behaviors, never to identity. Explicit board and draw features alter equity margins, sizes, and bluff eligibility; no keys or hand histories are stored.

All performance results below use 250-hand sessions, balanced candidate seating, and the supplied table-clustered standard error. The required 2 × 20 smoke tests are execution checks only and are excluded from performance tables. Training used only seeds 0–9; seed 1000 is held out and used once per finalist for the field. A repeated result with identical candidate, opponent, seed and sample size is displayed once, never counted as an independent sample. Field scores use the exact rank/sign weighting printed by bench.py. No aggregate score SE is invented from opponent SEs.

The first field run's printed per-opponent durations sum to 201 seconds. Later concurrent runs shared CPU resources, so their timings are not single-bot speed comparisons.

### Completed field scores

| Candidate | Seed | Tables × 250 per opponent | Weighted score | Log |
|---|---:|---:|---:|---|
| v1 | 0 | 64 × 250 | -13.65 | [field_v1_seed0.log](field_v1_seed0.log) |
| v2 | 0 | 64 × 250 | +14.54 | [field_v2_seed0.log](field_v2_seed0.log) |
| v3 | 0 | 64 × 250 | +30.82 | [field_v3_seed0.log](field_v3_seed0.log) |
| v4 | 0 | 64 × 250 | +34.32 | [field_v4_seed0.log](field_v4_seed0.log) |
| v5 | 0 | 64 × 250 | +48.47 | [field_v5_seed0.log](field_v5_seed0.log) |
| v6 | 0 | 64 × 250 | +50.28 | [field_v6_seed0.log](field_v6_seed0.log) |
| v7 | 0 | 64 × 250 | +53.35 | [field_v7_seed0.log](field_v7_seed0.log) |
| v8 | 0 | 64 × 250 | +49.26 | [field_v8_seed0.log](field_v8_seed0.log) |
| v10 | 0 | 64 × 250 | +54.45 | [field_v10_seed0.log](field_v10_seed0.log) |
| v7 | 2 | 128 × 250 | +49.35 | [field_v7_seed2.log](field_v7_seed2.log) |
| ablation_v7 | 2 | 128 × 250 | +49.26 | [field_ablation_v7_seed2.log](field_ablation_v7_seed2.log) |
| v7 | 1000 | 128 × 250 | +50.06 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| v8 | 1000 | 128 × 250 | +45.46 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| v10 | 1000 | 128 × 250 | +51.86 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| ablation_v10 | 1000 | 128 × 250 | +49.18 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |

## v1: Baseline uniform equity

**What it does:** Uses 64 random-opponent showdown samples, price-based calls, value raises and occasional postflop bluffs. Explicit suit, rank, pairing and draw features adjust postflop calls, bluff eligibility and sizes.

**Why it exists:** Establish an executable baseline with board texture and draws.

**Parameters:** Values are defaults for this immutable version. This initial version has no parent; new reasoning choices were tested in the runs below, not fitted to held-out seed 1000.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `samples` | `64` | Monte Carlo deals/decision | Reasoned experiment/default; tested in listed training runs |
| `open_equity` | `0.48` | showdown pot share | Reasoned experiment/default; tested in listed training runs |
| `value_equity` | `0.7` | adjusted pot share | Reasoned experiment/default; tested in listed training runs |
| `reraise_equity` | `0.84` | adjusted pot share | Reasoned experiment/default; tested in listed training runs |
| `call_margin` | `0.08` | pot-share probability points | Reasoned experiment/default; tested in listed training runs |
| `pressure_penalty` | `0.16` | pot-share points/unit pressure | Reasoned experiment/default; tested in listed training runs |
| `open_size_bb` | `2.5` | big blinds, street raise-to total | Reasoned experiment/default; tested in listed training runs |
| `bet_pot_fraction` | `0.65` | fraction of pot after calling | Reasoned experiment/default; tested in listed training runs |
| `texture_weight` | `0.06` | equity points and extra pot fraction per normalized feature | Reasoned experiment/default; tested in listed training runs |
| `draw_weight` | `0.12` | equity points per next-card improvement probability | Reasoned experiment/default; tested in listed training runs |
| `bluff_frequency` | `0.12` | probability on eligible free actions | Reasoned experiment/default; tested in listed training runs |
| `features_enabled` | `True` | Boolean | Reasoned experiment/default; tested in listed training runs |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 0 | 64 × 250 | -117.94 | 8.62 | [field_v1_seed0.log](field_v1_seed0.log) |
| leviathan_claude | 0 | 64 × 250 | -108.33 | 9.28 | [field_v1_seed0.log](field_v1_seed0.log) |
| megalodon_glm | 0 | 64 × 250 | -121.15 | 11.17 | [field_v1_seed0.log](field_v1_seed0.log) |
| windrunner_glm | 0 | 64 × 250 | -82.54 | 11.77 | [field_v1_seed0.log](field_v1_seed0.log) |
| invoker_gpt | 0 | 64 × 250 | -55.01 | 10.77 | [field_v1_seed0.log](field_v1_seed0.log) |
| meepo_claude | 0 | 64 × 250 | -29.61 | 12.67 | [field_v1_seed0.log](field_v1_seed0.log) |
| hadrian_marlowe_gpt | 0 | 64 × 250 | -77.11 | 9.81 | [field_v1_seed0.log](field_v1_seed0.log) |
| valka_onderra_claude | 0 | 64 × 250 | -59.48 | 12.97 | [field_v1_seed0.log](field_v1_seed0.log) |
| aggressor | 0 | 64 × 250 | -20.95 | 9.73 | [field_v1_seed0.log](field_v1_seed0.log) |
| alan_wake_claude | 0 | 64 × 250 | +23.76 | 11.62 | [field_v1_seed0.log](field_v1_seed0.log) |
| bassander_lin_glm | 0 | 64 × 250 | -30.01 | 19.64 | [field_v1_seed0.log](field_v1_seed0.log) |
| max_payne_gpt | 0 | 64 × 250 | +30.29 | 10.39 | [field_v1_seed0.log](field_v1_seed0.log) |
| jesse_faden_glm | 0 | 64 × 250 | +148.29 | 23.74 | [field_v1_seed0.log](field_v1_seed0.log) |
| jack_joyce_gemini | 0 | 64 × 250 | +180.74 | 20.92 | [field_v1_seed0.log](field_v1_seed0.log) |
| random | 0 | 64 × 250 | +307.71 | 10.53 | [field_v1_seed0.log](field_v1_seed0.log) |

Weighted field score, seed 0, 64 × 250: **-13.65**.

**Verdict:** Kept as the initial baseline; its loose interpretation of random-hand equity lost heavily to strong ranges.

**Decision code:** 62 nonblank, noncomment lines in act, observe and defined helpers, including signatures and multiline returns; constructors, init_state and metadata excluded. The shorter early baselines did not use the entire allowed budget.

## v2: Range selection correction

**What it does:** Discounts random-hand equity by the estimated fraction of the opponent strength distribution still present. Betting and repeated raising imply narrower ranges than checking.

**Why it exists:** Test whether treating a bettor as a random hand caused the baseline losses.

**Parameters:** Values are defaults for this immutable version. “Inherited” means unchanged from parent v1; new reasoning choices were tested in the runs below, not fitted to held-out seed 1000.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `samples` | `64` | Monte Carlo deals/decision | Inherited |
| `open_equity` | `0.48` | showdown pot share | Inherited |
| `value_equity` | `0.7` | adjusted pot share | Inherited |
| `reraise_equity` | `0.84` | adjusted pot share | Inherited |
| `call_margin` | `0.08` | pot-share probability points | Inherited |
| `pressure_penalty` | `0.16` | pot-share points/unit pressure | Inherited |
| `open_size_bb` | `2.5` | big blinds, street raise-to total | Inherited |
| `bet_pot_fraction` | `0.65` | fraction of pot after calling | Inherited |
| `texture_weight` | `0.06` | equity points and extra pot fraction per normalized feature | Inherited |
| `draw_weight` | `0.12` | equity points per next-card improvement probability | Inherited |
| `bluff_frequency` | `0.12` | probability on eligible free actions | Inherited |
| `features_enabled` | `True` | Boolean | Inherited |
| `bet_range_fraction` | `0.5` | fraction of current strength distribution | Reasoned experiment/default; tested in listed training runs |
| `quiet_range_fraction` | `0.8` | fraction of current strength distribution | Reasoned experiment/default; tested in listed training runs |
| `reraise_range_fraction` | `0.25` | fraction of current strength distribution | Reasoned experiment/default; tested in listed training runs |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 0 | 64 × 250 | -89.96 | 6.15 | [field_v2_seed0.log](field_v2_seed0.log) |
| leviathan_claude | 0 | 64 × 250 | -67.73 | 7.01 | [field_v2_seed0.log](field_v2_seed0.log) |
| megalodon_glm | 0 | 64 × 250 | -61.90 | 7.52 | [field_v2_seed0.log](field_v2_seed0.log) |
| windrunner_glm | 0 | 64 × 250 | -41.83 | 6.31 | [field_v2_seed0.log](field_v2_seed0.log) |
| invoker_gpt | 0 | 64 × 250 | -7.29 | 7.16 | [field_v2_seed0.log](field_v2_seed0.log) |
| meepo_claude | 0 | 64 × 250 | +1.03 | 7.06 | [field_v2_seed0.log](field_v2_seed0.log) |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +12.11 | 6.32 | [field_v2_seed0.log](field_v2_seed0.log) |
| valka_onderra_claude | 0 | 64 × 250 | +51.55 | 7.68 | [field_v2_seed0.log](field_v2_seed0.log) |
| aggressor | 0 | 64 × 250 | -50.98 | 5.36 | [field_v2_seed0.log](field_v2_seed0.log) |
| alan_wake_claude | 0 | 64 × 250 | +103.23 | 6.66 | [field_v2_seed0.log](field_v2_seed0.log) |
| bassander_lin_glm | 0 | 64 × 250 | +82.05 | 10.21 | [field_v2_seed0.log](field_v2_seed0.log) |
| max_payne_gpt | 0 | 64 × 250 | +60.38 | 5.29 | [field_v2_seed0.log](field_v2_seed0.log) |
| jesse_faden_glm | 0 | 64 × 250 | +175.46 | 12.35 | [field_v2_seed0.log](field_v2_seed0.log) |
| jack_joyce_gemini | 0 | 64 × 250 | +167.01 | 11.71 | [field_v2_seed0.log](field_v2_seed0.log) |
| random | 0 | 64 × 250 | +154.61 | 8.42 | [field_v2_seed0.log](field_v2_seed0.log) |

Weighted field score, seed 0, 64 × 250: **+14.54**.

**Verdict:** Kept as the next parent: a broad field improvement, while still losing to strong aggressive opponents.

**Decision code:** 65 nonblank, noncomment lines in act, observe and defined helpers, including signatures and multiline returns; constructors, init_state and metadata excluded. The shorter early baselines did not use the entire allowed budget.

## v3: Postflop behavior tracking

**What it does:** Tracks postflop raise frequency and responses to our bets with fixed-shape counts. Bluffs more when observed folds exceed the immediate break-even rate and widens the estimated betting range against frequent raisers.

**Why it exists:** Recover pots surrendered by a rigid, conservative range model.

**Parameters:** Values are defaults for this immutable version. “Inherited” means unchanged from parent v2; new reasoning choices were tested in the runs below, not fitted to held-out seed 1000.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `prior_actions` | `12.0` | pseudo-count actions | Reasoned experiment/default; tested in listed training runs |
| `prior_fold_rate` | `0.45` | fold/response probability | Reasoned experiment/default; tested in listed training runs |
| `prior_raise_rate` | `0.3` | raise/action probability | Reasoned experiment/default; tested in listed training runs |
| `exploit_bluff_frequency` | `0.8` | bet probability when steal condition holds | Reasoned experiment/default; tested in listed training runs |
| `fold_profit_margin` | `0.04` | fold-probability points above break-even | Reasoned experiment/default; tested in listed training runs |
| `samples` | `64` | Monte Carlo deals/decision | Inherited |
| `open_equity` | `0.48` | showdown pot share | Inherited |
| `value_equity` | `0.7` | adjusted pot share | Inherited |
| `reraise_equity` | `0.84` | adjusted pot share | Inherited |
| `call_margin` | `0.08` | pot-share probability points | Inherited |
| `pressure_penalty` | `0.16` | pot-share points/unit pressure | Inherited |
| `open_size_bb` | `2.5` | big blinds, street raise-to total | Inherited |
| `bet_pot_fraction` | `0.65` | fraction of pot after calling | Inherited |
| `texture_weight` | `0.06` | equity points and extra pot fraction per normalized feature | Inherited |
| `draw_weight` | `0.12` | equity points per next-card improvement probability | Inherited |
| `bluff_frequency` | `0.12` | probability on eligible free actions | Inherited |
| `features_enabled` | `True` | Boolean | Inherited |
| `bet_range_fraction` | `0.5` | fraction of current strength distribution | Inherited |
| `quiet_range_fraction` | `0.8` | fraction of current strength distribution | Inherited |
| `reraise_range_fraction` | `0.25` | fraction of current strength distribution | Inherited |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 0 | 64 × 250 | -55.99 | 6.89 | [field_v3_seed0.log](field_v3_seed0.log) |
| leviathan_claude | 0 | 64 × 250 | +14.53 | 8.93 | [field_v3_seed0.log](field_v3_seed0.log) |
| megalodon_glm | 0 | 64 × 250 | +19.79 | 9.44 | [field_v3_seed0.log](field_v3_seed0.log) |
| windrunner_glm | 0 | 64 × 250 | +16.71 | 7.24 | [field_v3_seed0.log](field_v3_seed0.log) |
| invoker_gpt | 0 | 64 × 250 | -2.62 | 6.36 | [field_v3_seed0.log](field_v3_seed0.log) |
| meepo_claude | 0 | 64 × 250 | +18.86 | 6.03 | [field_v3_seed0.log](field_v3_seed0.log) |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +42.62 | 6.52 | [field_v3_seed0.log](field_v3_seed0.log) |
| valka_onderra_claude | 0 | 64 × 250 | +66.07 | 6.41 | [field_v3_seed0.log](field_v3_seed0.log) |
| aggressor | 0 | 64 × 250 | -10.20 | 6.79 | [field_v3_seed0.log](field_v3_seed0.log) |
| alan_wake_claude | 0 | 64 × 250 | +99.22 | 5.55 | [field_v3_seed0.log](field_v3_seed0.log) |
| bassander_lin_glm | 0 | 64 × 250 | +87.37 | 9.62 | [field_v3_seed0.log](field_v3_seed0.log) |
| max_payne_gpt | 0 | 64 × 250 | +43.81 | 5.91 | [field_v3_seed0.log](field_v3_seed0.log) |
| jesse_faden_glm | 0 | 64 × 250 | +146.04 | 12.02 | [field_v3_seed0.log](field_v3_seed0.log) |
| jack_joyce_gemini | 0 | 64 × 250 | +156.47 | 12.50 | [field_v3_seed0.log](field_v3_seed0.log) |
| random | 0 | 64 × 250 | +138.46 | 8.19 | [field_v3_seed0.log](field_v3_seed0.log) |
| v2:Bot | 1 | 64 × 250 | +45.70 | 5.17 | [parent_v3_v2_seed1.log](parent_v3_v2_seed1.log) |

Weighted field score, seed 0, 64 × 250: **+30.82**.

**Verdict:** Kept: substantial field improvement and a clear head-to-head win against v2.

**Decision code:** 84 nonblank, noncomment lines in act, observe and defined helpers, including signatures and multiline returns; constructors, init_state and metadata excluded. The shorter early baselines did not use the entire allowed budget.

## v4: Preflop behavior tracking

**What it does:** Adds preflop raise frequency to reduce the strength discount against frequent preflop raisers. It also exposes the previously fixed semibluff and pressure thresholds as constructor arguments, without changing their values.

**Why it exists:** Played-hand diagnostics showed large losses in preflop pots conceded without showdown.

**Parameters:** Values are defaults for this immutable version. “Inherited” means unchanged from parent v3; new reasoning choices were tested in the runs below, not fitted to held-out seed 1000.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `semibluff_draw_probability` | `0.15` | next-card improvement probability | Reasoned experiment/default; tested in listed training runs |
| `semibluff_equity` | `0.4` | random-hand pot share | Reasoned experiment/default; tested in listed training runs |
| `pressure_odds_cap` | `0.5` | pot-odds fraction | Reasoned experiment/default; tested in listed training runs |
| `prior_actions` | `12.0` | pseudo-count actions | Inherited |
| `prior_fold_rate` | `0.45` | fold/response probability | Inherited |
| `prior_raise_rate` | `0.3` | raise/action probability | Inherited |
| `exploit_bluff_frequency` | `0.8` | bet probability when steal condition holds | Inherited |
| `fold_profit_margin` | `0.04` | fold-probability points above break-even | Inherited |
| `samples` | `64` | Monte Carlo deals/decision | Inherited |
| `open_equity` | `0.48` | showdown pot share | Inherited |
| `value_equity` | `0.7` | adjusted pot share | Inherited |
| `reraise_equity` | `0.84` | adjusted pot share | Inherited |
| `call_margin` | `0.08` | pot-share probability points | Inherited |
| `pressure_penalty` | `0.16` | pot-share points/unit pressure | Inherited |
| `open_size_bb` | `2.5` | big blinds, street raise-to total | Inherited |
| `bet_pot_fraction` | `0.65` | fraction of pot after calling | Inherited |
| `texture_weight` | `0.06` | equity points and extra pot fraction per normalized feature | Inherited |
| `draw_weight` | `0.12` | equity points per next-card improvement probability | Inherited |
| `bluff_frequency` | `0.12` | probability on eligible free actions | Inherited |
| `features_enabled` | `True` | Boolean | Inherited |
| `bet_range_fraction` | `0.5` | fraction of current strength distribution | Inherited |
| `quiet_range_fraction` | `0.8` | fraction of current strength distribution | Inherited |
| `reraise_range_fraction` | `0.25` | fraction of current strength distribution | Inherited |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 0 | 64 × 250 | -31.02 | 11.13 | [field_v4_seed0.log](field_v4_seed0.log) |
| leviathan_claude | 0 | 64 × 250 | +12.79 | 11.62 | [field_v4_seed0.log](field_v4_seed0.log) |
| megalodon_glm | 0 | 64 × 250 | +16.78 | 11.59 | [field_v4_seed0.log](field_v4_seed0.log) |
| windrunner_glm | 0 | 64 × 250 | +42.95 | 7.03 | [field_v4_seed0.log](field_v4_seed0.log) |
| invoker_gpt | 0 | 64 × 250 | +12.16 | 8.18 | [field_v4_seed0.log](field_v4_seed0.log) |
| meepo_claude | 0 | 64 × 250 | +22.44 | 8.38 | [field_v4_seed0.log](field_v4_seed0.log) |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +27.99 | 6.89 | [field_v4_seed0.log](field_v4_seed0.log) |
| valka_onderra_claude | 0 | 64 × 250 | +63.57 | 7.18 | [field_v4_seed0.log](field_v4_seed0.log) |
| aggressor | 0 | 64 × 250 | -14.05 | 7.91 | [field_v4_seed0.log](field_v4_seed0.log) |
| alan_wake_claude | 0 | 64 × 250 | +98.99 | 5.60 | [field_v4_seed0.log](field_v4_seed0.log) |
| bassander_lin_glm | 0 | 64 × 250 | +102.06 | 8.87 | [field_v4_seed0.log](field_v4_seed0.log) |
| max_payne_gpt | 0 | 64 × 250 | +42.24 | 5.89 | [field_v4_seed0.log](field_v4_seed0.log) |
| jesse_faden_glm | 0 | 64 × 250 | +162.19 | 11.91 | [field_v4_seed0.log](field_v4_seed0.log) |
| jack_joyce_gemini | 0 | 64 × 250 | +164.53 | 13.73 | [field_v4_seed0.log](field_v4_seed0.log) |
| random | 0 | 64 × 250 | +155.77 | 8.75 | [field_v4_seed0.log](field_v4_seed0.log) |
| v3:Bot | 1 | 64 × 250 | -8.78 | 7.13 | [parent_v4_v3_seed1.log](parent_v4_v3_seed1.log) |

Weighted field score, seed 0, 64 × 250: **+34.32**.

**Verdict:** Retained for subsequent sizing experiments; the parent match alone did not establish an improvement.

**Decision code:** 89 nonblank, noncomment lines in act, observe and defined helpers, including signatures and multiline returns; constructors, init_state and metadata excluded. The shorter early baselines did not use the entire allowed budget.

## v5: Smaller postflop bets

**What it does:** Uses 0.40-pot base bets instead of 0.65. Everything else matches v4, so steals need fewer folds and cost less when unsuccessful.

**Why it exists:** Test a materially different sizing rule after the diagnostics showed non-showdown losses.

**Parameters:** Values are defaults for this immutable version. “Inherited” means unchanged from parent v4; new reasoning choices were tested in the runs below, not fitted to held-out seed 1000.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `semibluff_draw_probability` | `0.15` | next-card improvement probability | Inherited |
| `semibluff_equity` | `0.4` | random-hand pot share | Inherited |
| `pressure_odds_cap` | `0.5` | pot-odds fraction | Inherited |
| `prior_actions` | `12.0` | pseudo-count actions | Inherited |
| `prior_fold_rate` | `0.45` | fold/response probability | Inherited |
| `prior_raise_rate` | `0.3` | raise/action probability | Inherited |
| `exploit_bluff_frequency` | `0.8` | bet probability when steal condition holds | Inherited |
| `fold_profit_margin` | `0.04` | fold-probability points above break-even | Inherited |
| `samples` | `64` | Monte Carlo deals/decision | Inherited |
| `open_equity` | `0.48` | showdown pot share | Inherited |
| `value_equity` | `0.7` | adjusted pot share | Inherited |
| `reraise_equity` | `0.84` | adjusted pot share | Inherited |
| `call_margin` | `0.08` | pot-share probability points | Inherited |
| `pressure_penalty` | `0.16` | pot-share points/unit pressure | Inherited |
| `open_size_bb` | `2.5` | big blinds, street raise-to total | Inherited |
| `bet_pot_fraction` | `0.4` | fraction of pot after calling | Reasoned experiment/default; tested in listed training runs |
| `texture_weight` | `0.06` | equity points and extra pot fraction per normalized feature | Inherited |
| `draw_weight` | `0.12` | equity points per next-card improvement probability | Inherited |
| `bluff_frequency` | `0.12` | probability on eligible free actions | Inherited |
| `features_enabled` | `True` | Boolean | Inherited |
| `bet_range_fraction` | `0.5` | fraction of current strength distribution | Inherited |
| `quiet_range_fraction` | `0.8` | fraction of current strength distribution | Inherited |
| `reraise_range_fraction` | `0.25` | fraction of current strength distribution | Inherited |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 0 | 64 × 250 | +9.41 | 8.67 | [field_v5_seed0.log](field_v5_seed0.log) |
| leviathan_claude | 0 | 64 × 250 | +61.53 | 7.15 | [field_v5_seed0.log](field_v5_seed0.log) |
| megalodon_glm | 0 | 64 × 250 | +48.57 | 8.55 | [field_v5_seed0.log](field_v5_seed0.log) |
| windrunner_glm | 0 | 64 × 250 | +57.63 | 7.21 | [field_v5_seed0.log](field_v5_seed0.log) |
| invoker_gpt | 0 | 64 × 250 | +42.98 | 6.37 | [field_v5_seed0.log](field_v5_seed0.log) |
| meepo_claude | 0 | 64 × 250 | +33.88 | 7.70 | [field_v5_seed0.log](field_v5_seed0.log) |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +50.97 | 4.60 | [field_v5_seed0.log](field_v5_seed0.log) |
| valka_onderra_claude | 0 | 64 × 250 | +42.68 | 6.52 | [field_v5_seed0.log](field_v5_seed0.log) |
| aggressor | 0 | 64 × 250 | +18.44 | 4.71 | [field_v5_seed0.log](field_v5_seed0.log) |
| alan_wake_claude | 0 | 64 × 250 | +70.65 | 4.45 | [field_v5_seed0.log](field_v5_seed0.log) |
| bassander_lin_glm | 0 | 64 × 250 | +78.21 | 8.78 | [field_v5_seed0.log](field_v5_seed0.log) |
| max_payne_gpt | 0 | 64 × 250 | +48.15 | 4.99 | [field_v5_seed0.log](field_v5_seed0.log) |
| jesse_faden_glm | 0 | 64 × 250 | +122.19 | 9.93 | [field_v5_seed0.log](field_v5_seed0.log) |
| jack_joyce_gemini | 0 | 64 × 250 | +135.28 | 13.14 | [field_v5_seed0.log](field_v5_seed0.log) |
| random | 0 | 64 × 250 | +104.93 | 6.80 | [field_v5_seed0.log](field_v5_seed0.log) |
| v4:Bot | 1 | 64 × 250 | +13.60 | 6.24 | [parent_v5_v4_seed1.log](parent_v5_v4_seed1.log) |

Weighted field score, seed 0, 64 × 250: **+48.47**.

**Verdict:** Kept: broad gains against the strong field and a positive parent match.

**Decision code:** 89 nonblank, noncomment lines in act, observe and defined helpers, including signatures and multiline returns; constructors, init_state and metadata excluded. The shorter early baselines did not use the entire allowed budget.

## v6: Street-appropriate fold statistics

**What it does:** Counts only postflop responses in the fold statistic used for postflop bluffs. Preflop calls no longer suppress the postflop steal estimate.

**Why it exists:** Condition the statistic on the situation in which it is used.

**Parameters:** Values are defaults for this immutable version. “Inherited” means unchanged from parent v5; new reasoning choices were tested in the runs below, not fitted to held-out seed 1000.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `semibluff_draw_probability` | `0.15` | next-card improvement probability | Inherited |
| `semibluff_equity` | `0.4` | random-hand pot share | Inherited |
| `pressure_odds_cap` | `0.5` | pot-odds fraction | Inherited |
| `prior_actions` | `12.0` | pseudo-count actions | Inherited |
| `prior_fold_rate` | `0.45` | fold/response probability | Inherited |
| `prior_raise_rate` | `0.3` | raise/action probability | Inherited |
| `exploit_bluff_frequency` | `0.8` | bet probability when steal condition holds | Inherited |
| `fold_profit_margin` | `0.04` | fold-probability points above break-even | Inherited |
| `samples` | `64` | Monte Carlo deals/decision | Inherited |
| `open_equity` | `0.48` | showdown pot share | Inherited |
| `value_equity` | `0.7` | adjusted pot share | Inherited |
| `reraise_equity` | `0.84` | adjusted pot share | Inherited |
| `call_margin` | `0.08` | pot-share probability points | Inherited |
| `pressure_penalty` | `0.16` | pot-share points/unit pressure | Inherited |
| `open_size_bb` | `2.5` | big blinds, street raise-to total | Inherited |
| `bet_pot_fraction` | `0.4` | fraction of pot after calling | Inherited |
| `texture_weight` | `0.06` | equity points and extra pot fraction per normalized feature | Inherited |
| `draw_weight` | `0.12` | equity points per next-card improvement probability | Inherited |
| `bluff_frequency` | `0.12` | probability on eligible free actions | Inherited |
| `features_enabled` | `True` | Boolean | Inherited |
| `bet_range_fraction` | `0.5` | fraction of current strength distribution | Inherited |
| `quiet_range_fraction` | `0.8` | fraction of current strength distribution | Inherited |
| `reraise_range_fraction` | `0.25` | fraction of current strength distribution | Inherited |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 0 | 64 × 250 | +5.88 | 8.25 | [field_v6_seed0.log](field_v6_seed0.log) |
| leviathan_claude | 0 | 64 × 250 | +64.21 | 6.72 | [field_v6_seed0.log](field_v6_seed0.log) |
| megalodon_glm | 0 | 64 × 250 | +49.90 | 8.22 | [field_v6_seed0.log](field_v6_seed0.log) |
| windrunner_glm | 0 | 64 × 250 | +58.67 | 6.89 | [field_v6_seed0.log](field_v6_seed0.log) |
| invoker_gpt | 0 | 64 × 250 | +40.61 | 6.22 | [field_v6_seed0.log](field_v6_seed0.log) |
| meepo_claude | 0 | 64 × 250 | +51.17 | 6.68 | [field_v6_seed0.log](field_v6_seed0.log) |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +50.81 | 4.60 | [field_v6_seed0.log](field_v6_seed0.log) |
| valka_onderra_claude | 0 | 64 × 250 | +54.94 | 7.25 | [field_v6_seed0.log](field_v6_seed0.log) |
| aggressor | 0 | 64 × 250 | +18.45 | 4.41 | [field_v6_seed0.log](field_v6_seed0.log) |
| alan_wake_claude | 0 | 64 × 250 | +64.26 | 4.36 | [field_v6_seed0.log](field_v6_seed0.log) |
| bassander_lin_glm | 0 | 64 × 250 | +82.16 | 9.49 | [field_v6_seed0.log](field_v6_seed0.log) |
| max_payne_gpt | 0 | 64 × 250 | +48.38 | 5.21 | [field_v6_seed0.log](field_v6_seed0.log) |
| jesse_faden_glm | 0 | 64 × 250 | +122.82 | 9.78 | [field_v6_seed0.log](field_v6_seed0.log) |
| jack_joyce_gemini | 0 | 64 × 250 | +123.07 | 11.99 | [field_v6_seed0.log](field_v6_seed0.log) |
| random | 0 | 64 × 250 | +104.36 | 6.65 | [field_v6_seed0.log](field_v6_seed0.log) |

Weighted field score, seed 0, 64 × 250: **+50.28**.

**Verdict:** Kept for statistical relevance; the modest aggregate score change is not strong evidence on its own.

**Decision code:** 89 nonblank, noncomment lines in act, observe and defined helpers, including signatures and multiline returns; constructors, init_state and metadata excluded. The shorter early baselines did not use the entire allowed budget.

## v7: Value three-bets

**What it does:** Uses the ordinary value threshold against a first preflop raise, reserving the strict reraise threshold for an already-reraised pot. The postflop strategy is otherwise unchanged.

**Why it exists:** Avoid treating an ordinary preflop open as though it were a postflop value reraise.

**Parameters:** Values are defaults for this immutable version. “Inherited” means unchanged from parent v6; new reasoning choices were tested in the runs below, not fitted to held-out seed 1000.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `semibluff_draw_probability` | `0.15` | next-card improvement probability | Inherited |
| `semibluff_equity` | `0.4` | random-hand pot share | Inherited |
| `pressure_odds_cap` | `0.5` | pot-odds fraction | Inherited |
| `prior_actions` | `12.0` | pseudo-count actions | Inherited |
| `prior_fold_rate` | `0.45` | fold/response probability | Inherited |
| `prior_raise_rate` | `0.3` | raise/action probability | Inherited |
| `exploit_bluff_frequency` | `0.8` | bet probability when steal condition holds | Inherited |
| `fold_profit_margin` | `0.04` | fold-probability points above break-even | Inherited |
| `samples` | `64` | Monte Carlo deals/decision | Inherited |
| `open_equity` | `0.48` | showdown pot share | Inherited |
| `value_equity` | `0.7` | adjusted pot share | Inherited |
| `reraise_equity` | `0.84` | adjusted pot share | Inherited |
| `call_margin` | `0.08` | pot-share probability points | Inherited |
| `pressure_penalty` | `0.16` | pot-share points/unit pressure | Inherited |
| `open_size_bb` | `2.5` | big blinds, street raise-to total | Inherited |
| `bet_pot_fraction` | `0.4` | fraction of pot after calling | Inherited |
| `texture_weight` | `0.06` | equity points and extra pot fraction per normalized feature | Inherited |
| `draw_weight` | `0.12` | equity points per next-card improvement probability | Inherited |
| `bluff_frequency` | `0.12` | probability on eligible free actions | Inherited |
| `features_enabled` | `True` | Boolean | Inherited |
| `bet_range_fraction` | `0.5` | fraction of current strength distribution | Inherited |
| `quiet_range_fraction` | `0.8` | fraction of current strength distribution | Inherited |
| `reraise_range_fraction` | `0.25` | fraction of current strength distribution | Inherited |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 0 | 64 × 250 | +6.48 | 8.26 | [field_v7_seed0.log](field_v7_seed0.log) |
| leviathan_claude | 0 | 64 × 250 | +66.94 | 7.01 | [field_v7_seed0.log](field_v7_seed0.log) |
| megalodon_glm | 0 | 64 × 250 | +57.00 | 8.57 | [field_v7_seed0.log](field_v7_seed0.log) |
| windrunner_glm | 0 | 64 × 250 | +68.24 | 6.17 | [field_v7_seed0.log](field_v7_seed0.log) |
| invoker_gpt | 0 | 64 × 250 | +38.63 | 5.83 | [field_v7_seed0.log](field_v7_seed0.log) |
| meepo_claude | 0 | 64 × 250 | +53.17 | 5.95 | [field_v7_seed0.log](field_v7_seed0.log) |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +52.52 | 4.26 | [field_v7_seed0.log](field_v7_seed0.log) |
| valka_onderra_claude | 0 | 64 × 250 | +57.62 | 7.37 | [field_v7_seed0.log](field_v7_seed0.log) |
| aggressor | 0 | 64 × 250 | +19.79 | 5.60 | [field_v7_seed0.log](field_v7_seed0.log) |
| alan_wake_claude | 0 | 64 × 250 | +62.74 | 4.54 | [field_v7_seed0.log](field_v7_seed0.log) |
| bassander_lin_glm | 0 | 64 × 250 | +98.28 | 9.06 | [field_v7_seed0.log](field_v7_seed0.log) |
| max_payne_gpt | 0 | 64 × 250 | +47.65 | 5.20 | [field_v7_seed0.log](field_v7_seed0.log) |
| jesse_faden_glm | 0 | 64 × 250 | +119.01 | 9.86 | [field_v7_seed0.log](field_v7_seed0.log) |
| jack_joyce_gemini | 0 | 64 × 250 | +128.08 | 11.99 | [field_v7_seed0.log](field_v7_seed0.log) |
| random | 0 | 64 × 250 | +107.15 | 6.80 | [field_v7_seed0.log](field_v7_seed0.log) |
| kraken_gpt | 2 | 128 × 250 | -5.39 | 5.97 | [targeted_v7_kraken_seed2.log](targeted_v7_kraken_seed2.log) |
| leviathan_claude | 2 | 128 × 250 | +53.51 | 5.76 | [field_v7_seed2.log](field_v7_seed2.log) |
| megalodon_glm | 2 | 128 × 250 | +54.45 | 6.49 | [field_v7_seed2.log](field_v7_seed2.log) |
| windrunner_glm | 2 | 128 × 250 | +71.43 | 5.54 | [field_v7_seed2.log](field_v7_seed2.log) |
| invoker_gpt | 2 | 128 × 250 | +35.73 | 4.92 | [field_v7_seed2.log](field_v7_seed2.log) |
| meepo_claude | 2 | 128 × 250 | +36.63 | 4.73 | [field_v7_seed2.log](field_v7_seed2.log) |
| hadrian_marlowe_gpt | 2 | 128 × 250 | +53.42 | 3.14 | [field_v7_seed2.log](field_v7_seed2.log) |
| valka_onderra_claude | 2 | 128 × 250 | +46.90 | 5.35 | [field_v7_seed2.log](field_v7_seed2.log) |
| aggressor | 2 | 128 × 250 | +26.48 | 3.73 | [field_v7_seed2.log](field_v7_seed2.log) |
| alan_wake_claude | 2 | 128 × 250 | +73.98 | 4.06 | [field_v7_seed2.log](field_v7_seed2.log) |
| bassander_lin_glm | 2 | 128 × 250 | +83.97 | 6.95 | [field_v7_seed2.log](field_v7_seed2.log) |
| max_payne_gpt | 2 | 128 × 250 | +54.68 | 3.37 | [field_v7_seed2.log](field_v7_seed2.log) |
| jesse_faden_glm | 2 | 128 × 250 | +131.56 | 7.68 | [field_v7_seed2.log](field_v7_seed2.log) |
| jack_joyce_gemini | 2 | 128 × 250 | +114.94 | 7.30 | [field_v7_seed2.log](field_v7_seed2.log) |
| random | 2 | 128 × 250 | +105.83 | 4.93 | [field_v7_seed2.log](field_v7_seed2.log) |
| ablation_v7:Bot | 2 | 128 × 250 | +0.80 | 3.80 | [ablation_match_v7_seed2.log](ablation_match_v7_seed2.log) |
| v6:Bot | 2 | 128 × 250 | +2.41 | 3.98 | [parent_v7_v6_seed2.log](parent_v7_v6_seed2.log) |
| kraken_gpt | 3 | 64 × 250 | -6.12 | 7.71 | [targeted_v7_kraken_seed3.log](targeted_v7_kraken_seed3.log) |
| kraken_gpt | 1000 | 128 × 250 | -3.13 | 5.75 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| leviathan_claude | 1000 | 128 × 250 | +54.65 | 5.54 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| megalodon_glm | 1000 | 128 × 250 | +54.67 | 7.08 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| windrunner_glm | 1000 | 128 × 250 | +68.37 | 5.30 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| invoker_gpt | 1000 | 128 × 250 | +37.55 | 4.51 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| meepo_claude | 1000 | 128 × 250 | +37.45 | 4.41 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| hadrian_marlowe_gpt | 1000 | 128 × 250 | +55.56 | 3.29 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| valka_onderra_claude | 1000 | 128 × 250 | +47.75 | 5.54 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| aggressor | 1000 | 128 × 250 | +21.73 | 4.19 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| alan_wake_claude | 1000 | 128 × 250 | +74.35 | 3.61 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| bassander_lin_glm | 1000 | 128 × 250 | +90.61 | 6.22 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| max_payne_gpt | 1000 | 128 × 250 | +59.29 | 4.32 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| jesse_faden_glm | 1000 | 128 × 250 | +126.86 | 7.10 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| jack_joyce_gemini | 1000 | 128 × 250 | +129.83 | 8.25 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| random | 1000 | 128 × 250 | +116.34 | 4.81 | [field_v7_seed1000.log](field_v7_seed1000.log) |
| v10:Bot | 1000 | 128 × 250 | +3.31 | 4.37 | [parent_v7_v10_seed1000.log](parent_v7_v10_seed1000.log) |

Weighted field score, seed 0, 64 × 250: **+53.35**.

Weighted field score, seed 1000, 128 × 250: **+50.06**.

Weighted field score, seed 2, 128 × 250: **+49.35**.

**Verdict:** Kept as the parent of v8, v9, v10 and v11, and as a finalist. Not deployed: v10 had the higher held-out weighted score and the corrected private-card draw model.

**Decision code:** 90 nonblank, noncomment lines in act, observe and defined helpers, including signatures and multiline returns; constructors, init_state and metadata excluded. The shorter early baselines did not use the entire allowed budget.

## v8: Quarter-pot sizing

**What it does:** Reduces the postflop base fraction from 0.40 to 0.25, with the same texture increment. This tests whether further reducing the price of a steal improves the full field result.

**Why it exists:** Test the small-bet hypothesis at a distinct second size.

**Parameters:** Values are defaults for this immutable version. “Inherited” means unchanged from parent v7; new reasoning choices were tested in the runs below, not fitted to held-out seed 1000.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `semibluff_draw_probability` | `0.15` | next-card improvement probability | Inherited |
| `semibluff_equity` | `0.4` | random-hand pot share | Inherited |
| `pressure_odds_cap` | `0.5` | pot-odds fraction | Inherited |
| `prior_actions` | `12.0` | pseudo-count actions | Inherited |
| `prior_fold_rate` | `0.45` | fold/response probability | Inherited |
| `prior_raise_rate` | `0.3` | raise/action probability | Inherited |
| `exploit_bluff_frequency` | `0.8` | bet probability when steal condition holds | Inherited |
| `fold_profit_margin` | `0.04` | fold-probability points above break-even | Inherited |
| `samples` | `64` | Monte Carlo deals/decision | Inherited |
| `open_equity` | `0.48` | showdown pot share | Inherited |
| `value_equity` | `0.7` | adjusted pot share | Inherited |
| `reraise_equity` | `0.84` | adjusted pot share | Inherited |
| `call_margin` | `0.08` | pot-share probability points | Inherited |
| `pressure_penalty` | `0.16` | pot-share points/unit pressure | Inherited |
| `open_size_bb` | `2.5` | big blinds, street raise-to total | Inherited |
| `bet_pot_fraction` | `0.25` | fraction of pot after calling | Reasoned experiment/default; tested in listed training runs |
| `texture_weight` | `0.06` | equity points and extra pot fraction per normalized feature | Inherited |
| `draw_weight` | `0.12` | equity points per next-card improvement probability | Inherited |
| `bluff_frequency` | `0.12` | probability on eligible free actions | Inherited |
| `features_enabled` | `True` | Boolean | Inherited |
| `bet_range_fraction` | `0.5` | fraction of current strength distribution | Inherited |
| `quiet_range_fraction` | `0.8` | fraction of current strength distribution | Inherited |
| `reraise_range_fraction` | `0.25` | fraction of current strength distribution | Inherited |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 0 | 64 × 250 | +11.68 | 7.19 | [field_v8_seed0.log](field_v8_seed0.log) |
| leviathan_claude | 0 | 64 × 250 | +53.65 | 8.57 | [field_v8_seed0.log](field_v8_seed0.log) |
| megalodon_glm | 0 | 64 × 250 | +60.99 | 8.90 | [field_v8_seed0.log](field_v8_seed0.log) |
| windrunner_glm | 0 | 64 × 250 | +56.59 | 7.11 | [field_v8_seed0.log](field_v8_seed0.log) |
| invoker_gpt | 0 | 64 × 250 | +47.12 | 5.58 | [field_v8_seed0.log](field_v8_seed0.log) |
| meepo_claude | 0 | 64 × 250 | +43.26 | 5.22 | [field_v8_seed0.log](field_v8_seed0.log) |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +54.04 | 3.04 | [field_v8_seed0.log](field_v8_seed0.log) |
| valka_onderra_claude | 0 | 64 × 250 | +42.45 | 5.79 | [field_v8_seed0.log](field_v8_seed0.log) |
| aggressor | 0 | 64 × 250 | +36.45 | 4.64 | [field_v8_seed0.log](field_v8_seed0.log) |
| alan_wake_claude | 0 | 64 × 250 | +48.35 | 3.93 | [field_v8_seed0.log](field_v8_seed0.log) |
| bassander_lin_glm | 0 | 64 × 250 | +76.48 | 7.68 | [field_v8_seed0.log](field_v8_seed0.log) |
| max_payne_gpt | 0 | 64 × 250 | +47.98 | 4.51 | [field_v8_seed0.log](field_v8_seed0.log) |
| jesse_faden_glm | 0 | 64 × 250 | +109.93 | 10.18 | [field_v8_seed0.log](field_v8_seed0.log) |
| jack_joyce_gemini | 0 | 64 × 250 | +103.09 | 13.00 | [field_v8_seed0.log](field_v8_seed0.log) |
| random | 0 | 64 × 250 | +79.80 | 7.05 | [field_v8_seed0.log](field_v8_seed0.log) |
| v7:Bot | 2 | 128 × 250 | +9.85 | 3.44 | [parent_v8_v7_seed2.log](parent_v8_v7_seed2.log) |
| kraken_gpt | 1000 | 128 × 250 | +5.62 | 5.08 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| leviathan_claude | 1000 | 128 × 250 | +56.68 | 5.83 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| megalodon_glm | 1000 | 128 × 250 | +50.59 | 5.84 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| windrunner_glm | 1000 | 128 × 250 | +55.82 | 5.71 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| invoker_gpt | 1000 | 128 × 250 | +38.23 | 3.16 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| meepo_claude | 1000 | 128 × 250 | +32.32 | 4.11 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| hadrian_marlowe_gpt | 1000 | 128 × 250 | +54.13 | 2.70 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| valka_onderra_claude | 1000 | 128 × 250 | +39.38 | 4.45 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| aggressor | 1000 | 128 × 250 | +35.95 | 3.35 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| alan_wake_claude | 1000 | 128 × 250 | +50.55 | 3.27 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| bassander_lin_glm | 1000 | 128 × 250 | +66.94 | 5.88 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| max_payne_gpt | 1000 | 128 × 250 | +52.65 | 3.97 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| jesse_faden_glm | 1000 | 128 × 250 | +105.89 | 6.41 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| jack_joyce_gemini | 1000 | 128 × 250 | +100.91 | 6.96 | [field_v8_seed1000.log](field_v8_seed1000.log) |
| random | 1000 | 128 × 250 | +84.29 | 4.74 | [field_v8_seed1000.log](field_v8_seed1000.log) |

Weighted field score, seed 0, 64 × 250: **+49.26**.

Weighted field score, seed 1000, 128 × 250: **+45.46**.

**Verdict:** A seed-2 parent match won 9.85 bb/100 (SE 3.44), so this frozen version was included in held-out evaluation. Discarded for deployment after its held-out weighted score of +45.46 fell below v7 and v10.

**Decision code:** 90 nonblank, noncomment lines in act, observe and defined helpers, including signatures and multiline returns; constructors, init_state and metadata excluded. The shorter early baselines did not use the entire allowed budget.

## v9: Adaptive repeated-raise ranges

**What it does:** Widens the repeated-raise range when the opponent raises postflop frequently. Other decisions match v7.

**Why it exists:** Test whether a fixed narrow range was causing excessive folds against frequent reraisers.

**Parameters:** Values are defaults for this immutable version. “Inherited” means unchanged from parent v7; new reasoning choices were tested in the runs below, not fitted to held-out seed 1000.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `semibluff_draw_probability` | `0.15` | next-card improvement probability | Inherited |
| `semibluff_equity` | `0.4` | random-hand pot share | Inherited |
| `pressure_odds_cap` | `0.5` | pot-odds fraction | Inherited |
| `prior_actions` | `12.0` | pseudo-count actions | Inherited |
| `prior_fold_rate` | `0.45` | fold/response probability | Inherited |
| `prior_raise_rate` | `0.3` | raise/action probability | Inherited |
| `exploit_bluff_frequency` | `0.8` | bet probability when steal condition holds | Inherited |
| `fold_profit_margin` | `0.04` | fold-probability points above break-even | Inherited |
| `samples` | `64` | Monte Carlo deals/decision | Inherited |
| `open_equity` | `0.48` | showdown pot share | Inherited |
| `value_equity` | `0.7` | adjusted pot share | Inherited |
| `reraise_equity` | `0.84` | adjusted pot share | Inherited |
| `call_margin` | `0.08` | pot-share probability points | Inherited |
| `pressure_penalty` | `0.16` | pot-share points/unit pressure | Inherited |
| `open_size_bb` | `2.5` | big blinds, street raise-to total | Inherited |
| `bet_pot_fraction` | `0.4` | fraction of pot after calling | Inherited |
| `texture_weight` | `0.06` | equity points and extra pot fraction per normalized feature | Inherited |
| `draw_weight` | `0.12` | equity points per next-card improvement probability | Inherited |
| `bluff_frequency` | `0.12` | probability on eligible free actions | Inherited |
| `features_enabled` | `True` | Boolean | Inherited |
| `bet_range_fraction` | `0.5` | fraction of current strength distribution | Inherited |
| `quiet_range_fraction` | `0.8` | fraction of current strength distribution | Inherited |
| `reraise_range_fraction` | `0.25` | fraction of current strength distribution | Inherited |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 2 | 128 × 250 | -7.42 | 5.93 | [targeted_v9_kraken_seed2.log](targeted_v9_kraken_seed2.log) |
| v7:Bot | 2 | 128 × 250 | -7.64 | 4.41 | [parent_v9_v7_seed2.log](parent_v9_v7_seed2.log) |

**Verdict:** Discarded after larger targeted tests: no demonstrated gain against the strong probe and a negative, inconclusive parent result. Full field not run.

**Decision code:** 91 nonblank, noncomment lines in act, observe and defined helpers, including signatures and multiline returns; constructors, init_state and metadata excluded. The shorter early baselines did not use the entire allowed budget.

## v10: Draw-supported semibluffs

**What it does:** Excludes straight-completing cards that only improve the shared board. Adds the probability of improving to the steal calculation, allowing drawing hands to semibluff when immediate folds alone would be insufficient.

**Why it exists:** Make drawing potential affect semibluff frequency directly and remove shared-board-only straight outs.

**Parameters:** Values are defaults for this immutable version. “Inherited” means unchanged from parent v7; new reasoning choices were tested in the runs below, not fitted to held-out seed 1000.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `semibluff_draw_probability` | `0.15` | next-card improvement probability | Inherited |
| `semibluff_equity` | `0.4` | random-hand pot share | Inherited |
| `pressure_odds_cap` | `0.5` | pot-odds fraction | Inherited |
| `prior_actions` | `12.0` | pseudo-count actions | Inherited |
| `prior_fold_rate` | `0.45` | fold/response probability | Inherited |
| `prior_raise_rate` | `0.3` | raise/action probability | Inherited |
| `exploit_bluff_frequency` | `0.8` | bet probability when steal condition holds | Inherited |
| `fold_profit_margin` | `0.04` | fold-probability points above break-even | Inherited |
| `samples` | `64` | Monte Carlo deals/decision | Inherited |
| `open_equity` | `0.48` | showdown pot share | Inherited |
| `value_equity` | `0.7` | adjusted pot share | Inherited |
| `reraise_equity` | `0.84` | adjusted pot share | Inherited |
| `call_margin` | `0.08` | pot-share probability points | Inherited |
| `pressure_penalty` | `0.16` | pot-share points/unit pressure | Inherited |
| `open_size_bb` | `2.5` | big blinds, street raise-to total | Inherited |
| `bet_pot_fraction` | `0.4` | fraction of pot after calling | Inherited |
| `texture_weight` | `0.06` | equity points and extra pot fraction per normalized feature | Inherited |
| `draw_weight` | `0.12` | equity points per next-card improvement probability | Inherited |
| `bluff_frequency` | `0.12` | probability on eligible free actions | Inherited |
| `features_enabled` | `True` | Boolean | Inherited |
| `bet_range_fraction` | `0.5` | fraction of current strength distribution | Inherited |
| `quiet_range_fraction` | `0.8` | fraction of current strength distribution | Inherited |
| `reraise_range_fraction` | `0.25` | fraction of current strength distribution | Inherited |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 0 | 64 × 250 | +6.48 | 8.26 | [field_v10_seed0.log](field_v10_seed0.log) |
| leviathan_claude | 0 | 64 × 250 | +66.57 | 7.08 | [field_v10_seed0.log](field_v10_seed0.log) |
| megalodon_glm | 0 | 64 × 250 | +57.09 | 8.55 | [field_v10_seed0.log](field_v10_seed0.log) |
| windrunner_glm | 0 | 64 × 250 | +68.24 | 6.17 | [field_v10_seed0.log](field_v10_seed0.log) |
| invoker_gpt | 0 | 64 × 250 | +41.72 | 5.44 | [field_v10_seed0.log](field_v10_seed0.log) |
| meepo_claude | 0 | 64 × 250 | +52.93 | 5.98 | [field_v10_seed0.log](field_v10_seed0.log) |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +53.19 | 4.21 | [field_v10_seed0.log](field_v10_seed0.log) |
| valka_onderra_claude | 0 | 64 × 250 | +54.63 | 7.49 | [field_v10_seed0.log](field_v10_seed0.log) |
| aggressor | 0 | 64 × 250 | +19.79 | 5.60 | [field_v10_seed0.log](field_v10_seed0.log) |
| alan_wake_claude | 0 | 64 × 250 | +72.99 | 5.73 | [field_v10_seed0.log](field_v10_seed0.log) |
| bassander_lin_glm | 0 | 64 × 250 | +98.16 | 10.81 | [field_v10_seed0.log](field_v10_seed0.log) |
| max_payne_gpt | 0 | 64 × 250 | +53.27 | 5.44 | [field_v10_seed0.log](field_v10_seed0.log) |
| jesse_faden_glm | 0 | 64 × 250 | +138.12 | 11.92 | [field_v10_seed0.log](field_v10_seed0.log) |
| jack_joyce_gemini | 0 | 64 × 250 | +120.16 | 12.28 | [field_v10_seed0.log](field_v10_seed0.log) |
| random | 0 | 64 × 250 | +105.30 | 7.69 | [field_v10_seed0.log](field_v10_seed0.log) |
| kraken_gpt | 1000 | 128 × 250 | -2.47 | 5.63 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| leviathan_claude | 1000 | 128 × 250 | +54.71 | 5.54 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| megalodon_glm | 1000 | 128 × 250 | +56.79 | 7.22 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| windrunner_glm | 1000 | 128 × 250 | +68.72 | 5.30 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| invoker_gpt | 1000 | 128 × 250 | +38.27 | 4.47 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| meepo_claude | 1000 | 128 × 250 | +37.92 | 4.39 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| hadrian_marlowe_gpt | 1000 | 128 × 250 | +55.83 | 3.27 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| valka_onderra_claude | 1000 | 128 × 250 | +56.99 | 5.53 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| aggressor | 1000 | 128 × 250 | +21.93 | 3.97 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| alan_wake_claude | 1000 | 128 × 250 | +80.82 | 4.14 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| bassander_lin_glm | 1000 | 128 × 250 | +102.43 | 7.11 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| max_payne_gpt | 1000 | 128 × 250 | +56.65 | 4.03 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| jesse_faden_glm | 1000 | 128 × 250 | +134.14 | 7.59 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| jack_joyce_gemini | 1000 | 128 × 250 | +124.75 | 9.11 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| random | 1000 | 128 × 250 | +108.64 | 5.17 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| ablation_v10:Bot | 1000 | 128 × 250 | -6.10 | 3.99 | [ablation_match_v10_seed1000.log](ablation_match_v10_seed1000.log) |
| v11:Bot | 1000 | 128 × 250 | +8.79 | 4.89 | [heldout_self_v10_v11_seed1000.log](heldout_self_v10_v11_seed1000.log) |
| v1:Bot | 1000 | 128 × 250 | +55.94 | 4.77 | [heldout_self_v10_v1_seed1000.log](heldout_self_v10_v1_seed1000.log) |
| v2:Bot | 1000 | 128 × 250 | +64.18 | 2.65 | [heldout_self_v10_v2_seed1000.log](heldout_self_v10_v2_seed1000.log) |
| v3:Bot | 1000 | 128 × 250 | +15.85 | 4.66 | [heldout_self_v10_v3_seed1000.log](heldout_self_v10_v3_seed1000.log) |
| v4:Bot | 1000 | 128 × 250 | +11.56 | 5.17 | [heldout_self_v10_v4_seed1000.log](heldout_self_v10_v4_seed1000.log) |
| v5:Bot | 1000 | 128 × 250 | -1.58 | 4.57 | [heldout_self_v10_v5_seed1000.log](heldout_self_v10_v5_seed1000.log) |
| v6:Bot | 1000 | 128 × 250 | +2.98 | 4.67 | [heldout_self_v10_v6_seed1000.log](heldout_self_v10_v6_seed1000.log) |
| v7:Bot | 1000 | 128 × 250 | +3.00 | 4.46 | [heldout_self_v10_v7_seed1000.log](heldout_self_v10_v7_seed1000.log) |
| v8:Bot | 1000 | 128 × 250 | +0.86 | 3.72 | [heldout_self_v10_v8_seed1000.log](heldout_self_v10_v8_seed1000.log) |
| v9:Bot | 1000 | 128 × 250 | +6.91 | 4.95 | [heldout_self_v10_v9_seed1000.log](heldout_self_v10_v9_seed1000.log) |

Weighted field score, seed 0, 64 × 250: **+54.45**.

Weighted field score, seed 1000, 128 × 250: **+51.86**.

**Verdict:** Deployed: it has the highest held-out weighted score of the three feature-enabled finalists (+51.86), a corrected private-card draw model, and no clear loss to its parent. The modest margin over v7 is not a claim of statistically proven superiority.

**Decision code:** 92 nonblank, noncomment lines in act, observe and defined helpers, including signatures and multiline returns; constructors, init_state and metadata excluded. The shorter early baselines did not use the entire allowed budget.

## v11: Sampled current-hand ranges

**What it does:** Uses 128 random deals and weights opponents by the rank of their current made hand, retaining a small below-cutoff bluff likelihood. Future cards affect showdown equity but never determine which sampled opponents enter the current betting range.

**Why it exists:** Test a more direct range model than the linear strength-percentile approximation.

**Parameters:** Values are defaults for this immutable version. “Inherited” means unchanged from parent v7; new reasoning choices were tested in the runs below, not fitted to held-out seed 1000.

| Parameter | Value | Unit | Choice |
|---|---:|---|---|
| `range_bluff_weight` | `0.15` | relative likelihood below current-strength cutoff | Reasoned experiment/default; tested in listed training runs |
| `semibluff_draw_probability` | `0.15` | next-card improvement probability | Inherited |
| `semibluff_equity` | `0.4` | random-hand pot share | Inherited |
| `pressure_odds_cap` | `0.5` | pot-odds fraction | Inherited |
| `prior_actions` | `12.0` | pseudo-count actions | Inherited |
| `prior_fold_rate` | `0.45` | fold/response probability | Inherited |
| `prior_raise_rate` | `0.3` | raise/action probability | Inherited |
| `exploit_bluff_frequency` | `0.8` | bet probability when steal condition holds | Inherited |
| `fold_profit_margin` | `0.04` | fold-probability points above break-even | Inherited |
| `samples` | `128` | Monte Carlo deals/decision | Reasoned experiment/default; tested in listed training runs |
| `open_equity` | `0.48` | showdown pot share | Inherited |
| `value_equity` | `0.7` | adjusted pot share | Inherited |
| `reraise_equity` | `0.84` | adjusted pot share | Inherited |
| `call_margin` | `0.08` | pot-share probability points | Inherited |
| `pressure_penalty` | `0.16` | pot-share points/unit pressure | Inherited |
| `open_size_bb` | `2.5` | big blinds, street raise-to total | Inherited |
| `bet_pot_fraction` | `0.4` | fraction of pot after calling | Inherited |
| `texture_weight` | `0.06` | equity points and extra pot fraction per normalized feature | Inherited |
| `draw_weight` | `0.12` | equity points per next-card improvement probability | Inherited |
| `bluff_frequency` | `0.12` | probability on eligible free actions | Inherited |
| `features_enabled` | `True` | Boolean | Inherited |
| `bet_range_fraction` | `0.5` | fraction of current strength distribution | Inherited |
| `quiet_range_fraction` | `0.8` | fraction of current strength distribution | Inherited |
| `reraise_range_fraction` | `0.25` | fraction of current strength distribution | Inherited |

**Results:**

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 3 | 64 × 250 | -11.49 | 10.87 | [targeted_v11_kraken_seed3.log](targeted_v11_kraken_seed3.log) |
| leviathan_claude | 3 | 64 × 250 | +54.25 | 14.03 | [targeted_v11_leviathan_seed3.log](targeted_v11_leviathan_seed3.log) |
| v7:Bot | 3 | 128 × 250 | -19.72 | 5.30 | [parent_v11_v7_seed3.log](parent_v11_v7_seed3.log) |

**Verdict:** Discarded: it lost the 128-table parent match by 19.72 bb/100 (SE 5.30), with no clear targeted strong-opponent gain. Full field not run.

**Decision code:** 109 nonblank, noncomment lines in act, observe and defined helpers, including signatures and multiline returns; constructors, init_state and metadata excluded. The shorter early baselines did not use the entire allowed budget.

## Board texture and draws in v10

Card ranks and suits use the contract's modulo-13 / integer-division encoding; invalid board slots are masked. These are explicit heuristics on top of Monte Carlo showdown equity, which already accounts for possible runouts.

| Feature | Computation | Decision effect at deployed defaults |
|---|---|---|
| Paired board | Any dealt board rank occurs at least twice | Adds 1/3 to normalized danger |
| Suit concentration | Maximum dealt board count in any suit | At least two contributes 1/2 to wetness; at least three contributes 1/3 to danger |
| Connected ranks | Maximum number of distinct board ranks in any five-rank straight window, including the wheel | Three contributes 1/2 to wetness; four contributes 1/3 to danger |
| High board | Highest dealt rank is queen or higher | A dry high board qualifies for an occasional in-position stab |
| Flush improvement cards | Four visible cards of a suit, at least one in our hand; count unseen cards of that suit | Included in next-card draw probability and semibluff support |
| Straight improvement cards | Four distinct ranks in a five-rank window, with at least one rank supplied uniquely by a hole card | Count missing-rank cards; exclude board-only completions and already-made straights |
| Overcard improvement cards | Unpaired hole ranks higher than all board ranks, when no rank is paired | Count unseen cards pairing an overcard |
| Street transition | Use only dealt board cards and denominator 50 minus board count | Draw probability is set to zero on the river; suit, connection and pairing features are recomputed |

Unseen improvement cards are counted as a union over all 52 card IDs, excluding known cards, so overlapping straight/flush/overcard possibilities are not double-counted. They are **candidate improvement cards, not guaranteed winning outs**: opponent holdings can make them dirty. Next-card probability uses one card even on the flop; the Monte Carlo estimate separately includes the full runout.

Let E be uniform-opponent equity, W normalized wetness, G normalized danger, D next-card improvement probability, F estimated fold rate, and b the base pot fraction. Postflop adjusted equity loses `0.06 × G × (1 − E)` and gains `0.12 × D`. Bet sizing grows from `b = 0.40` to `0.40 + 0.06 × W`, i.e. 0.40, 0.43 or 0.46 of the pot after calling, before legal clipping. A draw with D ≥ 0.15 and E ≥ 0.40 qualifies for an occasional semibluff; dry queen-high-or-higher boards qualify in position.

V10 also adds `(1 − F) × D × (1 + 2b) / (1 + b)` to the fold-equivalent steal estimate. This is the algebraic contribution of called-bet improvement equity to a simplified one-street bet EV, not a clean-outs guarantee. Crossing `b / (1 + b) + 0.04` raises bluff frequency from 0.12 on eligible hands to 0.80 on a free postflop action. Thus drawing potential can change betting frequency directly, rather than merely adding a small call margin.

### Feature ablation evidence

All rows below are matched on seed, 128 tables per opponent, and 250 hands per table. Disabling `features_enabled` removes explicit texture discounts, draw bonuses, wet-board sizing, texture/draw eligibility and draw-supported steal credit. Uniform-equity Monte Carlo still sees the board; otherwise the comparison would change more than the requested feature layer.

| Comparison | Features enabled | Features disabled | Enabled minus disabled |
|---|---:|---:|---:|
| v7 field, training seed 2 | +49.35 | +49.26 | +0.09 weighted |
| v10 field, held-out seed 1000 | +51.86 | +49.18 | +2.68 weighted |

Direct play, v7 versus its disabled counterpart on seed 2: **+0.80 bb/100, SE 3.80**. Direct play, deployed v10 versus its disabled counterpart on seed 1000: **-6.10 bb/100, SE 3.99**.

The v7 field difference and head-to-head result do not establish a benefit. The v10 direct-match point estimate favors the disabled control and is within two standard errors of zero, so it also does not establish a benefit. The deployed comparison is the measured effect for v10; modest differences must be interpreted with the reported uncertainty, not as proof that each feature helps. There is no per-feature causal isolation, multiple experiments were tried, and no formal aggregate-score significance test was performed. The mandatory features demonstrably alter decisions, but their performance value is not assumed from that fact.

Disabled v10 field results:

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 1000 | 128 × 250 | +4.99 | 5.47 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| leviathan_claude | 1000 | 128 × 250 | +44.25 | 5.50 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| megalodon_glm | 1000 | 128 × 250 | +50.23 | 7.04 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| windrunner_glm | 1000 | 128 × 250 | +70.39 | 5.03 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| invoker_gpt | 1000 | 128 × 250 | +38.54 | 4.29 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| meepo_claude | 1000 | 128 × 250 | +38.73 | 4.13 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| hadrian_marlowe_gpt | 1000 | 128 × 250 | +55.93 | 3.32 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| valka_onderra_claude | 1000 | 128 × 250 | +52.95 | 4.99 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| aggressor | 1000 | 128 × 250 | +25.87 | 4.12 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| alan_wake_claude | 1000 | 128 × 250 | +75.41 | 3.82 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| bassander_lin_glm | 1000 | 128 × 250 | +77.31 | 6.37 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| max_payne_gpt | 1000 | 128 × 250 | +58.63 | 4.09 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| jesse_faden_glm | 1000 | 128 × 250 | +120.76 | 7.79 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| jack_joyce_gemini | 1000 | 128 × 250 | +120.89 | 7.96 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| random | 1000 | 128 × 250 | +112.26 | 4.87 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |

## Final choice

Deploy **v10**, using the one-line export in `__init__.py`. Its held-out weighted score is **+51.86**, versus **+50.06** for v7 and **+45.46** for v8, each evaluated once on seed 1000 at 128 × 250 per opponent. The deployed version has the higher held-out score of the three feature-enabled finalists. V10's training score was +54.45, compared with v7's +53.35; that small training difference was not treated as a reliable improvement by itself.

The direct held-out parent comparison is **+3.00 bb/100 (SE 4.46)** against v7. All ten other written versions, including later experiment v11, were checked, rather than only selecting favorable ancestors. The disabled-feature strategy is a required control and is not eligible for deployment under this task's explicit board/draw requirement.

### Held-out field, every benchmark

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 1000 | 128 × 250 | -2.47 | 5.63 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| leviathan_claude | 1000 | 128 × 250 | +54.71 | 5.54 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| megalodon_glm | 1000 | 128 × 250 | +56.79 | 7.22 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| windrunner_glm | 1000 | 128 × 250 | +68.72 | 5.30 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| invoker_gpt | 1000 | 128 × 250 | +38.27 | 4.47 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| meepo_claude | 1000 | 128 × 250 | +37.92 | 4.39 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| hadrian_marlowe_gpt | 1000 | 128 × 250 | +55.83 | 3.27 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| valka_onderra_claude | 1000 | 128 × 250 | +56.99 | 5.53 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| aggressor | 1000 | 128 × 250 | +21.93 | 3.97 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| alan_wake_claude | 1000 | 128 × 250 | +80.82 | 4.14 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| bassander_lin_glm | 1000 | 128 × 250 | +102.43 | 7.11 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| max_payne_gpt | 1000 | 128 × 250 | +56.65 | 4.03 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| jesse_faden_glm | 1000 | 128 × 250 | +134.14 | 7.59 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| jack_joyce_gemini | 1000 | 128 × 250 | +124.75 | 9.11 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| random | 1000 | 128 × 250 | +108.64 | 5.17 | [field_v10_seed1000.log](field_v10_seed1000.log) |

### Held-out own-version and ablation matches

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| ablation_v10:Bot | 1000 | 128 × 250 | -6.10 | 3.99 | [ablation_match_v10_seed1000.log](ablation_match_v10_seed1000.log) |
| v11:Bot | 1000 | 128 × 250 | +8.79 | 4.89 | [heldout_self_v10_v11_seed1000.log](heldout_self_v10_v11_seed1000.log) |
| v1:Bot | 1000 | 128 × 250 | +55.94 | 4.77 | [heldout_self_v10_v1_seed1000.log](heldout_self_v10_v1_seed1000.log) |
| v2:Bot | 1000 | 128 × 250 | +64.18 | 2.65 | [heldout_self_v10_v2_seed1000.log](heldout_self_v10_v2_seed1000.log) |
| v3:Bot | 1000 | 128 × 250 | +15.85 | 4.66 | [heldout_self_v10_v3_seed1000.log](heldout_self_v10_v3_seed1000.log) |
| v4:Bot | 1000 | 128 × 250 | +11.56 | 5.17 | [heldout_self_v10_v4_seed1000.log](heldout_self_v10_v4_seed1000.log) |
| v5:Bot | 1000 | 128 × 250 | -1.58 | 4.57 | [heldout_self_v10_v5_seed1000.log](heldout_self_v10_v5_seed1000.log) |
| v6:Bot | 1000 | 128 × 250 | +2.98 | 4.67 | [heldout_self_v10_v6_seed1000.log](heldout_self_v10_v6_seed1000.log) |
| v7:Bot | 1000 | 128 × 250 | +3.00 | 4.46 | [heldout_self_v10_v7_seed1000.log](heldout_self_v10_v7_seed1000.log) |
| v8:Bot | 1000 | 128 × 250 | +0.86 | 3.72 | [heldout_self_v10_v8_seed1000.log](heldout_self_v10_v8_seed1000.log) |
| v9:Bot | 1000 | 128 × 250 | +6.91 | 4.95 | [heldout_self_v10_v9_seed1000.log](heldout_self_v10_v9_seed1000.log) |

### Known weaknesses and limits

- The strongest field opponent remains close to break-even or slightly losing in held-out play; this is not a claim to beat every opponent reliably.
- The main range model is a linear strength-percentile approximation to random-hand equity, not a solved strategy or a calibrated posterior over hands. Sixty-four samples introduce decision noise.
- Aggregate raise/fold rates mix situations and are affected by our own changing strategy. The twelve-action prior stabilizes early play but does not remove that selection effect.
- Draw outs can be dirty, and the improvement bonus overlaps information already present in Monte Carlo equity. The ablation therefore matters; feature usefulness is not guaranteed.
- The low bet sizes and frequent steals can be exploited by opponents that call or reraise appropriately. Validation against the fixed field cannot establish general superiority against unseen phase-5 lineages.
- v9's additional reraise adaptation earned no clear gain. v11's sampled range model lost to v7 by 19.72 bb/100 (SE 5.30) and was discarded without a full field run.
- No source version was revised after benchmarking. The initial out-of-directory filename search and the early parameter-exposure shortfall are disclosed above; they are not silently erased.

### Verification

```text
v10:Bot metadata, sample cap, single class: PASS
JIT/vmap shapes, int32 actions, legal raise totals, no free folds: PASS
Features enabled versus disabled: 13 action-type changes; 133 total action changes / 768 probes
```

All eleven versions passed smoke execution. The final version uses **64 samples per decision**, fixed-shape scalar float32 summary counts, no stored random keys, and **92 decision-code lines**. Static checks verified imports, JSON metadata and provenance, sample caps, one class per version, and fingerprints of frozen sources. The exact parameter-only ablation difference was checked. `bench.py` and `equity.py` retain their recorded hashes.


## Files and reproduction

- `v1.py` onward are immutable version sources; `__init__.py` names the deployed version.
- `ablation_v*.py` are self-contained copies of the indicated strategy with only the default `features_enabled=False` (and descriptive docstring) changed. They import no other versions. Their metadata reports the originating version and disabled parameter explicitly.
- `smoke_*.log` are short execution checks, not scored results. `field_*.log`, `parent_*.log`, `targeted_*.log`, `ablation_match_*.log` and `heldout_self_*.log` preserve raw benchmark output, including first-50/rest comparisons and timings.
- `diagnose.py` and its log summarize played public actions and outcomes; they do not inspect opponent code or hidden cards.
- `validate.py` checks metadata, sample limits, action dtypes, legal action sizes and feature-sensitive decisions on constructed observations; `validation_*.log` records its output.
- `run_notes.txt`, `final_notes.md`, `report.py`, `finish_notes.py`, `results.json` and `frozen_manifest.json` are audit/reporting scratch files. `frozen_manifest.json` records source fingerprints and decision counts.

Examples, run from this directory with the existing environment:

```sh
uv run python bench.py check
uv run python bench.py v10:Bot field --seed 1000 --tables 128 --hands 250
uv run python validate.py v10:Bot
```

The held-out command documents reproduction only; it was not repeatedly rerun to select a favorable result.

