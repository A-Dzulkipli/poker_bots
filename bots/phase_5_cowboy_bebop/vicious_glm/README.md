# vicious_glm — phase_5_cowboy_bebop

- **Bot name:** `vicious_glm`
- **Lineage:** phase_5_cowboy_bebop/vicious_glm
- **Model:** GLM-5.3 (account:zai-individual-coding-plan)
- **Prompt version:** botgen-v5
- **Engine:** decafb9, engine_sha256 295477eff7637056
- **Wall-clock time used:** started 16:05, wrapped 16:55 (50 minutes; the last background
  benchmark finished while the README was being written)
- **Code executed:** yes
- **Environment:** project `.venv` via `uv run` from the bot directory (`bench.py check` below)
- **`bench.py check` output:**

```
python      /Users/alik/Documents/Projects/poker_bots/poker_bots/.venv/bin/python3 (3.13.12)
jax         0.11.2 (cpu)
poker_bots  /Users/alik/Documents/Projects/poker_bots/poker_bots/src/poker_bots
engine      sha256 295477eff7637056 (matches the contract)
smoke test  random vs aggressor, 40 hands in 2s: OK
```

## Design (deployed version, v5)

Every decision starts from a 128-sample Monte Carlo estimate of showdown equity against
uniformly random opponent cards (`equity.py::equity`), then applies readable thresholds. Board
texture (flushiness, straight-window connectedness, high cards, pairing) and explicit draw outs
(the number of next-street cards that promote the made hand to two pair or better, computed by
enumerating all 52 candidates through the evaluator) shift bet sizing, calling margins,
semi-bluff frequency and dry-board air bluffs. Because equity against random cards overrates
hands against real ranges, the equity demanded to call grows with the fraction of the stack at
risk ("stack-off discipline", v3) and the equity demanded to bet for value grows with the
street and pot depth ("thin-value discipline", v5). A single tracked statistic — the opponent's
hand fold rate — scales air-bluff frequency. Preflop, the button open-raises almost every hand
at 2.5bb, the big blind raises limps with the top of range, and raise-defense folds stack risk
into the required equity.

## Board texture and draws (required feature evidence)

Features the deployed v3 computes, all in `board_texture` and `draw_outs`:

- **flushiness** — largest suit count on the board / 3, clipped to [0,1] (3+ suited cards ⇒ 1.0).
- **connectedness** — largest number of board ranks inside any 5-rank straight window (including
  the wheel A-5), / 3, clipped. Windows are a 10×13 0/1 table, so this is one matmul.
- **high cards** — (top board rank − 5)/7, clipped: 7-high boards score 0, ace-high 1.
- **wetness** = 0.45·flushiness + 0.35·connectedness + 0.20·high_cards, in [0,1].
- **paired flag** — any rank twice on the board.
- **draw outs** — for every one of the 52 candidate next cards (masked to unseen ones), evaluate
  the 6- or 7-card hand it would create; count cards whose hand category is two pair or better
  and above the current category. 0 preflop and on the river.

Decisions each feature changes:

- **Value/semi-bluff sizing:** bet fraction = 0.55 + 0.30·wetness·scale of pot after call
  (0.55 pot on a dry board, 0.85 on a soaked one) — texture.
- **Facing a bet, call margin:** +0.10·wetness·faced-bet-fraction demanded, −0.02 on paired
  boards (fewer draws complete; bluff-catchers hold up) — texture.
- **Raise sizing vs a bet:** +0.15·wetness on top of the base 0.65 pot-after-call — texture.
- **Air bluffs** (equity < 0.40) only fire on boards with wetness < 0.45 — texture.
- **Semi-bluffs:** with ≥6 improving cards and equity > 0.35, bet with probability 0.75 when
  checked to; with ≥8 outs, raise bets with probability 0.5 — draws.
- **Implied odds:** the call threshold is reduced by 0.05 equity with a big draw (≥8 outs) — draws.

**Evidence** (deployed v3 vs `flat:Bot`, the identical code with `texture_effect_scale=0.0`
and `draw_effect_scale=0.0`, so every effect above is switched off while sizing, pot odds and
stack-off discipline stay; the same ablation was also run on the v5 code for comparison):

| comparison | tables × hands | seed | result |
|---|---|---|---|
| v3 vs flat, head to head | 16×250 | 1 | v3 +30.5 bb/100 (SE 42.1) |
| v3 vs flat, head to head | 48×250 | 2 | v3 +12.4 bb/100 (SE 26.8) |
| v3 vs flat, head to head | 48×250 | 4 | v3 −55.2 bb/100 (SE 25.0) |
| v3 vs field | 32×250 | 0 | **+13.14 weighted** |
| flat(v3 code) vs field | 32×250 | 0 | **+13.54 weighted** |
| v5 vs field | 32×250 | 0 | +15.71 weighted |
| flat(v5 code) vs field | 32×250 | 0 | +9.66 weighted |
| v5 vs flat(v5 code), head to head | 48×250 | 2 | v5 −7.0 bb/100 (SE 25.0) |

Honest reading: the features demonstrably change decisions (bet sizing spans 0.55–0.85 pot with
wetness; draws add 0.75-frequency semi-bluffs and 0.05-equity implied-odds credit that fire in
roughly a quarter of postflop spots), but at these sample sizes their measured worth does not
separate from zero: the three v3-vs-flat head-to-heads average about −4 bb/100 with ±25–42 SEs,
the v3-pairing field comparison is a wash (+13.14 vs +13.54), and only the v5 pairing shows the
features worth +6.05 weighted on the field. The fair claim is "implemented, materially active,
worth roughly 0 to +6 weighted depending on pairing, not resolvable at 32-table precision" —
not a proven win.

## Versions

### v1 — equity core with texture and draws

- **What it does:** Monte Carlo equity (128 samples) against random cards drives everything:
  button open-raises at 3bb above 0.44 equity and limps the rest; postflop it bets equity > 0.63
  sized 0.55–0.85 pot by wetness, semi-bluffs 6+ outs, calls bets when equity clears pot odds
  plus a small wetness-scaled margin, and bluffs air on dry boards scaled by the tracked
  opponent fold rate.
- **Why it exists:** baseline that satisfies the texture/draws requirement with readable knobs.
- **Parameters:** 28 constructor args, all listed in `v1.py` with units in their comments;
  defaults set by first-principles poker reasoning, none tuned (see `metadata().params`).
- **Results** (32 tables × 250 hands, seed 0; full log `field_v1_t32_s0.log`):

| opponent | bb/100 | SE | | opponent | bb/100 | SE |
|---|---|---|---|---|---|---|
| kraken_gpt | −121.5 | 13.2 | | aggressor | −31.3 | 16.5 |
| leviathan_claude | −135.3 | 27.6 | | alan_wake_claude | +34.4 | 26.3 |
| megalodon_glm | −122.3 | 21.3 | | bassander_lin_glm | −96.4 | 35.4 |
| windrunner_glm | −138.3 | 25.9 | | max_payne_gpt | +47.3 | 19.7 |
| invoker_gpt | −134.2 | 19.1 | | jesse_faden_glm | +208.8 | 45.8 |
| meepo_claude | −63.6 | 23.9 | | jack_joyce_gemini | +281.7 | 44.1 |
| hadrian_marlowe_gpt | −79.3 | 20.5 | | random | +629.5 | 33.1 |
| valka_onderra_claude | −102.1 | 29.9 | | **weighted score** | **−23.87** | |

- **Verdict:** kept as parent. Diagnosis (scratch `debug.py`, 16×250 vs kraken_gpt): showdown
  losses averaged −26.3 bb while showdown wins averaged +12.8 — random-card equity called
  all-ins far too light — and limped buttons bled chips.
- **Line count:** 76 (act, observe, helpers, module table setup; excluding `__init__`,
  `init_state`, `metadata`).

### v2 — stop playing weak-passive

- **What it does:** one group of changes on v1: open-raise nearly every button at 2.5bb instead
  of limping below 0.44 equity, value-bet from 0.56 equity instead of 0.63, semi-bluff draws 75%
  of the time, bluff more air, and discount faced bets harder (0.22 vs 0.10 per bet-fraction).
- **Why it exists:** v1's weak-passive lines (limps, passive medium hands) looked like the leak.
- **Parameters:** only the six above changed from v1 (`open_equity` 0.44→0.33, `open_size_bb`
  3.0→2.5, `limp_raise_equity` 0.53→0.50, `value_bet_equity` 0.63→0.56, `semi_bluff_freq`
  0.60→0.75, `air_bluff_fold_scale` 0.80→1.00, `bet_discount` 0.10→0.22); reasoning, not tuned.
- **Results** (16 tables × 250 hands, seed 0, quick probes): kraken_gpt −129.1 (SE 20.8),
  windrunner_glm −155.0 (SE 31.9), meepo_claude −31.4 (SE 41.5), random +548.8 (2×20 smoke).
  No full `field` run — the probes showed the change was not the fix.
- **Verdict:** discarded as a strategy step (worse vs windrunner and kraken than v1), but kept
  as v3's parent because its aggression settings are standard HU play and the diagnosis it
  enabled (stack-off leak) was the actual fix; v3 = v2 + discipline.
- **Line count:** 76.

### v3 — stack-off discipline (deployed)

- **What it does:** v2 plus: the equity demanded to call grows with the fraction of the stack at
  risk — +0.25 equity per unit of postflop stack risk, +0.20 preflop, plus +0.04 on turn and
  river — so deep calls and preflop all-ins need a real hand while cheap bets still get normal
  pot-odds treatment.
- **Why it exists:** v2's diagnostic showed −26 bb average showdown losses; random-card equity
  made it call all-ins with hands crushed by any strong bot's jam range.
- **Parameters:** three added (`stack_risk_extra` 0.25, `preflop_stack_risk_extra` 0.20,
  `late_street_extra` 0.04; reasoning from the −26 bb average showdown loss, not tuned).
- **Results** (32 tables × 250 hands, seed 0; full log `field_v3_t32_s0.log`):

| opponent | bb/100 | SE | | opponent | bb/100 | SE |
|---|---|---|---|---|---|---|
| kraken_gpt | −72.6 | 12.9 | | aggressor | −4.7 | 10.0 |
| leviathan_claude | −86.4 | 23.4 | | alan_wake_claude | +63.9 | 23.2 |
| megalodon_glm | −94.4 | 20.5 | | bassander_lin_glm | +93.8 | 36.1 |
| windrunner_glm | −83.4 | 19.8 | | max_payne_gpt | +77.9 | 14.6 |
| invoker_gpt | −60.4 | 19.8 | | jesse_faden_glm | +305.9 | 38.9 |
| meepo_claude | −41.6 | 19.6 | | jack_joyce_gemini | +407.3 | 46.6 |
| hadrian_marlowe_gpt | −38.8 | 14.9 | | random | +614.4 | 32.5 |
| valka_onderra_claude | −25.8 | 31.2 | | **weighted score** | **+13.14** | (plain +31.49) |

- **Verdict:** kept as parent of v4 and deployed as the final version (see Final choice).
  Weighted score went from −23.87 (v1) to +13.14; the strong half's losses were cut roughly in
  half everywhere. Held-out seed 1000: +13.03 (plain +34.31), consistent with seed 0.
- **Line count:** 79.

### v4 — deeper discipline and near-nut raising (discarded)

- **What it does:** v3 with stack-risk cushions pushed further (0.35 postflop / 0.28 preflop),
  late-street extra 0.06, and re-raises only with near-nut equity (0.83 vs random) or a huge
  combo draw.
- **Why it exists:** v3's remaining lost showdowns still averaged −25 bb, so one more turn of
  the same screw seemed the cheapest lever.
- **Parameters:** four changed from v3 (`stack_risk_extra` 0.25→0.35,
  `preflop_stack_risk_extra` 0.20→0.28, `late_street_extra` 0.04→0.06,
  `monster_equity` 0.78→0.83).
- **Results** (16 tables × 250 hands, seed 0 probes): kraken_gpt −86.2 (SE 17.0; v3: −56.9),
  windrunner_glm −84.5 (SE 24.6; v3: −79.6), meepo_claude +11.8 (SE 33.0; v3: −48.3),
  invoker_gpt −65.1 (SE 17.7; v3: −60.4 at 16 tables). No field run — the probes regressed
  against the bots that matter most for the weighted score.
- **Verdict:** discarded. Over-folding: kraken and invoker punish tightness by pounding every
  checked pot, and the lost bluff-eating outweighs the fewer bad showdowns. v3 kept.
- **Line count:** 79.

### v5 — thin-value discipline (deployed)

- **What it does:** v3 (v4 was a discarded side branch) plus: the equity needed to bet for value
  grows by 0.06 on turn and river and by 0.15 per unit of pot-to-stack depth, so it stops
  value-betting "strong versus random" hands into big late-street pots where only better hands
  continue; flop c-bets stay thin because worse hands still float.
- **Why it exists:** v3's diagnostic vs invoker_gpt showed 556 lost showdowns (13.9% of hands)
  at −19.7 bb average against 277 wins — the betting mirror of v2's calling leak: initiators'
  continuing opponents also beat random-card equity.
- **Parameters:** two added (`late_value_extra` 0.06, `deep_value_extra` 0.15; sized from the
  −19.7 bb average showdown loss, not tuned).
- **Results:** 16×250 probes seed 0 — invoker_gpt −40.7 (SE 20.2; v3 field: −60.4),
  valka_onderra_claude **+58.1** (SE 35.7; v3: −25.8), kraken_gpt −67.4 (SE 17.9; v3: −72.6),
  meepo_claude −77.4 (SE 24.5; v3: −41.6, an acknowledged regression), random smoke +548.8.
  Field 32×250 seed 0 (full log `field_v5_t32_s0.log`):

| opponent | bb/100 | SE | | opponent | bb/100 | SE |
|---|---|---|---|---|---|---|
| kraken_gpt | −72.9 | 11.7 | | aggressor | −0.2 | 12.8 |
| leviathan_claude | −106.3 | 21.4 | | alan_wake_claude | +35.7 | 23.4 |
| megalodon_glm | −107.6 | 19.4 | | bassander_lin_glm | +66.1 | 34.7 |
| windrunner_glm | −92.2 | 16.6 | | max_payne_gpt | +66.3 | 14.7 |
| invoker_gpt | −66.7 | 17.6 | | jesse_faden_glm | +293.6 | 43.3 |
| meepo_claude | −41.2 | 20.2 | | jack_joyce_gemini | +449.5 | 49.0 |
| hadrian_marlowe_gpt | −28.8 | 9.3 | | random | +565.0 | 27.9 |
| valka_onderra_claude | **+61.4** | 25.4 | | **weighted score** | **+15.71** | (plain +32.63) |

  Held-out field 32×250 seed 1000: weighted **+10.21** (plain +24.03; valka reverted to −26.2,
  invoker improved to −20.0 — see log `field_v5_t32_s1000.log`).
  Head-to-head vs parent v3 (32×250, seed 3): v5 **lost −26.9 bb/100** (SE 27.3).
- **Verdict:** discarded despite the seed-0 field win. Its +2.6 weighted edge over v3 on seed 0
  came almost entirely from a valka_onderra_claude flip (+61.4) that reverted to −26.2 on the
  held-out seed, where v5 scored +10.21 against v3's +13.03; v5 also loses head-to-head to its
  own parent (−26.9), the exact warning the protocol flags. v3 stayed deployed.
- **Line count:** 81. (After benchmarking, one provenance-only fix was made to `v5.py`:
  `metadata()["parent"]` 2 → 3. No decision code, parameter or behavior changed.)

## Final choice

**Deployed: v3** (`__init__.py` → `from .v3 import Bot as DeployedBot`). Both finalists were
run once on the held-out seed 1000 (32 tables × 250 hands): v3 **+13.03** weighted (plain
+34.31), v5 +10.21 — and v5 additionally loses head-to-head to v3 (−26.9 bb/100 at 32×250,
seed 3), the parent-beating warning sign the protocol names. v3's field scores are also the more
consistent pair (+13.14 seed 0, +13.03 seed 1000) where v5 swings (+15.71 / +10.21). v3 beats
its own parent v2 by +53.0 bb/100 (32×250, seed 3). Full held-out v3 table (seed 1000,
32×250): kraken_gpt −61.8 (SE 14.7), leviathan_claude −42.0 (19.9), megalodon_glm −86.5
(20.4), windrunner_glm −84.4 (20.8), invoker_gpt −43.4 (21.6), meepo_claude −71.5 (21.2),
hadrian_marlowe_gpt −17.4 (12.5), valka_onderra_claude −22.5 (24.9), aggressor −19.6 (11.1),
alan_wake_claude +66.7 (28.6), bassander_lin_glm +77.0 (34.7), max_payne_gpt +39.0 (15.5),
jesse_faden_glm +337.5 (34.8), jack_joyce_gemini +409.0 (47.6), random +658.5 (28.0);
**weighted +13.03** (log `field_v3_t32_s1000.log`).

Known weaknesses: it loses to every rank-8+ opponent (they adapt within sessions; the only
counter-adaptation is the fold-rate-scaled bluff frequency); it rarely slow-plays or
check-raises as a trap; equity versus random cards remains a crude range model; and the
texture/draw features, while active, are not proven worth >0 at 32-table precision (see the
ablation table above). No parameter was tuned against the field beyond four reasoned one-step
changes, so there is headroom.

Scratch files kept in this directory: `field_v1_t32_s0.log`, `field_v3_t32_s0.log`,
`field_v3_t32_s1000.log`, `field_v5_t32_s0.log`, `field_v5_t32_s1000.log`,
`field_flat_t32_s0.log` (flat = v3 code ablated), `field_flat_v5code_t32_s0.log` (v5 code
ablated), `h2h_v5_flat_t48_s2.log`, `h2h_v5_v3_t32_s3.log` (benchmark logs), `debug.py`
(outcome-bucketing diagnostic), `flat.py` (ablation bot, points at the deployed v3).

## Interpretable parameters

All 31 constructor arguments of v3 are documented with units in `v1.py`/`v3.py` comments and
echoed in `metadata()["params"]`. Only v2's six aggression changes and v3's three
stack-discipline additions deviate from v1's first-principles defaults; nothing was tuned
against the benchmark beyond those reasoned one-step changes.
