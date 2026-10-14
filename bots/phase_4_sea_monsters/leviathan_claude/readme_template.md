# leviathan_claude

- **Bot name:** leviathan_claude
- **Lineage:** phase_4_sea_monsters/leviathan_claude
- **Model:** claude-opus-5-5 (Claude Opus 5.5)
- **Prompt version:** botgen-v4
- **Engine:** decafb9, engine_sha256 295477eff7637056
- **Wall-clock time used:** about 43 minutes, from the passing `bench.py check` at 03:59 UTC to the finished README at 04:42 UTC. About 4 minutes of environment setup came before that.
- **Code executed:** yes. All numbers below come from `bench.py` runs.
- **Environment note:** the session's shell on the requester's machine is a Linux (aarch64) VM. The repository's `.venv` is a macOS environment and cannot run there. With the coordinator's approval, every run used a separate Linux venv outside the repository: `UV_PROJECT_ENVIRONMENT=$HOME/p4venv uv run --frozen python bench.py ...`. It installed only jax and the project dependencies. Nothing in the repository outside this directory was changed.
- **`bench.py check` output:**

```
python      /sessions/rcw-01ekebevaflbfav6p6zcd3go/p4venv/bin/python (3.13.15)
jax         0.11.2 (cpu)
poker_bots  /sessions/rcw-01ekebevaflbfav6p6zcd3go/mnt/poker_bots/poker_bots/src/poker_bots
engine      sha256 295477eff7637056 (matches the contract)
smoke test  random vs aggressor, 40 hands in 4s: OK
```

**Deployed: v4** (`__init__.py`: `from .v4 import Bot as DeployedBot`). Held-out weighted score: **+72.59** (seed 1000).

**How the runs were made.** Background processes did not survive between shell calls in this environment, so `bench.py ... field` could not run to completion in one call. Every field result here is twelve separate `bench.py vK:Bot <opponent> --seed S` runs (64 tables × 250 hands each), run several at a time by `run_part.sh`. The runs use the same settings and seeds as `field` would. The weighted score was then computed by `score.py`, which uses bench.py's formula: a win counts rank/6.5, a loss counts (13 − rank)/6.5, and the mean is taken over the twelve opponents.

**Scratch files in this directory:**
- `logs/*.log`: raw bench output for each version and seed.
- `logs/tables.md`: every result table.
- `run_part.sh`: the parallel runner.
- `score.py`: computes the weighted score.
- `count_lines.py`: counts decision-code lines.
- `make_tables.py`: builds the tables.
- `readme_template.md`: source for this file.

`logs/v1_field_s0.log` and `logs/v2_field_s0.log` are empty leftovers from the killed background attempts. The sandbox does not allow deleting them.

**Borrowed ideas:** the v1 baseline follows the structure of the benchmark bot `windrunner_glm`, which was strongest in phase 3. That covers opening every button, the per-bet equity discount ("betting pressure"), the street-decaying bluff rates, and the initial parameter values. The code was rewritten into one helper per situation. Nothing in any version identifies or targets a particular opponent.

---

## v1: stateless baseline

**What it does:** It opens every button hand to 2.5bb. From the big blind it 3-bets hands with at least 63% equity against a random hand, plus 15% of the hands it would call, and raises limps with at least 55% equity. Postflop when checked to, it bets 65% of the pot with at least 62% equity. It bluffs air (equity below 45%) 33% of the time and medium hands or draws 50% of the time on the flop, and those rates shrink by 0.55× per later street. Facing a bet, it raises with at least 80% equity. Otherwise it calls only when its equity, multiplied by 0.58 for each opponent bet this hand (0.5 weight for bets on earlier streets) and by 0.9 when more betting can follow, beats the pot odds by 2%.

**Why it exists:** It gives a sound heads-up baseline with no state, so the effect of tracking can be measured against the same strategy without it.

**Parameters (all default/reasoning, taken from the phase-3 winner's tuned values, not retuned here):**
- `num_samples` = 128 (Monte Carlo samples)
- `open_size_bb` = 2.5 (big blinds)
- `reraise_multiple` = 3.0 (× current bet)
- `three_bet_equity` = 0.63 (probability)
- `light_three_bet_rate` = 0.15 (probability)
- `limp_raise_equity` = 0.55 (probability)
- `value_bet_equity` = 0.62 (probability)
- `air_bluff_rate` = 0.33 (probability)
- `semi_bluff_rate` = 0.50 (probability)
- `air_equity_limit` = 0.45 (probability)
- `bluff_street_decay` = 0.55 (multiplier per street)
- `bet_pot_fraction` = 0.65 (pot fraction)
- `value_raise_equity` = 0.80 (probability)
- `call_margin` = 0.02 (probability)
- `pressure_discount` = 0.58 (equity multiplier per bet)
- `earlier_street_weight` = 0.5 (bets)
- `realization_factor` = 0.90 (share of equity)

**Results (seed 0), weighted score +67.34:**
{v1_s0}

**Verdict:** Kept as the parent of v2 and v3.

**Line count:** 45.

## v2: v1 + tracking of postflop aggression (discarded)

**What it does:** It is the same as v1, except that it tracks the share of the opponent's postflop actions that are bets or raises. The prior is 0.35, worth 20 pseudo-actions. That share then scales how deeply each opponent bet discounts calling equity. Against a frequent bettor the discount is shallower, so it calls down lighter. Against a rare bettor the discount is up to 1.5× deeper.

**Why it exists:** It tests whether "call down lighter against a frequent bettor" pays off when frequency is the only evidence.

**Parameters:** v1's, plus three more, all set by reasoning and not tuned:
- `typical_aggression` = 0.35 (share of actions)
- `aggression_prior_actions` = 20 (pseudo-count actions)
- `max_pressure_scale` = 1.5 (multiplier)

**Results (seed 0, stopped after the four strongest-weighted opponents):**
{v2_s0}

**Verdict:** Discarded. On the same seed it lost 13 bb/100 against v1's windrunner result and 44 against v1's meepo result. Strong opponents bet often *and* mostly for value, so bet frequency alone made it pay off value bets. The idea came back in v4, driven by showdown evidence instead of frequency.

**Line count:** 57.

## v3: v1 + tracking of the fold rate after our bets

**What it does:** It is the same strategy as v1. It counts the hands in which we bet postflop and how many of those the opponent folded. The prior is a 0.5 fold rate, worth 10 pseudo-hands. The ratio of the observed fold rate to that typical rate does two things:
- **Bluff rates:** they are multiplied by this ratio, clipped to 0.3–1.6. A folder gets bluffed more and a caller less.
- **Value threshold:** against an opponent that folds less than typical, the value-bet threshold drops by up to 0.08 equity, so it bets thinner for value.

**Why it exists:** It is the first tracking step. Bluffs pay only against opponents that fold, and thin value pays only against opponents that call. Both depend on one statistic.

**Parameters:** v1's, plus five more, all set by reasoning and not tuned:
- `typical_fold_rate` = 0.5 (share of hands)
- `fold_prior_hands` = 10 (pseudo-count hands)
- `min_bluff_scale` = 0.3 (multiplier)
- `max_bluff_scale` = 1.6 (multiplier)
- `thin_value_shift` = 0.08 (equity probability)

**Results (seed 0), weighted score +69.97 (v1: +67.34):**
{v3_s0}

Head to head (seed 2): v3 vs v1 **+27.84 ± 6.55**.

**Verdict:** Kept as the parent of v4. The field gain is within noise, but v3 clearly beats its stateless parent head to head (about 4 SE). v1 never changes how often it folds to bets, so v3's bluff-rate scaling finds and uses that.

**Line count:** 59.

## v4: v3 + showdown evidence of bluffing (DEPLOYED)

**What it does:** On top of v3, it counts the showdowns in which the opponent had bet or raised postflop, and how many of those the opponent lost (its shown hand ranked below ours). The prior is a 0.35 loss rate, worth 6 pseudo-showdowns. The ratio of the typical loss rate to the observed one, clipped to 0.4–1.25, scales the depth of the per-bet equity discount. Against an opponent whose bets often lose at showdown (a bluffer), it calls down lighter. Against one whose bets are always the best hand, it folds more.

**Why it exists:** v2 showed that bet frequency cannot tell aggression from strength. A showdown loss after betting is direct evidence of a bluff or thin value, so it should help against wild bettors without paying off value bettors.

**Parameters:** v3's, plus four more, all set by reasoning and not tuned:
- `typical_bet_loss_rate` = 0.35 (share of showdowns)
- `showdown_prior_count` = 6 (pseudo-count showdowns)
- `min_pressure_scale` = 0.4 (multiplier)
- `max_pressure_scale` = 1.25 (multiplier)

**Results (seed 0), weighted score +76.21 (v3: +69.97):**
{v4_s0}

Head to head (seed 2):
- v4 vs v3: **+0.77 ± 7.02**
- v4 vs v1: **+16.70 ± 6.65**

**Verdict:** Kept and deployed. The gain is concentrated where the hypothesis predicts. Against the wild bettors it rose sharply (random +140, jack_joyce +68, jesse_faden +47, alan_wake +18, bassander +20 bb/100 over v3 on the same seed), and the strong opponents stayed within noise.

**Line count:** 69.

## v5: v4 + a larger bet size against non-folders (discarded)

**What it does:** It is the same as v4, except that the postflop bet size grows from 0.65 pot by up to 0.35 pot as the tracked fold rate falls below typical. Bluffs use the same size, so the size reveals nothing about the hand.

**Why it exists:** Against opponents that call too much, value bets earn more when they are bigger.

**Parameters:** v4's, plus one more:
- `caller_extra_bet_fraction` = 0.35 (pot fraction), set by reasoning.

**Results (seed 0), weighted score +78.16 (v4: +76.21):**
{v5_s0}

Held-out results (seed 1000), weighted score **+71.64** (v4: +72.59):
{v5_s1000}

Head to head (seed 2): v5 vs v4 **−9.53 ± 8.06**.

**Verdict:** Discarded. On seed 0 it was ahead only within noise, it was behind on the held-out seed, and it lost to its parent head to head. The strong opponents fold at or above the typical rate, so the larger size rarely triggers against them.

**Line count:** 71.

---

## Opponent tracking (deployed v4)

State: four float32 counters, all starting at 0. Each rate is computed as (count + prior × pseudo-count) / (opportunities + pseudo-count).

| statistic | counters | prior | decisions it changes |
|---|---|---|---|
| fold rate after our postflop bets | `hands_i_bet_postflop`, `opponent_folds_after_my_bet` | 0.5, worth 10 hands | **Bluff rates:** both the air and the semi-bluff rate when checked to are multiplied by observed rate / 0.5, clipped to 0.3–1.6. **Value threshold:** when the opponent folds less than typical, it drops from 0.62 by up to 0.08 equity. |
| loss rate at showdown after the opponent bet postflop | `opponent_bet_showdowns`, `opponent_bet_showdowns_lost` | 0.35, worth 6 showdowns | **Equity discount per opponent bet:** the discount is 1 − 0.42 × clip(0.35 / observed rate, 0.4, 1.25), giving a range from 0.83 against a heavy bluffer to 0.475 against a pure value bettor, with 0.58 at the prior. It drives every call or fold decision facing a bet, preflop 3-bets included. |

**Evidence that tracking helps:**
- **Against the same strategy without tracking.** On seed 0 the field score went from +67.34 (v1, stateless) to +69.97 (v3) and +76.21 (v4). Head to head, v4 beats v1 by +16.70 ± 6.65 and v3 beats v1 by +27.84 ± 6.55.
- **First 50 hands against the rest.** Against the wild bettors, v4 earns clearly more after hand 50 on both seeds. That is when the showdown counts have moved off the prior. On seed 0 / seed 1000 (first 50 → rest):
  - bassander: 122 → 193 / 154 → 186
  - jesse_faden: 196 → 273 / 216 → 262
  - jack_joyce: 259 → 370 / 326 → 409
  - random: 465 → 538 / 433 → 536
  - valka (seed 1000 only): 47 → 75
  Against the strong stateless bots the split is noisy and points both ways. Windrunner went −7 → +23 on seed 0 but +24 → +16 on seed 1000, and aggressor +44 → +48 then +56 → +38. On seed 0, alan_wake (122 → 112), max_payne (68 → 38) and hadrian (76 → 53) also earned less after hand 50.
- **Overall.** Most of the measurable gain from tracking is against loose bettors and opponents with fixed folding habits. Against the top three opponents it is within noise.

## Final choice

**Deployed: v4.** It has the best held-out weighted score: **+72.59** on seed 1000, against +71.64 for v5. It is simpler than v5 by one parameter and two lines, and v5 lost to it head to head. v5's slightly higher seed-0 score (+78.16 against +76.21) is inside the noise and did not hold on the held-out seed.

**Held-out results for v4 (seed 1000, 64 × 250 per opponent), weighted score +72.59:**
{v4_s1000}

v4 wins against all twelve benchmark opponents on both seeds. For reference, the phase-3 scores are windrunner_glm +69.9 and invoker_gpt +52.0. Those were computed against eleven opponents rather than twelve, so the comparison is approximate.

**Known weaknesses:**
- **Weak against the top opponents.** v4 beats windrunner (+17 ± 7) and meepo (+9 ± 12) only narrowly. Its edge comes mostly from the weaker and wilder bots.
- **No retuning.** None of the inherited baseline parameters were retuned in the hour, and the tracking parameters (priors and clip ranges) were set by reasoning, not tuned.
- **Possible exploits by an adapting opponent:**
  - *Fold-rate scaling:* an opponent can fold a lot early to make v4 bluff up to 1.6× more, then switch to calling.
  - *Showdown statistic:* it moves only on showdowns. An opponent that bets strong hands early, so that it rarely loses at showdown, pushes v4 toward deeper folding (discount down to 0.475) and can then bluff more freely. With 6 pseudo-showdowns and the 1.25 cap this takes several showdowns and is bounded, but it is real.
  - *Fixed bet size:* v4 always bets 0.65 pot postflop, so the size never reveals the hand. However, its bluff frequency is predictable from its own tracked state.
- **Biased showdown sample.** The loss rate counts only showdowns we reached, so it is measured against our calling range rather than a random one. The 0.35 prior was chosen with that in mind, but it was not calibrated.
