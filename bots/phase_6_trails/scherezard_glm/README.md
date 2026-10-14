# scherezard_glm — phase_6_trails

| | |
|---|---|
| bot name | `scherezard_glm` |
| lineage | phase_6_trails/scherezard_glm |
| model | GLM-5.3 (account:zai-individual-coding-plan) |
| prompt version | botgen-v6 |
| engine | decafb9, engine_sha256 295477eff7637056 |
| wall-clock used | ~105 minutes (17:57–19:42 local). Iteration stopped at minute 40 (18:37); the overage is held-out runs — v4's, then v1's after v4 lost the held-out comparison — plus the exact-deployment ablation and README. |
| code executed | yes — all benchmark numbers below are real runs |
| environment | project `.venv` via `uv run` from the bot directory (no separate environment needed) |

`bench.py check` output:

```
python      /Users/alik/Documents/Projects/poker_bots/poker_bots/.venv/bin/python3 (3.13.12)
jax         0.11.2 (cpu)
poker_bots  /Users/alik/Documents/Projects/poker_bots/poker_bots/src/poker_bots
engine      sha256 295477eff7637056 (matches the contract)
smoke test  random vs aggressor, 40 hands in 2s: OK
```

## Design

The deployed bot (v5 — v1's strategy plus a no-op ablation flag; see v5 below) represents the
opponent's range as a **(52,52) weight matrix over ordered card pairs**. The preflop line fixes
its shape as a soft band on a readable preflop hand-quality score (opens keep strong hands,
limps keep weak ones, flats keep a band capped below what would have 3-bet). Every villain
action after that multiplies the matrix by a likelihood that grows with how strongly the
holding connects with the board and with the bet size in big blinds; checks and calls tilt the
range mildly away from strength. At decision time 80 holdings are sampled ∝ the final weights
(Efraimidis–Spirakis weighted top-k, one batched evaluator call, 81 holdings ≤ the 256 cap) and
the hero's current made hand is compared against them; flop/turn decisions credit flush-draw
and overcard outs arithmetically. Preflop decisions use thresholds on the same quality score,
shifted by the villain's preflop bet sizes; the all-in-call check converts the villain's mean
range quality into an approximate equity. Bluffs are drawn only from `key`, only on lines where
the villain never raised postflop. v1 was chosen over its descendants because it scored best on
the held-out field: the calibrated descendants beat it in spots but lost more to the pure
pressure bot.

## v1 — range bot, first parameter set

*What it does:* the full range pipeline above with initial thresholds (button opens top ~65%,
BB defends top ~55% vs opens), value betting above 0.72 range equity, calling above pot odds
+0.02, bluffing 45% of the time on passive lines.
*Why it exists:* baseline for the whole approach.
*Parameters:* all defaults in `v1.py.__init__`; chosen by reasoning and a preflop-score
percentile calibration (`scratch_calib.py`), not tuned against the field.
*Results* (64 tables × 250 hands, seed 0, `field`):

| opponent | bb/100 | SE | | opponent | bb/100 | SE |
|---|---|---|---|---|---|---|
| kraken_gpt | -40.45 | 10.25 | | meepo_claude | +16.55 | 14.19 |
| leviathan_claude | -16.27 | 12.38 | | hadrian_marlowe_gpt | -1.53 | 9.52 |
| megalodon_glm | -32.36 | 10.89 | | vicious_glm | +55.40 | 18.68 |
| spike_claude | -18.31 | 10.46 | | valka_onderra_claude | -3.94 | 14.94 |
| faye_gpt | -30.06 | 7.53 | | aggressor | -20.64 | 9.27 |
| windrunner_glm | -36.72 | 11.19 | | alan_wake_claude | +57.36 | 13.81 |
| invoker_gpt | -19.97 | 12.91 | | max_payne_gpt | +59.26 | 13.25 |
| jesse_faden_glm | +209.44 | 21.54 | | bassander_lin_glm | +58.73 | 18.53 |
| jack_joyce_gemini | +265.94 | 25.64 | | random | +441.11 | 16.30 |

**Weighted score +12.27** (plain vs non-random +29.56).

*Verdict:* kept as parent of v2 (and, ultimately, deployed as v5 after the descendants lost
the held-out comparison). Its field pattern: heavy losses to the strong half, big wins against
the weak half — the leverage for improvement looked to be calling down pressure and defending
slightly wider.
*Line count:* 103.

## v2 — range-model calibration

*What it does:* one group of range-model fixes: (a) `board_connection` is now computed **per
street** (v1 used the full board for flop/turn likelihoods — the range "saw" future cards);
(b) steeper preflop bands for raised pots (softness 0.12 → 0.06) because the measured 3-bet
range kept 40% of hands (target ≈ 12%); (c) the postflop bet likelihood keeps a real bluff
share (air intercept 0.05 → 0.12) and overcards count a little more.
*Why it exists:* measured range tables (`scratch_ranges.py`) showed a too-wide 3-bet range and
the look-ahead bug; both distort every call decision.
*Parameters:* `band_softness` split (0.06 raise / 0.12 passive), raise intercept 0.12 — chosen
from the measured shares, not the benchmark.
*Results* (32 tables × 250 hands, seed 0, spot runs):

| opponent | bb/100 | SE |
|---|---|---|
| aggressor | -60.83 | 11.13 |
| jesse_faden_glm | +279.24 | 28.17 |
| windrunner_glm | -14.65 | 17.85 |

(no full `field` run for v2 — time box; windrunner improved from v1's -36.72 (64 tables) to
-14.65 (32 tables), but that is only ~1 combined SE — suggestive, not conclusive.)
*Verdict:* kept as parent of v3/v4 — same decision core, better-calibrated opponent model.
*Line count:* 109.

## v3 — loose-defense experiment (discarded)

*What it does:* v2 with much looser thresholds: call margin -0.05 (calls below direct odds),
wider villain betting ranges (air intercept 0.22), wider preflop defense (defend 0.16 ≈ top
85% vs opens, button opens 85%).
*Why it exists:* tests the hypothesis that v1/v2 lose mainly by over-folding to pressure.
*Parameters:* defaults in `v3.py.__init__`.
*Results* (32 tables × 250 hands, seed 0): aggressor -35.98 (SE 14.89, improved from -60.83)
but windrunner_glm **-102.10** (SE 18.72, catastrophic; v2 was -14.65).
*Verdict:* **discarded.** Calling below odds with an inflated opponent range turns the bot
into a calling station against value-heavy barrels. The over-fold hypothesis is only half
right: loosen the preflop defense, not the calling.
*Line count:* 110.

## v4 — calibrated looseness (discarded)

*What it does:* v2's narrow range model with v3's *preflop* loosening only: button opens top
~76% (quality > 0.20), BB defends vs opens from quality > 0.24 (~top 68%), call margin exactly
pot odds (0.00), bluff frequency down to 0.30. Adds the `uniform_range` constructor flag that
replaces the narrowed range with uniform weights (all live holdings equally likely) for the
ablation evidence below — with the flag off it is bit-for-bit v2's decision rule.
*Why it exists:* separates the two knobs v3 had conflated; keeps every change that helped
(aggressor improved) and reverts the one that did not (negative-margin calling).
*Parameters:* `open_min_quality=0.20` (~top 76% of hands by the calibration table),
`defend_min_quality=0.24` (~top 68%), `call_margin=0.00` (break-even calling),
`bluff_frequency=0.30`, `uniform_range=False`. Values chosen from the spot runs above; SEs on
the 32-table spots are ±15–35, so these are directional, not fitted.
*Results* (32 tables × 250 hands, seed 0): aggressor -46.62 (SE 16.98), windrunner_glm -16.50
(SE 14.45), jesse_faden_glm +281.75 (SE 34.26), random +543.75 (smoke). Held-out `field`
(seed 1000, 64×250): **weighted +7.40** (plain +23.96) — kraken -53.65, leviathan -40.73,
megalodon -26.26, spike -60.46, faye -50.83, windrunner -46.32, invoker -12.69, meepo -4.76,
hadrian +9.85, vicious +27.27, valka +19.84, **aggressor -49.93**, alan_wake +32.57,
max_payne +52.27, bassander +21.34, jesse +255.02, jack_joyce +334.85, random +500.02
(full log `v4_field_seed1000.log`).
*Verdict:* **discarded.** On the same held-out seed it scored +7.40 vs v1's +10.39; the
dominant real regression was aggressor (-49.93 vs v1's +1.13, >4 SE apart) — the tighter,
calibrated ranges and looser preflop con too much to pure pressure.
*Line count:* 110.

## v5 — v1 with an ablation switch (deployed)

*What it does:* exactly v1's decision rule (identical parameters, identical code path when the
flag is off) plus a `uniform_range` constructor flag that replaces the narrowed range with
uniform weights, so the range model's contribution can be measured on the deployed strategy.
*Why it exists:* the README requires deployed-vs-uniform evidence, and v1 beat every descendant
on the held-out field, so the deployed version needed the switch without changing behaviour.
*Parameters:* identical to v1; `uniform_range=False` (default off).
*Results:* identity verified twice against v1 — vs random 2×20 seed 0: +473.75 (SE 6.25,
matches v1 to every digit); vs aggressor 64×250 seed 0: -20.64 (SE 9.27, matches v1's field
cell exactly). Therefore v1's field runs are v5's: seed 0 weighted **+12.27**, held-out seed
1000 weighted **+10.39**. Ablation results in the Range model section.
*Verdict:* deployed.
*Line count:* 104.

## Range model (deployed version, v5)

**Representation.** A (52,52) float weight matrix over ordered card pairs; the hero's hole
cards and dealt board cards are masked to zero. Weight 1.0 ≈ "unconditionally in the range".

**Initial shape (preflop).** Each hand has a readable quality score (high card 0.42·hi/12 +
low card 0.18·lo/12 + pair bonus 0.45…0.80 + suited 0.08 − gap penalty up to 0.09). The preflop
line picks a soft band: villain raises → lower bound 0.26 (+0.02/bb above a 2.5bb open) with
softness 0.12; two-plus raises → 0.48; button limp → 0.08 with the top trimmed at 0.62 (they
would have raised premiums); big-blind flat → 0.30 with the same trim. A 3% floor on everything
models slowplays and wild lines. Measured bands: an open keeps the top ~45% by weight, a limp
~71% skewed weak, a BB flat a 44% band capped below the 3-bet hands.

**Narrowing (postflop).** Every villain action multiplies the matrix: a bet/raise of
`size_bb` multiplies weight by `0.05 + (0.30 + 0.07·min(size_bb,12)) · connection`; a call by
`0.25 + 0.55·connection`; a check by `0.90 − 0.35·connection`. `connection` ∈ [0, 1.4] sums
per-card terms — 0.30 for pairing a board rank, 0.90/0.35 for a made/flush-draw suit match,
0.12 per overcard — plus 0.45 for a pocket pair above the board, all computed from the board
dealt so far. Likelihoods compound across streets, so a river range reflects the whole line;
bluff weight decays geometrically (air ≈ 8% of a value hand after one bet, less after two).
One known approximation: the connection features use the full board dealt so far even when
weighting an earlier street's action (v2 corrected this; the correction did not hold out
better, so the deployed version keeps v1's behaviour).

**Decisions.** 80 holdings sampled ∝ weight (weighted top-k over fresh `key` bits); hero's
made hand is compared to each sample's made hand on the exact current board (batched
`evaluate_hands`; 81 holdings ≤ 256 cap). Draw credit is arithmetic: +flush-draw/overcard outs
for the hero, −flush-draw share behind for the villain, scaled by streets left. Bet above 0.72
range equity (size 0.45–1.0 pot, increasing in equity), raise above 0.80, call above pot odds
+0.02, bluff ≤ 0.40 equity at 45% frequency on lines the villain never raised. Preflop uses
quality thresholds against the villain's bet sizes (open 0.26 ≈ top 65%, 3-bet 0.48 ≈ top
12%, BB defend 0.30 ≈ top 55%); all-in calls convert the villain's mean range quality into
approximate equity.

**Measured range table** (`scratch_ranges.py v5`; flop K♠7♥4♣, turn J♣, river 9♦; share =
mean weight over all 1326 hands, 1.0 = fully kept; "top" = highest-weight combos):

| line | weighted share | the range, plainly |
|---|---|---|
| button opens to 3bb | 0.58 | 55%+ wide-aggressive: pairs 77+ and AK at 0.98, tapering through suited connectors |
| button limps | 0.71 | weak-wide: suited connectors (79s), K6-type, premiums trimmed (0.81 max) |
| big blind calls a 3bb open | 0.44 | capped medium: A7s/AJs band, no premiums (would 3-bet), 0.64 max weight |
| big blind 3-bets to 9bb | 0.40 | 88+/99+ and big pairs near 1.0, but the soft band leaks weight down to broadways |
| button c-bets the flop after opening | 0.08 | sets 77/44, top pair (0.35), overpairs; a 5% bluff floor |
| big blind check-raises the flop | 0.07 | K7/K4 two pair, sets behind, AK (0.27), thin bluffs |
| opponent bets turn after flop bet-call | 0.04 | K-x value, clubs (flush draw/arrival) lead, J-x pairs |
| opponent bets all three streets | 0.03 | made/arrived clubs (TcJc, Kc7c) and K-x; air decays to a few % |

**Evidence the model helps.** With `uniform_range=True` the identical strategy decides against
a uniformly random holding (the `equity()` assumption) instead of the narrowed range
(`scratch_ablation.py`; uniform side at 32 tables × 250 hands seed 0, range side from the seed-0
field runs at 64 tables):

| match | v5 with range | v5 uniform |
|---|---|---|
| head-to-head v5(range) vs v5(uniform) | **+41.45** (SE 34.17) | — |
| vs windrunner_glm | -36.72 (SE 11.19) | **-117.58** (SE 20.22) |
| vs jesse_faden_glm | **+209.44** (SE 21.54) | +139.09 (SE 39.45) |
| vs aggressor | -20.64 (SE 9.27) | **+40.44** (SE 19.70) |

The range-off variant loses the head-to-head, collapses against windrunner (whose bets mean
something) and drops ~70 bb/100 against the weak jesse_faden. The one reversal is instructive:
against `aggressor`'s constant pressure, treating its range as uniform — never crediting its
bets with strength — wins more, which is exactly the over-narrowing weakness listed under
Final choice.

## Final choice

**Deployed: v5** (`from .v5 import Bot as DeployedBot`) — v1's decision rule with the range
ablation switch, identity-verified against v1 on the benchmark (two exact reproductions, see
v5 section). The choice was made on held-out seed 1000, run once per finalist: v1/v5 **+10.39**
vs v4 **+7.40**. The best-scoring version is therefore the deployed one; v5 rather than v1
purely so the required uniform-range ablation is a parameter setting of the deployed strategy
itself. The spot improvements of v2/v4 (windrunner -36.7 → -16.5) did not survive the held-out
field: the calibrated-ranges family lost ~50 bb/100 to `aggressor` relative to v1, which at
loss weight 1.26 outweighed everything it gained elsewhere.

Held-out field run for the deployed strategy (seed 1000, 64 tables × 250 hands, = v1's run;
log `v1_field_seed1000.log`):

| opponent | bb/100 | SE | | opponent | bb/100 | SE |
|---|---|---|---|---|---|---|
| kraken_gpt | -40.61 | 10.22 | | vicious_glm | +15.98 | 20.20 |
| leviathan_claude | -13.14 | 13.09 | | valka_onderra_claude | +2.72 | 14.86 |
| megalodon_glm | -23.63 | 11.31 | | aggressor | +1.13 | 9.41 |
| spike_claude | -24.17 | 11.16 | | alan_wake_claude | +53.68 | 12.88 |
| faye_gpt | -23.63 | 8.21 | | max_payne_gpt | +44.12 | 11.11 |
| windrunner_glm | -47.84 | 11.20 | | bassander_lin_glm | +32.36 | 22.46 |
| invoker_gpt | -29.50 | 12.05 | | jesse_faden_glm | +265.29 | 26.26 |
| meepo_claude | -2.13 | 12.12 | | jack_joyce_gemini | +308.98 | 26.18 |
| hadrian_marlowe_gpt | -6.72 | 9.39 | | random | +440.53 | 16.93 |

**Weighted score +10.39** (plain vs non-random +30.17); seed-0 field run: weighted +12.27.

**Known weaknesses.** (1) Loses to the whole top half of the field — the range model has no
notion of how *often* an opponent bluffs, only that bluffs exist, so well-sized mixed barrels
are under-called. (2) Conversely, against relentless pure pressure the narrowed range
over-credits bets: the uniform ablation beat `aggressor` by +40 bb/100 over the deployed
version, the clearest single leak in the model. (3) The flop comparison uses current made
hands plus crude draw credit; straight draws are not modelled at all (neither side's).
(4) The connection features weight earlier streets with the later board already dealt.
(5) Preflop all-in equity from mean range quality is rough. (6) 80 samples give ±5% equity
noise per decision. (7) Bluffing is fixed-frequency rather than EV-driven, and never happens
into a raised pot except at 15% of the bluff rate as a semi-bluff raise.

## Files

`v1.py`–`v5.py` (versions, never edited after benchmarking), `equity.py`/`bench.py` (shared,
untouched), `__init__.py` (deploy line: v5), `scratch_calib.py` (preflop score percentiles),
`scratch_ranges.py` (range table measurement), `scratch_ablation.py` (range on/off evidence),
`v1_field_seed0.log`, `v1_field_seed1000.log`, `v4_field_seed1000.log` (benchmark logs),
`ablation.log` (v4-family ablation), `ablation_v5.log` (deployed-version ablation, reported above).
