# valka_onderra_claude

> **I could not execute code myself.** The linked shell only had this bot directory mounted, so
> `uv run python -c "import poker_bots"` failed (`ModuleNotFoundError`). v1–v5 were written by reasoning.
> Afterwards the requester ran `bench.py` on the machine with the project environment and pasted the
> output; those numbers are reported verbatim below. Fields still marked `not run` were not run.

| Field | Value |
|---|---|
| Bot name | `valka_onderra_claude` |
| Lineage | phase_2_suneater/valka_onderra_claude |
| Model | Claude Opus 5.5 (`claude-opus-5-5`) |
| Prompt version | botgen-v2 |
| Engine | `02abc49` + working tree, engine_sha256 `295477eff7637056` |
| Wall-clock time used | ≈50 minutes (15:41–16:31 UTC, 2026-10-13); v1–v5 by 15:50, v6 at 16:17 after the requester's seed-0 benchmarks |
| Code executed against engine | Not by me; seed-0/2 benchmarks run by the requester (see results) |
| Deployed | **v5** (`__init__.py`: `from .v5 import Bot as DeployedBot`) |

## What was checked without the engine

These checks ran in a separate sandbox, outside this directory. They are **not** benchmarks.

- **Trace/compile check.** Every version was run through `jax.jit(Bot().act)` with jax 0.11.2 against a
  stub `poker_bots` built from the documented contract (field names, shapes, dtypes) and a fake hand
  evaluator. All five compile; `act` returns `ActionData` with scalar `int32` fields and returns `state`
  (`None`) unchanged; `metadata()` is JSON-serialisable; no version defines `observe`.
- **Unit check.** With equity pinned to fixed values, v1 and v5 produce the intended chip amounts: button
  open to 5 chips (2.5bb), BB 3-bet vs a 2.5bb open to 15 chips, limp isolation to 6 chips, flop bets in
  pot fractions, and in v5 a turn bet that would leave a stub becomes all-in (50 chips, not 40).
- **Threshold calibration** using an external Python evaluator (`treys`), random opponent hands:
  - Preflop equity vs random ≥ 0.40: 84% of hands. ≥ 0.50: 48%. ≥ 0.60: 16%. ≥ 0.64: 6%. ≥ 0.68: 3%.
  - Flop K♦8♣3♥, equity vs random → with `missed_board_weight` 0.35 after an opponent bet:
    AK 0.89→0.85, K9 0.84→0.79, 87s (middle pair) 0.69→0.60, 55 0.60→0.46, AQ (no pair) 0.56→0.39, JT 0.42→0.32.

The scripts were not saved to this directory. There are no scratch files here.

## Opponents read

All four phase-1 finals were read. They all play off Monte Carlo equity against a random hand: call when
equity beats pot odds (plus a small margin), and raise above a fixed equity (0.60–0.75). max_payne_gpt
folds preflop unless raw equity ≥ ~0.68, so it folds most hands to a raise. Their bets and raises therefore mean
pair-or-better far more often than random, and they call bets by pot odds against random hands, which
means they call bets with too many hands. No version detects or keys on any specific opponent. Borrowed idea: the per-raise equity
discount (alan_wake_claude's `raise_discount`).

---

## v1: baseline

**What it does:** Measures equity against a random hand (128 samples) and shrinks it by 12% for every
raise the opponent has made this hand. Preflop the button opens 2.5bb with equity ≥ 0.40 (≈84% of hands)
and folds the rest. Otherwise it re-raises to 3× the current bet with adjusted equity ≥ 0.60 and calls
when adjusted equity beats pot odds by 0.02. Postflop it bets or raises call + 0.66 pot with adjusted equity
≥ 0.66, calls by pot odds, and otherwise checks or folds.

**Why it exists:** It is a sound heads-up baseline: it plays a wide button range because folding only costs the
small blind, defends the BB by price, and makes no bluffs.

**Parameters:**

| Name | Value | Unit | How chosen |
|---|---|---|---|
| num_samples | 128 | samples | half the 256 cap |
| open_equity | 0.40 | equity prob. | calibrated: opens ≈84% of buttons |
| preflop_raise_equity | 0.60 | equity prob. (adjusted) | reasoning |
| postflop_bet_equity | 0.66 | equity prob. (adjusted) | reasoning: middle pair or better on a dry flop |
| call_margin | 0.02 | equity prob. | reasoning |
| raise_discount | 0.12 | fraction of equity per opponent raise | borrowed from alan_wake_claude (0.15), softened |
| open_size_bb | 2.5 | big blinds | standard HU open |
| reraise_multiple | 3.0 | × current bet | standard |
| bet_pot_fraction | 0.66 | pot fraction | standard |

**Results:**

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| max_payne_gpt | 0 | 64 × 250 | +34.60 | 8.41 |
| other three finals | 0 | 64 × 250 | not reported (run, but output not pasted) | — |
| mean vs finals | 0 | 64 × 250 each | +128.75 | — |
| random | not run | not run | not run | not run |

**Verdict:** Kept as the parent of v2, by reasoning. Leaks found on review: (1) the flat discount counts the preflop
open as a postflop raise as well, and it treats every hand alike, so a facing-bet decision for top pair
(0.84 → 0.65 after open + bet) is cut nearly as hard as for ace-high. (2) Facing a 2.5bb open, the 3-bet needs raw equity ≥ 0.68,
which is ≈3% of hands. That is far too narrow heads-up.

**Line count:** 18. This is well under the ~50 budget. It was kept minimal on purpose so later changes are attributable.

## v2: board-aware range after postflop aggression (parent v1)

**What it does:** Same as v1 preflop. Postflop it uses its own Monte Carlo equity (`range_equity`, 128
deals). Once the opponent has bet or raised on any postflop street, the opponent's sampled hands
that missed the board count only 0.35 as much. Missed means no pocket pair and no hole card pairing a board card. The flat 12% discount now counts only the opponent's
preflop raises.

**Why it exists:** It targets v1 leak (1). A postflop bettor mostly holds hands that connected. Re-weighting the
range cuts weak holdings (ace-high 0.56→0.39, underpair 0.60→0.46) while barely touching strong ones
(top pair 0.84→0.79), unlike a flat multiplier, which cuts everything by the same factor.

**Parameters:** v1's set, plus `missed_board_weight` = 0.35 (relative likelihood a postflop bettor missed
the board). It was chosen by reasoning and checked with the calibration above: it moves middle pair to ~0.60, still a call vs half
pot, and ace-high to ~0.39.

**Results:** not run (all fields, all opponents).

**Verdict:** Kept as the parent of v3, by reasoning. Known blind spot: made straights and flushes with no pair count as
missed, so equity is slightly overstated on boards where those complete. On paired boards the filter is weaker.

**Line count:** 40 (`act` + `range_equity`).

## v3: equity-scaled bet size and commitment shove (parent v2)

**What it does:** Postflop value bets grow linearly from call + 0.5 pot at the bet threshold to call + 1.0
pot at equity 1.0. If a bet would use ≥ 45% of our remaining chips (this street's all-in total), it is made all-in instead.

**Why it exists:** All four benchmarks, and probably most equity-based phase-2 bots, call by pot odds
against a random hand, so they call large bets with too many hands. Bigger bets with our strongest hands earn more from
them. The shove removes small stubs that leave no fold equity and still commit us (v1 example: bet 40 of 50
on the turn).

**Parameters:** `bet_pot_fraction` replaced by `min_bet_pot_fraction` 0.5 and `max_bet_pot_fraction` 1.0 (pot
fractions), and `shove_commit_fraction` 0.45 (share of remaining chips). All by reasoning. The 0.45 is close to
alan_wake_claude's 0.40 commit rule.

**Results:** not run.

**Verdict:** Kept as the parent of v4, by reasoning. Risk: it stacks off with middle-pair-strength hands (adjusted equity ≥ 0.66)
when the stack-to-pot ratio is low.

**Line count:** 46.

## v4: re-raise threshold rises within a street (parent v3)

**What it does:** Postflop, the equity needed to bet or raise is 0.66 plus 0.08 for every bet or raise
already made on this street. So betting needs 0.66, raising a bet 0.74, and 3-betting 0.82.

**Why it exists:** It targets raise wars. In v1–v3 any hand above 0.66 re-raises indefinitely, so top pair keeps raising
into sets. The v2 range weighting does not escalate with the number of raises.

**Parameters:** v3's set, plus `reraise_equity_step` = 0.08 (equity per prior raise this street), by reasoning.

**Results:** not run.

**Verdict:** Kept as the parent of v5, by reasoning.

**Line count:** 48.

## v5: wider preflop re-raise range (parent v4)

**What it does:** It is identical to v4 except `preflop_raise_equity` is 0.55 instead of 0.60. Facing an open, it 3-bets with
raw equity ≥ ~0.625 (≈8% of hands, up from ≈3%). It raises limps with ≥ 0.55 (≈30%).

**Why it exists:** It targets v1 leak (2). A 3% 3-bet range lets openers see cheap flops, and tight-preflop opponents
(like max_payne_gpt, which folds most hands preflop) fold to re-raises. This is a single-parameter change.

**Parameters:** as v4, with preflop_raise_equity = 0.55 (adjusted equity prob.), by calibration above.

**Results (run by the requester):**

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| alan_wake_claude | 0 | 64 × 250 | +79.39 | 12.75 |
| jack_joyce_gemini | 0 | 64 × 250 | +288.83 | 26.28 |
| jesse_faden_glm | 0 | 64 × 250 | +289.31 | 25.38 |
| max_payne_gpt | 0 | 64 × 250 | +37.58 | 10.49 |
| mean vs finals | 0 | 64 × 250 each | +173.78 | ≈10 (from per-opponent SEs) |
| v1 (root) | 2 | 64 × 250 | +51.42 | 13.87 |

v5 beats v1 head to head by 3.7 SE and beats v1's finals mean by 45 bb/100 (about 3 SE). v2–v4 were not
benchmarked individually, so the gain cannot be split among v2–v5.

**Verdict:** Kept as the parent of v6. The weakest result is vs max_payne_gpt (+37.6; alan_wake_claude
gets +76 there). Diagnosis: max opens only ≈16% of buttons, while v1–v5 defend the big blind with raw
equity ≥ 0.364, ≈92% of hands. That is too loose against any opener; sound heads-up defence is ≈65–75%.

**Line count:** 48.

## v6: big blind discounts equity for playing out of position (parent v5)

**What it does:** It is identical to v5, except that preflop calls from the big blind require
`equity × oop_realization` (0.85) to beat pot odds + margin. Re-raise decisions and the button are unchanged.

**Why it exists:** It targets the v5 diagnosis above. The big blind acts first on every later street, so it
keeps less of its raw equity. Against a 2.5bb open, defence tightens from raw equity ≥ 0.364 (≈92% of hands) to ≥ 0.43 (≈72%).
This is a general positional principle; it does not detect any particular opponent.

**Parameters:** v5's set, plus `oop_realization` = 0.85 (share of equity realized out of position), by reasoning.

**Results (run by the requester; only these lines were pasted):**

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| alan_wake_claude | 1000 | 64 × 250 | +82.15 | 14.40 |
| jack_joyce_gemini | 1000 | 64 × 250 | +300.68 | 22.83 |
| jesse_faden_glm | 1000 | 64 × 250 | +262.26 | 25.66 |
| max_payne_gpt | 1000 | 64 × 250 | +45.91 | 10.05 |
| mean vs finals | 1000 | 64 × 250 each | +172.75 | ≈9.5 (from per-opponent SEs) |
| random | 1000 | 64 × 250 | +466.23 | 13.98 |
| finals, seed 0 | 0 | 64 × 250 | not reported | — |
| v5 (parent) | 3 | 64 × 250 | not reported | — |

**Verdict:** Discarded. On held-out seed 1000 its finals mean (+172.8) is the same as v5's (+176.0) within noise,
and it is 26 bb/100 lower vs random (≈1.3 SE, not significant). The gain against max_payne_gpt that it was
built for (+45.9 vs v5's +37.6 on seed 0) is within the error bars. A change with no measurable gain does not
replace the parent.

**Line count:** 50.

---

## Final choice

**Deployed: v5.** Finalists v5 and v6 were run on held-out seed 1000. Their finals means are equal within noise
(+176.0 vs +172.8), so the simpler parent stays. v5 also beats the root v1 head to head (+51.4 ± 13.9, seed 2).
The original reasoning: Each step fixes a specific leak found
on review, and none adds bluffing or opponent-specific logic. v5 is also the most complete version inside the
line budget.

**Held-out results (seed 1000):**

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| alan_wake_claude | 1000 | 64 × 250 | not reported | — |
| max_payne_gpt | 1000 | 64 × 250 | not reported | — |
| jesse_faden_glm | 1000 | 64 × 250 | not reported | — |
| jack_joyce_gemini | 1000 | 64 × 250 | not reported | — |
| mean vs finals | 1000 | 64 × 250 each | +176.02 | — |
| random | 1000 | 64 × 250 | +492.23 | 13.82 |
| v1 (root) | 2 | 64 × 250 | +51.42 | 13.87 |

The per-opponent seed-1000 lines for v5 were run but not pasted. The +176.02 mean was pasted without its seed.
It is attributed to seed 1000 because it appeared in the position of the `v5:Bot finals --seed 1000` run.
v4 and v5 were not benchmarked against each other.

To run them on the machine with the project environment, from this directory:

```
uv run python bench.py v5:Bot random --tables 2 --hands 20   # smoke test
uv run python bench.py v5:Bot finals --seed 1000
uv run python bench.py v5:Bot random --seed 1000
uv run python bench.py v5:Bot v1:Bot --seed 2                # a version that loses to its parent is a warning sign
```

If v5 loses to v1, or its finals mean is clearly below v1's, the likely culprits in order are v3's shove
rule (stacking off with medium hands) and v5's 4-bet width (0.55 adjusted equity also governs 4-bets).

**Known weaknesses:**
- **No bluffs or semi-bluffs.** The bot bets only for value, so a thinking opponent could fold to its bets. This
  matters less here because no opponent can keep state across hands.
- **It folds to large bets with weak holdings by design.** An aggressive heads-up bot that over-bluffs big can push it off
  marginal hands. Calls are still priced by pot odds, so it does not fold to small bets.
- **The range filter ignores draws and made straights or flushes without a pair**, and is weaker on paired boards.
- **Preflop equity is noisy.** At 128 samples the standard deviation is ≈0.044, so hands near the thresholds play mixed.
  This is harmless, and is arguably useful as randomisation.
- **Never benchmarked.** The thresholds were set by reasoning and calibration, not tuned against the benchmarks.
