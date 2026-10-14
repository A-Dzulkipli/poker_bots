# meepo_claude

| | |
|---|---|
| Bot name | meepo_claude |
| Lineage | phase_3_dota/meepo_claude |
| Model | Claude Opus 5.5 (`claude-opus-5-5`; the serving model may differ) |
| Prompt version | botgen-v3 |
| Engine | `02abc49` + working tree, engine_sha256 `295477eff7637056` |
| Wall-clock time used | about 45 minutes (17:15 to 18:00 UTC, timed from when the environment issue was resolved; ~5 min earlier went to the failed first check) |
| Code executed | **Yes**, but in a substitute Python environment; see "Environment issue" |
| Deployed | **v5** (`__init__.py`: `from .v5 import Bot as DeployedBot`) |

`bench.py check` output (in the substitute environment):
```
python      /sessions/.../venv/bin/python (3.13.15)
jax         0.11.2 (cpu)
poker_bots  /sessions/.../mnt/poker_bots/src/poker_bots
engine      sha256 295477eff7637056 (matches the contract)
smoke test  random vs aggressor, 40 hands in 2s: OK
```

## Environment issue (deviation from the protocol, read this)

The session ran in Claude desktop, whose shell is a **Linux VM** on the requester's Mac that sees only the connected folders.
1. The first `uv run python bench.py check` (with only the bot folder mounted) failed with:
   `FAILED: No module named 'jax'. This shell cannot reach the project's Python environment...` I stopped and reported this, as the prompt requires.
2. The requester then mounted the repository root. `uv run` now found the project, but the project `.venv` is a macOS build that cannot run on Linux. uv tried to recreate `.venv` and was **blocked** ("failed to remove `.venv/.lock`: Operation not permitted") because deletes are not allowed in connected folders, so nothing in `.venv` was removed. I did not retry.
3. With the requester's explicit approval, I built a **separate Linux venv in the VM's home directory, outside the repository**: Python 3.13.15, `jax==0.11.2` (the version pinned in `uv.lock`) and its dependencies from pip. `poker_bots` was imported from the repository's `src/` via `PYTHONPATH` and was not installed. `bench.py check` confirms the engine hash matches the contract. This breaks the "do not install packages" rule. It was done at the requester's direction, and the repository and its environment were not modified.
4. Outside the bot directory I read only the repository root listing and `pyproject.toml`/`uv.lock` (to pin jax). I did not read the engine source, other bots or git state.
5. Background benchmark jobs are killed when a VM shell call ends, so every benchmark ran in the foreground (at most 180 s per call). A few runs that were killed are not reported. Some runs shared the 4 CPUs, which affects their timings but not their results.

Results should match a native run of the same engine and jax version, but these numbers came from Linux/CPU jax 0.11.2, not the project's own macOS venv.

## Scratch files
- `logs/*.txt`: raw benchmark output for every run reported below.
- `count_lines.py`: counts decision-code lines (non-blank, non-comment lines of `act` and helper methods; excludes `__init__`, `init_state`, `metadata` and docstrings).

## Design shared by all versions

Every decision draws 192 Monte Carlo samples of equity against a uniformly random hand (`equity` helper). The opponent's bets and raises **in this hand** narrow their range. Each one removes a fixed fraction of the weakest part of the range, and equity is re-normalised over what remains: `adjusted = (raw − (1 − remaining)) / remaining`. Code is grouped into helpers by situation: `range_equity`, `button_open`, `preflop_response`, `postflop_unopened` and `postflop_facing_bet`.

The decision code is about 40 lines, well under the ~100-line allowance. I spent the hour on benchmarking single ideas rather than on adding more rules.

---

## v1: equity-versus-range baseline
**What it does:** The button opens 2.5bb with equity ≥ 0.42 against a random hand. Facing a limp or raise, it re-raises to 3× with adjusted equity ≥ 0.55 (limp) or ≥ 0.62 (raise), otherwise calls on pot odds + 0.02. Postflop it bets 0.66 pot with adjusted equity ≥ 0.64 and otherwise checks; facing a bet it raises 0.66 pot with ≥ 0.78 and otherwise calls or folds on pot odds.
**Why it exists:** a sound value-only baseline, similar in spirit to the phase-2 finals.
**Parameters:** num_samples 192 (within the 256 cap); open_equity 0.42; preflop_raise_equity 0.62; limp_raise_equity 0.55; value_bet_equity 0.64; value_raise_equity 0.78 (equity probabilities); street_raise_discount 0.25 and earlier_raise_discount 0.12 (range fraction removed per opponent raise, on this street / earlier streets); call_margin 0.02 (equity probability); open_size_bb 2.5 (bb); reraise_multiple 3.0 (× current bet); bet_pot_fraction 0.66 (pot fraction). All are defaults chosen by reasoning, close to the phase-2 finals' values.

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| aggressor | 0 | 64 × 250 | −37.27 | 4.11 |
| bassander_lin_glm | 0 | 64 × 250 | +72.10 | 14.00 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | −13.61 | 6.38 |
| valka_onderra_claude | 0 | 64 × 250 | +23.38 | 12.82 |
| **core mean** | 0 | | **+11.15** | |

**Verdict:** kept as parent. It never bluffs, so `aggressor` beats it.
**Line count:** 35.

## v2: bluffs when checked to (parent v1)
**What it does:** v1, plus: when checking is free and adjusted equity is < 0.35, it bets 0.66 pot as a bluff with probability 0.35 (drawn from `key`). Medium hands still check.
**Why it exists:** v1 gave up every pot it could not win at showdown. The phase-2 bots fold over half the time to bets, and a 2/3-pot bluff needs 40% folds.
**Parameters:** v1's, plus bluff_max_equity 0.35 (equity probability) and bluff_frequency 0.35 (probability). Both chosen by reasoning: bluffs plus value bets give roughly the 1:2 bluff:value ratio a 2/3-pot bet calls for.

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| aggressor | 0 | 64 × 250 | −27.63 | 4.89 |
| bassander_lin_glm | 0 | 64 × 250 | +65.66 | 15.41 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | −5.83 | 7.26 |
| valka_onderra_claude | 0 | 64 × 250 | +32.30 | 15.21 |
| **core mean** | 0 | | **+16.13** | |

**Verdict:** kept as parent. Small gains against aggressor and hadrian.
**Line count:** 37.

## v3: smaller range discount (parent v1)
**What it does:** v1 with street_raise_discount 0.25 → 0.15 and earlier_raise_discount 0.12 → 0.06, applied to preflop and postflop raises alike.
**Why it exists:** hypothesis that v1 over-folds to bets, which loses to `aggressor`.
**Parameters:** as v1 except the two discounts above.

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| aggressor | 0 | 64 × 250 | −43.51 | 6.28 |
| bassander_lin_glm | 0 | 64 × 250 | +51.49 | 21.61 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | −47.52 | 9.29 |
| valka_onderra_claude | 0 | — | not run (killed background job; dropped once v3 was clearly worse) | |

**Verdict:** discarded. Calling more everywhere lost heavily to hadrian (−48 vs −14) and did not help against aggressor.
**Line count:** 35.

## v4: bluff raises facing a bet (parent v2)
**What it does:** v2, plus: facing the first bet of a street with adjusted equity < 0.35, it raises 0.66 pot as a bluff with probability 0.15.
**Why it exists:** to punish opponents who bet often and fold to raises.
**Parameters:** v2's plus bluff_raise_frequency 0.15 (probability), by reasoning.

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| aggressor | 0 | 64 × 250 | −15.88 | 6.30 |
| bassander_lin_glm | 0 | 64 × 250 | +27.22 | 17.45 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | −23.92 | 8.63 |
| valka_onderra_claude | 0 | 64 × 250 | −6.82 | 16.52 |
| **core mean** | 0 | | **−4.85** | |

**Verdict:** discarded. It helped against aggressor (+12) but raising into value bettors cost more against all three phase-2 finals.
**Line count:** 41.

## v5: smaller preflop range discount (parent v2), **deployed**
**What it does:** v2, but an opponent's preflop raise removes only 10% of their range, for the rest of the hand. Postflop raises keep v1's 0.25 (this street) and 0.12 (earlier streets). The big blind therefore defends more against opens, and the bot 3-bets a little more.
**Why it exists:** in v1/v2 a single 2.5bb open removed 25% of the range, so the big blind needed about 0.49 raw equity to call 3 more into 7 and folded about half its hands. Heads-up opens are wide (aggressor opens 100%, the phase-2 finals about 60–75%), so that was over-folding. Unlike v3, this leaves the postflop discount alone.
**Parameters:** v2's plus preflop_raise_discount 0.10 (range fraction per preflop raise), by reasoning, not tuned.

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| aggressor | 0 | 64 × 250 | −28.84 | 5.35 |
| bassander_lin_glm | 0 | 64 × 250 | +164.78 | 17.86 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | +0.18 | 8.91 |
| valka_onderra_claude | 0 | 64 × 250 | +33.07 | 16.53 |
| **core mean** | 0 | | **+42.30** | |
| aggressor | 1 | 64 × 250 | −18.59 | 5.96 |
| bassander_lin_glm | 1 | 64 × 250 | +81.83 | 20.24 |
| hadrian_marlowe_gpt | 1 | 64 × 250 | +1.51 | 9.55 |
| valka_onderra_claude | 1 | 64 × 250 | +58.57 | 13.49 |
| **core mean** | 1 | | **+30.83** | |
| v2:Bot (parent) | 2 | 64 × 250 | −10.38 | 12.47 |

**Verdict:** kept; it became the parent of v6–v9, none of which beat it. Most of the gain is against bassander, which over-folds to it. Hadrian goes from −6/−14 to about 0.
**Line count:** 40.

## v6: higher value-bet threshold (parent v5)
**What it does:** v5 with value_bet_equity 0.64 → 0.72.
**Why it exists:** opponents call with stronger-than-random hands, so thin value bets may be called mostly by better hands.

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| aggressor | 0 | 64 × 250 | −19.17 | 7.60 |
| bassander_lin_glm | 0 | 64 × 250 | +120.21 | 21.26 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | −12.10 | 9.25 |
| valka_onderra_claude | 0 | 64 × 250 | +36.38 | 14.23 |
| **core mean** | 0 | | **+31.33** | |

**Verdict:** discarded. No better than v5 (−11 on the mean, within noise).
**Line count:** 40.

## v7: smaller postflop discount (parent v5)
**What it does:** v5 with street_raise_discount 0.25 → 0.18 (folds less to postflop bets).
**Why it exists:** retests v3's idea on the v5 base, now without changing preflop.

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| aggressor | 0 | 64 × 250 | −26.64 | 6.18 |
| bassander_lin_glm | 0 | 64 × 250 | +151.39 | 17.21 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | −23.75 | 9.65 |
| valka_onderra_claude | 0 | 64 × 250 | +10.50 | 15.03 |
| **core mean** | 0 | | **+27.88** | |

**Verdict:** discarded. Calling more postflop again costs against hadrian and does not help against aggressor.
**Line count:** 40.

## v8: wider button opens (parent v5)
**What it does:** v5 with open_equity 0.42 → 0.34 (opens about 95% of buttons instead of about 75%).
**Why it exists:** folding the button forfeits the small blind, and big blinds that over-fold make even trash opens profitable.

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| aggressor | 0 | 64 × 250 | −17.44 | 5.50 |
| bassander_lin_glm | 0 | 64 × 250 | +131.36 | 18.46 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | −12.38 | 10.37 |
| valka_onderra_claude | 0 | 64 × 250 | +6.13 | 15.68 |
| **core mean** | 0 | | **+26.91** | |

**Verdict:** discarded. Better against aggressor (+11) but worse elsewhere, so no mean gain.
**Line count:** 40.

## v9: more bluffs when checked to (parent v5), finalist
**What it does:** v5 with bluff_frequency 0.35 → 0.55.
**Why it exists:** value-only opponents fold to bets more than the 40% a 2/3-pot bluff needs.

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| aggressor | 0 | 64 × 250 | −17.25 | 6.05 |
| bassander_lin_glm | 0 | 64 × 250 | +125.24 | 17.96 |
| hadrian_marlowe_gpt | 0 | 64 × 250 | −0.90 | 9.06 |
| valka_onderra_claude | 0 | 64 × 250 | +29.14 | 16.20 |
| **core mean** | 0 | | **+34.06** | |
| aggressor | 1 | 64 × 250 | −18.62 | 6.55 |
| bassander_lin_glm | 1 | 64 × 250 | +97.05 | 21.11 |
| hadrian_marlowe_gpt | 1 | 64 × 250 | −7.14 | 10.06 |
| v5:Bot (parent) | 2 | 64 × 250 | −7.67 | 13.46 |

(The seed-1 valka run was cut off by the shell time limit.)
**Verdict:** finalist; lost to v5 on held-out (below).
**Line count:** 40.

---

## Final choice: v5

Held-out seed 1000, 64 tables × 250 hands per opponent:

| opponent | v5 bb/100 (SE) | v9 bb/100 (SE) |
|---|---|---|
| aggressor | −27.79 (5.58) | −16.81 (5.42) |
| bassander_lin_glm | +89.08 (22.05) | +68.12 (23.27) |
| hadrian_marlowe_gpt | −15.64 (10.58) | −20.90 (9.84) |
| valka_onderra_claude | +35.15 (18.75) | +18.04 (16.11) |
| **core mean** | **+20.20** | **+12.11** |
| alan_wake_claude | +84.31 (12.54) | +59.79 (14.13) |
| jack_joyce_gemini | +293.77 (18.72) | +250.56 (23.42) |
| jesse_faden_glm | +249.49 (24.26) | +222.86 (21.70) |
| max_payne_gpt | +61.83 (10.67) | +57.45 (10.48) |
| **legacy mean** | **+172.35** | **+147.67** |
| random | +474.58 (14.68) | +449.95 (14.98) |

**Why v5:** it scores higher on held-out core (+20.2 vs +12.1) and legacy (+172 vs +148), it was best on seed 0, and v9 vs v5 head-to-head is −7.7 ± 13.5 (no evidence v9 is stronger). The two differ by one number, so clarity is the same. The differences between v5 and v9 are mostly within 2 SE, so the choice is close.

**Mean core score across the seeds v5 was run on:** +42.3 (seed 0), +30.8 (seed 1), +20.2 (held-out 1000). Seed-to-seed variance is large, mostly from bassander (SE about 20).

**Known weaknesses:**
- **Loses to `aggressor`** (−19 to −29 bb/100 on every seed). Calling more (v3, v7) and bluff-raising (v4) did not fix it without losing more to the value bettors. Opening wider (v8) and bluffing more (v9) each recovered about 10 bb/100 against aggressor but not on the mean. An opponent that bets every time it is checked to still profits against the fixed range discount.
- **About break-even to slightly negative against hadrian_marlowe_gpt** (+0.2, +1.5, −15.6 across seeds).
- **Warning sign:** v5 vs its parent v2 is −10.4 ± 12.5 (within noise, not significant). The gain comes from exploiting over-folding by the benchmark bots, which other phase-3 bots may not share.
- The range model is crude: a fixed fraction removed per raise, regardless of bet size or board texture. Bluffs pick hands only by low equity and ignore draws and blockers. Bet size is fixed at 0.66 pot.
- Uses about 40 of the ~100 allowed lines. The unused budget could go into bet-size-aware discounts or semi-bluffs with draws.
