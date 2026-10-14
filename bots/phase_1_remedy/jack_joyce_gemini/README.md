# jack_joyce_gemini

**Lineage:** phase_1_remedy/jack_joyce_gemini
**Model:** Gemini-2.5-Pro
**Prompt Version:** botgen-v1
**Engine Commit:** 02abc49
**Wall-Clock Time Used:** 11 minutes
**Status:** Gemini generated this code in a browser and never executed it. Claude later ran it unchanged (see *Repair log* and *Benchmark results* below). Result cells in the version sections were `not run` in Gemini's original and have been filled in by Claude.

---

## v1
* **What it does:** Uses a simple baseline expected value approach, calling when equity > pot odds + a 3% margin, and raising 75% of the pot when equity is strong.
* **Why it exists:** To establish a sound, mathematically grounded baseline against RandomBot.
* **Parameters:** `raise_equity=0.65` (equity threshold for raising), `call_margin=0.03` (buffer against pot odds calls), `bet_fraction=0.75` (sizing).
* **Results:**
  | Opponent | Seed | Tables x Hands | bb/100 | SE |
  |----------|------|----------------|--------|----|
  | random   | 0    | 32 x 250       | +599.94 | 29.45 |
* **Verdict:** Kept as parent of v2. Solid starting heuristic.
* **Decision code line count:** 5

---

## v2
* **What it does:** Differentiates between preflop and postflop environments. It demands a significantly higher equity threshold to raise preflop, relaxing the requirement postflop.
* **Why it exists:** Equity preflop is highly unrealized. A flat threshold of 0.65 is too low for preflop (leading to overplaying weak hands) but potentially too high on the turn/river where hand values are crystallized.
* **Parameters:** `raise_eq_pre=0.70` (higher standard preflop), `raise_eq_post=0.60` (lower standard postflop), `bet_fraction=0.75` (carried over).
* **Results:**
  | Opponent | Seed | Tables x Hands | bb/100 | SE |
  |----------|------|----------------|--------|----|
  | random   | 0    | 32 x 250       | +689.65 | 34.27 |
  | v1       | 0    | 32 x 250       | -66.85  | 22.72 |
* **Verdict:** Kept as parent of v3. Adding street awareness stops the bot from wildly overplaying marginal hands preflop.
* **Decision code line count:** 7

---

## v3
* **What it does:** Introduces dynamic sizing and a randomized bluffing mechanic. It sizes up to full pot for massive equity (value) and sizes down to half-pot otherwise. It also bluffs 10% of the time if the pot is un-raised to capitalize on dead money.
* **Why it exists:** v2 is highly exploitable as its bet sizing communicates exactly one meaning. v3 attempts to extract max value from nutted hands while weaving in light aggression when checked to.
* **Parameters:** `raise_eq=0.60` (standard raise threshold), `massive_eq=0.80` (value threshold), `bluff_prob=0.10` (pure bluffs in un-raised pots).
* **Results:**
  | Opponent | Seed | Tables x Hands | bb/100 | SE |
  |----------|------|----------------|--------|----|
  | random   | 0    | 32 x 250       | +723.19 | 34.38 |
  | v1       | 0    | 32 x 250       | -79.79  | 21.84 |
  | v2       | 0    | 32 x 250       | -3.86   | 23.93 |
* **Verdict:** Selected for deployment. Dynamic sizing and bluffing should meaningfully improve win rate by picking up orphaned pots and maximizing nut hands.
* **Decision code line count:** 10

---

## Final Choice
**Deployed version:** v3
**Held-out result (seed 1000, run by Claude):** vs random +751.75 (SE 35.74), vs v1 -61.28 (SE 21.76), vs v2 -18.30 (SE 23.15) bb/100

**Reasoning:** v3 maintains the sound mathematical foundation of its predecessors but adds the structural complexity (variable sizing, bluffing) necessary for a bot to perform decently against non-random opponents. 
**Known weaknesses:** Because the bot has no opponent modeling and bluffs using a simple RNG check, it may bluff into trapping players or miss obvious exploits against folding stations. It also evaluates "massive equity" statically, ignoring board texture wetness.

---

## Repair log (Claude, 2026-10-13)

Scope: make the minimum fixes needed for the code to import, jit and play under the engine at commit `02abc49`, without changing Gemini's strategy or parameters.

**Fixes applied: none were needed.** All three versions imported, compiled under `jit`/`vmap` and played full benchmark runs exactly as Gemini wrote them. No file in this directory was modified other than this README and the added `*_seed*.txt` benchmark logs.

What was checked:
- `python bench.py vN:Bot random --tables 2 --hands 20` smoke test passes for v1, v2 and v3.
- `equity.py` and `bench.py` match the prompt's verbatim copies (only difference: no trailing newline at end of file).
- Engine fields the bots rely on (`can_raise`, `current_bet`, `pot`, `contributed`, `Round.PRE_FLOP`, `raise_lower_bound`/`raise_upper_bound`) exist with the meanings the bots assume. `obs.pot` already includes this street's bets, so `pot + current_bet - contributed[0]` is the pot after calling, i.e. the sizing formula is a genuine pot-fraction raise.
- No persistent state (`init_state` returns `None`), randomness only from `key`, 128 Monte Carlo samples per decision (under the 256 cap).

Observations against the prompt, not fixed because they are strategy/spec choices rather than code errors:
- Decision code is well under the ~20-line guide (v1 5, v2 7, v3 9 by the prompt's counting rule; Gemini's README says 10 for v3).
- v2 changes two ideas at once: it adds street-dependent raise thresholds and also drops v1's `call_margin` (calls at bare pot odds). v3 likewise drops the preflop/postflop split from v2 while adding sizing and bluffing. This makes the per-version results hard to attribute.
- v3's bluff condition `obs.current_bet == 0` can only fire postflop on an unopened street (preflop the current bet is always at least the big blind), which matches its docstring.

## Benchmark results (run by Claude after repair)

Run in a Linux cloud container (Python 3.13, `jax` 0.11.2, CPU) on a copy of the repo at engine commit `02abc49`, because the project `.venv` is macOS-only. Code was the files in this directory, unmodified. Command: `python bench.py <candidate> <opponent> --seed <s>` with defaults (32 tables x 250 hands, 6-max, 24,000 seat-hands per side). Raw output is in `<candidate>_<opponent>_seed<s>.txt`.

| Candidate | Opponent | Seed | bb/100 | SE |
|-----------|----------|------|--------|----|
| v1 | random | 0 | +599.94 | 29.45 |
| v2 | random | 0 | +689.65 | 34.27 |
| v2 | v1 | 0 | -66.85 | 22.72 |
| v3 | random | 0 | +723.19 | 34.38 |
| v3 | v1 | 0 | -79.79 | 21.84 |
| v3 | v2 | 0 | -3.86 | 23.93 |
| **v3** | random | **1000** | +751.75 | 35.74 |
| **v3** | v1 | **1000** | -61.28 | 21.76 |
| **v3** | v2 | **1000** | -18.30 | 23.15 |
| v1 | random | 1000 | +592.06 | 29.55 |
| v2 | v1 | 1000 | -56.93 | 21.38 |

The last two rows are extra held-out runs for v1, added by Claude because the seed 0 results made v1 a finalist; Gemini only nominated v3.

**Reading (Claude's, not Gemini's):** Head-to-head, v1 beats both later versions on seed 0 and on held-out seed 1000, by roughly 2.5 to 3.7 SE each time. v3 vs v2 is inside the noise on both seeds. v2 and v3 beat RandomBot by more than v1 does, but the prompt says to weight results against earlier versions above results against RandomBot, so by the prompt's own protocol **v1 is the strongest version and v3 is the wrong deployment choice.** The likely cause is that v2 and v3 dropped v1's 3% `call_margin` and call at bare pot odds against uniformly random hands, which over-calls against an opponent whose betting range is stronger than random. `__init__.py` still deploys v3, as Gemini chose; that has not been changed.
