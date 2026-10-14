# bassander_lin_glm — phase 2 (heads-up)

- **Bot:** `bassander_lin_glm`, lineage `phase_2_suneater/bassander_lin_glm`, deployed version **v8**
- **Model:** GLM-5.3 (Z.ai individual coding plan)
- **Prompt version:** `botgen-v2` · **Engine:** `02abc49` + working tree, engine_sha256 `295477eff7637056`
- **Wall-clock time used:** ~25 minutes (02:13–02:38 local), well inside the 60-minute box; each `finals` run took ~70 s, so iteration was not time-bound. **Code was executed** — every benchmark number below is a real `bench.py` run on this machine (`uv run python bench.py ...`, 64 tables × 250 hands = 16,000 hands per opponent unless noted).
- **Borrowed ideas:** the per-raise equity discount generalizes `alan_wake_claude`'s `raise_discount` (made multiplicative and stronger); the "call when equity beats pot odds" skeleton is the one all four phase-1 finals share, per the helpers' own docstrings. No opponent-specific keying anywhere.
- **Line counts** are non-blank, non-comment physical lines inside `act` (signature excluded; a wrapped statement counts each line).
- **Scratch files:** none kept; all benchmark outputs are transcribed below.

## Strategy in one paragraph

Every decision starts from Monte Carlo equity against a uniformly random hand (`equity`, 96 samples). That number overstates our chances against a raiser, so when calling it is multiplied by 0.75 for each counted raise this street (the first *preflop* raise is exempt — an open is routine, not a strength signal). Calls require discounted equity ≥ pot odds + 2%. Undiscounted equity drives betting: 0.64+ to bet when checked to, 0.80+ to re-raise a bettor, sized at 0.90 of the pot after calling. Preflop we open every button hand to 2.5bb, raise button limps to 3.5bb with 0.55+ equity, and re-raise to 3× the current bet with 0.60+ equity.

## v1 — equity baseline, wide heads-up preflop

**What it does:** opens 100% of buttons at 2.5bb, punishes limps with the top half of the range, defends the big blind on pot odds, and postflop calls when equity beats pot odds and bets 0.65 pot at 0.64+ equity. Facing raises, equity is shrunk toward 0.5 by 0.85 per raise.
**Why it exists:** a simple, sound HU baseline: never fold the button, defend wide, decide on equity vs pot odds.
**Parameters:** all defaults by reasoning (`num_samples=96`, `button_open_bb=2.5`, `limp_raise_equity=0.55`, `three_bet_equity=0.60`, `value_bet_equity=0.64`, `bet_pot_fraction=0.65`, `call_margin=0.02`, `raise_shrink=0.85`, `preflop_reraise_multiple=3.0`).
**Results** (seed 0): mean **−51.03**. alan −176.23 ± 20.06 · jack +43.06 ± 39.49 · jesse −37.57 ± 34.13 · max_payne −33.39 ± 18.64. (Smoke test vs random, 2×20 hands: +545.)
**Verdict:** discarded. Two leaks found: shrinking equity *toward 0.5* never folds a hand above 0.5 — postflop equity vs random averages 0.5, so we called nearly every bet — and re-raising bettors at only 0.64 bloated pots against stronger ranges.
**Line count:** 20.

## v2 — respect aggression

**What it does:** identical to v1 except equity facing raises is *multiplied* by 0.75 per raise (so real folds happen), and re-raising a bettor requires 0.80 equity (near-nut) while betting unchecked still needs 0.64.
**Why it exists:** tests whether v1's losses came from overcalling and spazz-raising vs betting ranges.
**Parameters changed:** `per_raise_factor=0.75` (replaces `raise_shrink=0.85`), `reraise_equity=0.80` (new); reasoning from v1's leaks, not tuned.
**Results** (seed 0): mean **+138.40**. alan +1.44 ± 16.36 · jack +276.57 ± 35.64 · jesse +239.21 ± 27.67 · max_payne +36.40 ± 12.79.
**Verdict:** kept (parent of everything below). +189 mean over v1, far outside noise.
**Line count:** 21.

## v3 — thinner value betting

**What it does:** v2 with `value_bet_equity` 0.56 instead of 0.64 — betting medium hands on the theory the opponents call half-pot bets with roughly break-even random equity.
**Why it exists:** tests whether thin value prints against call-wide ranges.
**Results** (seed 0): mean **+120.99**. alan −9.12 ± 20.90 · jack +288.39 ± 30.82 · jesse +165.11 ± 30.46 · max_payne +39.59 ± 12.71.
**Verdict:** discarded. Thin bets get called and raised by better (jesse −74 vs v2). Lesson: bet *stronger*, not weaker.
**Line count:** 21.

## v4 — occasional bluffs

**What it does:** v2 plus, when unchecked postflop with a non-value hand, bluff at the same 0.65-pot size with probability 0.30.
**Why it exists:** tests whether the fold equity vs these nitty-vs-bets opponents pays for bluffs.
**Parameters changed:** `bluff_prob=0.30`, chosen as a moderate stab frequency.
**Results** (seed 0): mean **+72.83**. alan −34.16 ± 15.71 · jack +205.20 ± 32.31 · jesse +114.39 ± 33.55 · max_payne +5.87 ± 14.92.
**Verdict:** discarded. Bluffs burn vs the call-wide stations (jesse, jack), and max_payne's postflop range is what survives its huge preflop folds — strong — so bluffing it fails too.
**Line count:** 23.

## v5 — bigger value sizing

**What it does:** v2 with `bet_pot_fraction` 0.90 instead of 0.65 — same hands bet, nearly pot-sized.
**Why it exists:** the opponents need roughly the same random-equity to call a 0.65-pot bet (≈0.50) and our value threshold (0.64) clears even a pot-sized bet's requirement (≈0.59), so bigger bets are paid by the same worse hands.
**Parameters changed:** `bet_pot_fraction=0.90`, tuned against seed-0 finals.
**Results** (seed 0): mean **+184.55**. alan +70.67 ± 19.82 · jack +380.13 ± 29.18 · jesse +234.64 ± 32.80 · max_payne +52.76 ± 13.13.
**Verdict:** kept. Every matchup improved; alan alone moved +69 (>3 SE paired).
**Line count:** 21.

## v6 — stronger-only value

**What it does:** v5 with `value_bet_equity` 0.68 instead of 0.64 — only clearly strong hands bet.
**Why it exists:** v3 showed thinner is bad; tests the opposite direction (thicker) at the new big size.
**Results** (seed 0): mean **+177.67**. alan +66.50 ± 18.95 · jack +350.39 ± 35.70 · jesse +242.16 ± 27.47 · max_payne +51.64 ± 12.18.
**Verdict:** discarded — statistically tied with v5 (Δ7, well inside noise); v5 kept as parent.
**Line count:** 21.

## v7 — first preflop raise exempt from discount

**What it does:** v5, but preflop equity is only discounted from the *second* raise on (`counted_raises = max(opp_raises − 1, 0)` preflop). Postflop unchanged.
**Why it exists:** v5 folds ~90% of buttons to a single 3-bet (0.75 factor on one raise), which any aggressive heads-up opponent could exploit; a preflop open is routine, not a strength signal.
**Results** (seed 0): mean **+169.09** (alan +60.70 ± 18.97 · jack +322.96 ± 31.78 · jesse +227.10 ± 33.24 · max_payne +65.60 ± 14.61). Head-to-head vs parent v5 (seed 0): **−12.93 ± 28.51** — a statistical tie, no parent warning. Vs random (seed 0): **+702.38 ± 19.18**.
**Verdict:** kept as finalist. Costs ~1 SE vs v5 on the benchmark field, gains structural robustness against the unseen phase-2 opponents (3-bet defense) and beat v5 vs alan and max_payne on the held-out seed.
**Line count:** 22.

## v8 — compliance fix, decisions identical to v7

**What it does:** v7 with the limp-raise size (3.5bb) exposed as the constructor argument `limp_raise_bb` instead of a hardcoded number, per the interpretable-parameters rule. Every decision is bit-identical to v7's.
**Why it exists:** restriction compliance; no strategic change.
**Parameters:** `limp_raise_bb=3.5` (the value v7 used).
**Results:** held-out seed 1000 below; identical by construction to v7.
**Verdict:** deployed.
**Line count:** 22.

## Final choice: v8

Chosen on the held-out seed 1000, run once per finalist (64 tables × 250 hands each):

| v8 (deployed) vs | bb/100 | SE |
|---|---|---|
| alan_wake_claude | +64.68 | 19.75 |
| jack_joyce_gemini | +346.36 | 29.52 |
| jesse_faden_glm | +234.53 | 36.80 |
| max_payne_gpt | +48.05 | 13.28 |
| **mean vs finals** | **+173.41** | |
| random (seed 0) | +702.38 | 19.18 |

The other finalist v5 scored +173.42 on the same held-out run — an exact tie. v8 (via v7) was preferred for robustness, not score: it does not overfold the button to a single 3-bet, which matters against heads-up-native phase-2 opponents, and it beat v5 vs both alan and max_payne on the held-out seed. It also ties its parent head-to-head (−12.93 ± 28.51).

**Known weaknesses:**
- **No bluffing at all** (v4 showed naive random-air bluffs lose badly vs this field, so none were kept). Against a field that over-folds postflop, v8 forfeits fold equity; a semi-bluff-only stab (draws at 0.45–0.60 equity) is the untested idea worth trying with more time.
- **Range-blind equity:** one global 0.75 factor models "a raiser beats random" coarsely — it overfolds vs frequent bluffers and overcalls vs pure nut-peddlers.
- **Fixed sizings:** 2.5bb opens, 3× 3-bets, 0.90-pot bets regardless of board texture, stack depth or street; no limp strategy on the button.
- **Never slowplays or traps:** the 0.80 re-raise bar is face-up value, so a hand-reading opponent can fold correctly and cheaply.
