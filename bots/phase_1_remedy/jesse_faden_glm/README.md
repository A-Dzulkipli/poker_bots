# jesse_faden_glm

- **Bot name:** `jesse_faden_glm`
- **Lineage:** phase_1_remedy/jesse_faden_glm
- **Model:** GLM-5.3
- **Prompt version:** botgen-v1
- **Engine commit:** 02abc49
- **Wall-clock time used:** about 12 minutes (00:16:54 → 00:28:30 local)
- **Code executed:** yes (bench.py on the local machine, `poker_bots` importable from the project venv)

All versions are stateless (`init_state` returns `None`, `act` returns `state` unchanged) and decide
from a 128-sample Monte Carlo equity estimate against uniformly random opponent cards, via the shared
`equity.py` helpers. Iteration used seeds 0–4; finalists were compared once on held-out seed 1000.

## v1 — equity baseline

- **What it does:** Estimates its equity with 128 Monte Carlo samples, calls whenever equity beats
  pot odds plus a small margin, and raises to 75% of the pot after calling when equity is at least 0.62.
  Never folds when checking is free.
- **Why it exists:** Simple, sound baseline: every decision reduces to "equity vs price".
- **Parameters:** `call_margin=0.03` (probability, default), `raise_equity=0.62` (probability, default),
  `bet_fraction=0.75` (pot fraction, default), `num_samples=128` (count, half the compute cap).
- **Results:**

  | opponent | seed | tables × hands | bb/100 | SE |
  |---|---|---|---|---|
  | random | 0 | 32 × 250 | +639.69 | 30.95 |
  | random | 1000 | 32 × 250 | +600.60 | 30.22 |

- **Verdict:** kept as parent of v2; ultimately deployed (see Final choice).
- **Line count:** 5.

## v2 — multiway tightness

- **What it does:** Same as v1, but the equity a call requires grows by 0.04 for each extra live
  opponent beyond the first, so it folds more in multiway pots.
- **Why it exists:** v1's call test uses heads-up pot odds even when five players see the flop;
  calling too wide in family pots is v1's clearest leak.
- **Parameters:** adds `multiway_tightness=0.04` (probability per extra opponent, reasoning: roughly
  the equity drop one extra random hand causes); others inherited from v1.
- **Results:**

  | opponent | seed | tables × hands | bb/100 | SE |
  |---|---|---|---|---|
  | v1 | 0 | 32 × 250 | −10.57 | 19.21 |
  | v1 | 1 | 32 × 250 | +32.57 | 18.66 |
  | v1 | 2 | 32 × 250 | +24.77 | 19.31 |
  | v1 | 3 | 32 × 250 | +22.86 | 19.74 |
  | v1 | 4 | 32 × 250 | −1.25 | 19.59 |
  | v1 | 1000 | 32 × 250 | −37.30 | 19.90 |

  Mean over six seeds ≈ +5.2 bb/100 (SE of mean ≈ 8): a wash, not a win.

- **Verdict:** kept as parent of v3, but the hypothesis failed: multiway folding gained nothing
  measurable against its own family and cost ~240 bb/100 against RandomBot.
- **Line count:** 6.

## v3 — stabs in uncontested pots

- **What it does:** Same as v2, plus a stab: when nothing real has been bet this street
  (to call at most one big blind), at most two live opponents remain, and equity is at least 0.52,
  it bets the usual pot fraction instead of checking.
- **Why it exists:** v2 checks through too many uncontested pots; medium-equity hands should win
  the pot immediately often enough for a stab to profit.
- **Parameters:** adds `stab_equity=0.52` (probability, slightly above the median vs one random hand)
  and `max_stab_opponents=2` (count, limits stabs to short fields); others inherited from v2.
- **Results:**

  | opponent | seed | tables × hands | bb/100 | SE |
  |---|---|---|---|---|
  | v2 | 0 | 32 × 250 | +19.74 | 20.42 |
  | v2 | 1 | 32 × 250 | −4.85 | 21.02 |
  | v2 | 2 | 32 × 250 | −17.86 | 19.71 |
  | v2 | 1000 | 32 × 250 | −29.33 | 20.48 |
  | v1 | 0 | 32 × 250 | +7.03 | 20.78 |
  | v1 | 1000 | 32 × 250 | −1.90 | 20.87 |

  Mean vs v2 over four seeds ≈ −8.1 bb/100: the stab gains nothing against its own
  calling-station family.

- **Verdict:** discarded; neutral-to-slightly-negative against its parent.
- **Line count:** 9.

## v4 — preflop open-raising

- **What it does:** Same as v3, but the uncontested-pot bet also fires preflop at any table size
  (open-raise with equity ≥ 0.50 whenever at most a big blind is owed).
- **Why it exists:** Tests whether taking the initiative preflop, instead of limping along,
  is worth chips.
- **Parameters:** adds `open_equity=0.50` (probability); others inherited from v3.
- **Results:**

  | opponent | seed | tables × hands | bb/100 | SE |
  |---|---|---|---|---|
  | v3 | 0 | 32 × 250 | −18.00 | 21.51 |
  | v3 | 1 | 32 × 250 | −24.77 | 21.19 |
  | v3 | 2 | 32 × 250 | −15.21 | 21.85 |

  Mean ≈ −19.3 bb/100, negative on all three seeds (consistent in direction, if individually inside
  2 SE).

- **Verdict:** discarded. Against opponents who rarely fold, wider preflop aggression only builds
  pots without fold equity.
- **Line count:** 10.

## v5 — pot-sized value bets

- **What it does:** Back to v2's strategy, but value bets and raises target the full pot
  (`bet_fraction=1.0`) instead of 75% of it.
- **Why it exists:** The family calls far too often, so when v2 does bet a strong hand, a bigger
  bet should win more from the callers. This is the mirror-image of v4's failed idea: more money
  from strong hands, not more pots contested.
- **Parameters:** `bet_fraction=1.0` (pot fraction, changed from v2's 0.75); others inherited from v2.
- **Results:**

  | opponent | seed | tables × hands | bb/100 | SE |
  |---|---|---|---|---|
  | v2 | 0 | 32 × 250 | +2.13 | 19.09 |
  | v2 | 1 | 32 × 250 | −29.68 | 19.28 |
  | v2 | 2 | 32 × 250 | −9.27 | 19.84 |

  Mean ≈ −12.3 bb/100, negative on 2 of 3 seeds: pot-sizing gains nothing measurable here.

- **Verdict:** discarded. Doubling down on bet size did not extract more from callers than it gave
  back in bigger losses when behind.
- **Line count:** 6.

## Final choice

**Deployed: v1** (`__init__.py` → `from .v1 import Bot as DeployedBot`).

Held-out seed 1000, run once per finalist against its benchmark set (random + earlier versions):

| candidate | opponent | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| v1 | random | 32 × 250 | +600.60 | 30.22 |
| v2 | random | 32 × 250 | +397.77 | 23.92 |
| v2 | v1 | 32 × 250 | −37.30 | 19.90 |
| v3 | random | 32 × 250 | +404.74 | 24.12 |
| v3 | v1 | 32 × 250 | −1.90 | 20.87 |
| v3 | v2 | 32 × 250 | −29.33 | 20.48 |

Why v1: the lineage head-to-heads cannot separate v1, v2 and v3 — v2 vs v1 over six seeds averages
+5.2 ± 8 bb/100, v3 vs v2 over four seeds −8.1, all inside noise, and v4/v5 lost outright. With the
family tied, the deciding signals are the RandomBot matches, where v1 (+600/+640) beats v2 and v3
(~+400) by roughly 8 SE, and simplicity. This is not exploiting RandomBot specifically — v1 is the
unmodified equity-vs-pot-odds baseline; the later "improvements" only removed its profitable
looseness. Since the eventual opponents are other lineages' final bots, of unknown strength, the
version that ties the family and dominates the weak tail is the robust pick.

Known weaknesses:

- Equity assumes opponents hold uniformly random cards, so it overcalls against tight ranges;
  real equity vs strong hands is lower than the estimate.
- It never bluffs and rarely slowplays: a fixed 0.62 raise threshold makes its betting range
  transparent and lets observant opponents fold cheaply.
- The 128-sample equity estimate has roughly ±4 points of noise, so marginal calls are sometimes
  decided by sampling noise.
- No position, bet-sizing variation, or opponent awareness (forbidden by the statelessness rule).
