# alan_wake_claude

| Field | Value |
|---|---|
| Bot name | `alan_wake_claude` |
| Lineage | `phase_1_remedy/alan_wake_claude` |
| Model | Claude Opus 5.5 (`claude-opus-5-5`) |
| Prompt version | `botgen-v1` |
| Engine commit | `02abc49` (checked with `git log` in the repo) |
| Wall-clock time used | ~10 minutes for the executed run (timer restarted 00:24, wrap-up 00:34 Adelaide). An earlier ~3-minute pass wrote v1–v3 without execution. |
| Code executed | **Yes.** Every number below is real `bench.py` output. |

**Execution environment (read this).** The run happened in the Claude desktop app's Linux VM
on this Mac, which mounts the repo folder. The repo's `.venv` holds macOS binaries that cannot
load in that VM, so I built a throwaway Linux venv **outside the repo** (VM home directory).
It uses Python 3.13.15 plus `jax==0.11.2`, `jaxlib==0.11.2` and `numpy==2.5.3`, the versions
pinned in `uv.lock`. The engine was imported from the repo's `src/` through `PYTHONPATH`, not
installed. Nothing outside this bot directory was modified. Runs were on CPU, often several in
parallel, so the timings shown by `bench.py` include contention. This deviates from the letter of
"do not install packages": the user asked for execution after the first, unexecuted pass. The
versions match the lock file, so the results should carry over to the macOS `.venv`.

`equity.py` and `bench.py` were extracted programmatically from the prompt's code blocks, so they
are verbatim copies. v1–v3 were written before any benchmark ran and **none of the versions was
edited after benchmarking**. Decision-line counts were measured with an AST script (non-blank,
non-comment lines of `act`; there are no helper methods). All runs are 32 tables × 250 hands,
which gives 24,000 candidate seat-hands.

---

## v1: baseline

**What it does.** Estimates equity against the live opponents (128 Monte Carlo samples, random
opponent hands). It calls when equity beats pot odds by 0.03 and raises when equity is at least
0.60. Preflop raises go to 3× the current bet, postflop bets are call + 0.66 pot, and any raise
using 40% or more of our total chips becomes an all-in.

**Why it exists.** A simple, sound baseline: it never folds when checking is free, never calls
below pot odds, and its sizes are in sensible units.

**Parameters.**
| Name | Value | Unit | Chosen by |
|---|---|---|---|
| `num_samples` | 128 | MC samples per decision | half the 256 cap: the helper's default, keeps `vmap` cost down |
| `raise_equity` | 0.60 | equity probability | reasoning: a clear favourite heads-up |
| `call_margin` | 0.03 | equity probability above pot odds | reasoning: buffer for MC noise (about ±0.04 at 128 samples) |
| `preflop_raise_multiple` | 3.0 | × current bet | standard 3bb open, 3× 3-bet |
| `bet_pot_fraction` | 0.66 | pot fraction | standard value-bet size |
| `shove_commit_fraction` | 0.40 | share of our total chips | reasoning: past about 40% committed, folding later is rarely right |

**Results.**
| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---|---|---|---|
| random | 0 | 32 × 250 | +588.95 | 31.12 |

**Verdict.** Kept as parent. Structural leak: a fixed 0.60 raise bar is almost never reached
multiway (AA against 5 random hands is only about 0.49), so its best hands limp or call at full tables.

**Line count.** 14. This is under the ~20 budget on purpose, as a minimal baseline.

## v2: raise bar relative to fair share

**What it does.** Same as v1, except the raise test uses *strength* = equity × (players in the
hand). 1.0 is an average hand, and it raises at strength ≥ 1.5.

**Why it exists.** It targets v1's multiway passivity. The bar becomes 0.75 heads-up, 0.50
three-way and 0.25 six-way.

**Parameters.** As v1, with `raise_equity` replaced by:
| Name | Value | Unit | Chosen by |
|---|---|---|---|
| `raise_strength` | 1.5 | × fair-share equity | reasoning: about the top 10–15% of preflop hands against 5 random hands |

**Results.**
| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---|---|---|---|
| random | 0 | 32 × 250 | +835.86 | 40.12 |
| v1 | 0 | 32 × 250 | +78.04 | 20.59 |

**Verdict.** Kept as parent. It beats v1 by +78 (3.8 SE). Remaining leak: equity assumes
opponents hold random cards even after they raise, so it over-calls and re-raises into strength.

**Line count.** 15.

## v3: discount equity against raisers (deployed)

**What it does.** Same as v2, but it multiplies equity by (1 − 0.15) for each raise an opponent
(not us) made on this street, before both the call and raise tests. Blinds are recorded as
`SMALL_BLIND`/`BIG_BLIND`, so they are not counted.

**Why it exists.** It targets v2's random-range assumption. A raiser's range is far stronger than
random, so the discount tightens calls and re-raises only when facing aggression.

**Parameters.** As v2, plus:
| Name | Value | Unit | Chosen by |
|---|---|---|---|
| `raise_discount` | 0.15 | fraction of equity removed per opponent raise this street | reasoning: AK drops from about 0.65 vs random to 0.45–0.55 vs a typical raising range; not tuned |

**Results.**
| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---|---|---|---|
| random | 0 | 32 × 250 | +668.12 | 32.52 |
| v1 | 0 | 32 × 250 | +86.12 | 13.69 |
| v2 | 0 | 32 × 250 | +55.81 | 11.85 |

**Verdict.** Kept as parent, and deployed. It beats v2 by +56 (4.7 SE) and v1 by +86 (6.3 SE).
It wins *less* against RandomBot than v2 (+668 vs +836): folding to random raises gives up some
of RandomBot's spew. That is the expected cost of respecting aggression, and the right trade
against real opponents.

**Line count.** 18.

## v4: wider raising in late position (discarded)

**What it does.** Same as v3, but on the button or in the cutoff the raise bar drops from 1.5 to
1.2 × fair share.

**Why it exists.** Hypothesis: v3 ignores position, and the button and cutoff can profitably raise
wider because fewer players act after them.

**Parameters.** As v3, plus:
| Name | Value | Unit | Chosen by |
|---|---|---|---|
| `late_position_bonus` | 0.3 | × fair-share equity, subtracted from the raise bar | reasoning; not tuned |

**Results.**
| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---|---|---|---|
| random | 0 | 32 × 250 | +761.29 | 34.85 |
| v1 | 0 | 32 × 250 | +86.85 | 17.65 |
| v2 | 0 | 32 × 250 | +40.63 | 15.63 |
| v3 | 0 | 32 × 250 | −5.63 | 12.44 |

**Verdict.** Discarded. Against its parent it is flat (−5.6, under 0.5 SE), and against v2 it
scores lower than v3 does (+41 vs +56). The only clear gain is against RandomBot (+761 vs
+668), which is exploitation of a weak bot, not better poker. The likely reason: wider raises
with equity computed against random hands end up in bloated pots that the raise discount then
folds out of. It was not run on the held-out seed because it did not qualify as a finalist.

**Line count.** 20.

---

## Final choice

**Deployed: v3** (`__init__.py`: `from .v3 import Bot as DeployedBot`).

It is the best version against its own lineage on seed 0, the signal the protocol weights most,
and v4's change made no difference against it.

**Held-out result (seed 1000).**
| Opponent | Seed | Tables × hands | bb/100 | SE |
|---|---|---|---|---|
| random | 1000 | 32 × 250 | +632.80 | 30.68 |
| v1 | 1000 | 32 × 250 | +79.84 | 14.53 |
| v2 | 1000 | 32 × 250 | +56.36 | 12.50 |

The held-out results reproduce seed 0 closely (+56 vs v2 on both seeds), so the gain is not
seed noise.

**Known weaknesses.**
- No bluffing or semi-bluffing. Bets are pure value, so its bets are easy to read and it can be
  pushed off pots by aggression.
- No position awareness. The one attempt (v4) did not help, but a better-shaped version
  (stealing only in unopened pots) was not tested.
- The raise discount is crude: one factor per raise, regardless of raise size or the raiser's position.
- No cap on re-raise wars. Two strong hands escalate to all-in via the shove rule.
- 128 samples give about ±0.04 equity noise near the pot-odds boundary.
- Every opponent in the benchmark is from this lineage or RandomBot. Strength against other lineages is untested.

*Housekeeping:* a `__pycache__/` folder in this directory comes from the first pass's syntax
check, and the sandbox could not delete it. It is harmless; delete it at will.
