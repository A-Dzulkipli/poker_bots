# Task: write a heads-up poker bot for a JAX No-Limit Hold'em simulator

You are designing and iterating on a poker bot for a research simulator. Everything you need is in this prompt and in your bot directory. Read all of it before writing code.

## Your assignment

- **Bot name:** `kraken_gpt` (snake_case; it is also your directory and Python package name)
- **Lineage:** phase_4_sea_monsters/kraken_gpt
- **Bot directory:** `/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_4_sea_monsters/kraken_gpt/`. It already exists and holds everything you need. All your work goes there.
- **Benchmark set:** The twelve-bot field (`field`): the eleven opponents in `opponents/` plus RandomBot (`random`), scored by the weighted score, which is your primary score; and every earlier version of your own bot (`vK:Bot`).
- **Additional constraints:** None beyond the restrictions below.

Design a bot, benchmark it, improve it over several versions, and pick one final version for deployment. Aim for the highest **weighted score** (defined below) you can reach **heads-up** under the restrictions below. New in this phase: your bot may keep state across hands and track its opponent.

## First two minutes: check that you can run code

Before anything else, run this from your bot directory:

```
uv run python bench.py check
```

It prints the Python, `jax` and engine it found, checks the engine hash against this contract, and plays a 40-hand smoke test.
- **If it ends with `OK`,** you can run code. Follow the benchmark protocol below.
- **If it fails,** stop and tell the requester the exact output; do not spend the hour writing unbenchmarked code. The usual cause is a shell that cannot reach the project's Python environment. The fix is on the requester's side. Continue without execution only if the requester explicitly tells you to, and then follow "If you cannot run code" below.

## Where this phase comes from

Phase 1 produced stateless bots in about 20 lines; phase 2, heads-up bots in about 50 lines; phase 3, heads-up bots in about 100 lines, all without memory between hands. This phase keeps about 100 lines and an hour, and allows **opponent tracking**: state that persists across hands and is updated from what the bot observes.

The phase-3 heads-up round robin (32,000 hands per pair, bb/100 for the row bot, ± table-clustered SE). All twelve bots in it are your benchmark field:

| | vs windrunner_glm | vs invoker_gpt | vs meepo_claude | vs hadrian_marlowe_gpt | vs valka_onderra_claude | vs aggressor | vs bassander_lin_glm | vs alan_wake_claude | vs random |
|---|---|---|---|---|---|---|---|---|---|
| windrunner_glm | | +9.6 ± 6.5 | +21.7 ± 7.9 | +61.3 ± 4.7 | +85.9 ± 10.2 | +37.2 ± 3.7 | +153.6 ± 10.7 | +96.2 ± 8.4 | +399 ± 10 |
| invoker_gpt | −9.6 | | +35.2 ± 7.8 | +20.1 ± 5.3 | +26.0 ± 9.2 | +9.2 ± 4.7 | +120.0 ± 12.7 | +71.3 ± 7.6 | +367 ± 9 |
| meepo_claude | −21.7 | −35.2 | | +0.9 ± 6.5 | +45.8 ± 10.7 | −23.7 ± 4.0 | +123.3 ± 13.9 | +58.6 ± 9.3 | +457 ± 13 |

What the phase-3 hand histories show, as general lessons rather than facts to exploit:
- **The phase-3 bots bluff; the phase-2 bots did not.** When checked to, windrunner bets 50% of the time, meepo 38% and invoker 27%, against 21 to 25% for the phase-2 finals.
- **Folding a lot is fine if you fight back.** windrunner, the strongest bot, folds 71% of the time facing a postflop bet and 78% against bets over two-thirds pot, yet beats the aggressor (+37). It bets half of all checked-to spots and raises 12% of bets it faces, so a bettor cannot simply print money against it. meepo folds 50% facing bets but raises only 8%, and loses to the aggressor (−24).
- **The edge is in postflop lines, not hole-card selection.** An experiment that played each hole-card class with whichever existing bot had the best result for it gained nothing over windrunner alone.
- **Every benchmark opponent is stateless.** Each has fixed, measurable tendencies (how often it folds to a bet, bluffs, limps, calls down) and never adjusts to you. That is exactly what tracking can find. Your final version will also face the other phase-4 lineages' finals, written independently from the same brief, and those may adapt. So track general statistics and respond with general rules; do not build counters to particular bots.

`aggressor` is a probe bot. It opens every button, raises every limp, and bets two-thirds pot whenever checking is free, on every street. Facing a bet or raise it continues only with at least 65% equity against a random hand, so its profit comes from opponents folding to its bets. It punishes folding and is punished by calling or raising.

Clarity is assessed as strictly as strength: a human reviewer reads every version and rejects any they cannot understand quickly.

## The weighted score

The benchmark field is the twelve bots above: the ten phase-1 to phase-3 finals, `aggressor` and `random`. Each has a strength rank, from the order of their mean result against non-random opponents in phase 3:

| rank | opponent | weight of a win | weight of a loss |
|---|---|---|---|
| 12 | windrunner_glm | 1.85 | 0.15 |
| 11 | invoker_gpt | 1.69 | 0.31 |
| 10 | meepo_claude | 1.54 | 0.46 |
| 9 | hadrian_marlowe_gpt | 1.38 | 0.62 |
| 8 | valka_onderra_claude | 1.23 | 0.77 |
| 7 | aggressor | 1.08 | 0.92 |
| 6 | bassander_lin_glm | 0.92 | 1.08 |
| 5 | alan_wake_claude | 0.77 | 1.23 |
| 4 | max_payne_gpt | 0.62 | 1.38 |
| 3 | jesse_faden_glm | 0.46 | 1.54 |
| 2 | jack_joyce_gemini | 0.31 | 1.69 |
| 1 | random | 0.15 | 1.85 |

A win (positive bb/100) against an opponent of rank r counts r / 6.5; a loss counts (13 − r) / 6.5. The weighted score is the mean of the twelve weighted results. So beating the strong bots is worth the most, and losing to a weak one costs the most. `bench.py ... field` prints it. For reference, the phase-3 results give these scores (each against the other eleven): windrunner_glm +69.9, invoker_gpt +52.0, meepo_claude +47.2, hadrian_marlowe_gpt +43.3, valka_onderra_claude +35.9.

## Opponent tracking: what is now possible

- Your bot's `state` now persists across hands. Each **table is one session of 250 hands against one opponent**, and the state starts fresh from `init_state` at every table. Anything you learn about the opponent is learned within those 250 hands, and the first hands are played with no information.
- You may define `observe(record, state, key) -> state`. The engine calls it at the end of **every** hand, including hands where you never acted, with the whole hand: every action, the board, the payouts and, if the hand reached showdown, the opponent's hole cards. See "Observing hands" in the contract.
- `act` may also update state, for example to remember what you did earlier in the hand.
- Typical things to track: how often the opponent folds to a bet or raise, how often it bets or raises when it could check, how often it calls down to showdown, and how strong its shown hands are relative to its betting. Use the counts to shift a few decisions (bluff more against a frequent folder, call down lighter against a frequent bettor, value bet thinner against a calling station), and fall back to a sound default until the counts mean something.

## Time box: 60 minutes of wall-clock time

- When you start, record the time. If your environment lets you schedule a reminder, timer or self-ping, set one for 50 minutes from now that says "wrap up now". Otherwise check elapsed time (e.g. `date`) between steps.
- At 50 minutes, stop iterating, whatever state you are in. Use the remaining time to run the held-out evaluation if it is not done, pick the final version, finish the README and check the files.
- Record the time you actually used in the README.

## If you cannot run code

Some environments (browser chats) cannot execute code at all. If `bench.py check` cannot run or fails, and the requester tells you to continue anyway:
- Still produce the full set of files. Iterate by reasoning: review each version for contract violations and strategic leaks, and justify each change.
- In the README, mark every benchmark field as `not run`. **Never invent benchmark numbers.** State clearly at the top of the README that the code has not been executed, and include the output of `bench.py check` if you have it.

## Restrictions for this generation

1. **Opponent tracking is allowed, within the state contract.** State is a fixed-shape pytree of `jax.Array`s, updated only in `observe` and `act` (see the contract). Keep it to summary statistics of what you have observed this session: counts, rates or running averages, not a log of hands. Start from sensible priors (for example, pseudo-counts) so the bot plays soundly from hand 1 and does not swing on a handful of observations. Do not store random keys. Do not use Python attributes, module globals or caches as memory; they do not work under `jit` and they break the rule.
2. **About 100 lines of decision code per version.** Treat it like an essay word count: around 100, give or take 10%, is fine. Count the non-blank, non-comment lines of `act`, `observe` and any helper methods or functions your version file defines, **including the tracking code**. Do not count imports, docstrings, `__init__`, `init_state` or `metadata`. The provided `equity.py` helpers do not count either. One statement per line. No semicolons, no stacking several decisions into one expression to save lines, no golfing. You do not have to use all the lines.
3. **Readable at a glance.** Use descriptive names. Each line should say one poker idea ("call if equity beats pot odds"). Add a short comment where the poker reasoning is not obvious. Group the code by situation (preflop as button, preflop as big blind, postflop when checked to, postflop facing a bet, and tracking) and give each group a heading comment, so the reviewer can find each part. A helper method per situation is a good way to do this. Name every tracked statistic for what it measures (`opponent_fold_to_bet_rate`, not `s[3]`). The reviewer should understand what the bot does, and what it learns, in a few minutes.
4. **Interpretable parameters.** Every tunable number is a constructor argument with a default, a descriptive name and a unit in its name or comment (big blinds, pot fraction, equity probability, pseudo-count hands). You may tune them with the benchmark. Keep the count small enough that each one means something. No lookup tables of fitted numbers (for example a 169-entry preflop chart); a short rule that a human can read is fine.
5. **Compute cap.** At most 256 Monte Carlo samples per decision in total, and no loops longer than that. `observe` runs once per hand per seat; keep it cheap (counting, no Monte Carlo). Under `vmap`, every bot's `act` runs at every step at every table.
6. **Randomised actions are allowed and expected.** Bluffing and mixing at sensible frequencies is how a bot avoids being exploitable, including by an adapting opponent. Draw randomness from `key` only, and give any frequency a named parameter.
7. **Adapt to behaviour, not to identity.** Adjustments must follow from statistics observed during the session and apply to any opponent that shows the same behaviour. Do not hard-code knowledge of particular benchmark bots, do not try to recognise which bot you are playing (from bet sizes, timing of actions or any other fingerprint), and do not switch to a strategy written for a specific opponent. You may read the benchmark bots in `opponents/` to understand them, and you may borrow ideas from them (say so in the README).

## Sandbox rules (strict)

- Your bot directory `/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_4_sea_monsters/kraken_gpt/` is your whole workspace. Create and modify files **only** inside it, and read only files inside it. Do not read, list or modify anything outside it, including the engine source, other bots and other lineages, and git state. Do not commit.
- Your session may have been given access to the whole repository so that it can run code. That does not widen what you may read or change.
- The single exception is the project's Python environment, which you use only by running `uv run python ...` from inside your directory (uv finds the project on its own). If `uv` is unavailable, use `../../../.venv/bin/python`. Do not install packages or change the environment.
- Do not edit `equity.py`, `bench.py` or anything in `opponents/`. They are shared with other lineages and must stay identical.
- Import only `jax`, `jax.numpy`, the `poker_bots` package, and your own package's files (`from .equity import ...`).
- No import-time side effects, no monkeypatching, no reading engine internals at runtime, no access to anything other than `obs`, `record`, `state` and `key` inside `act` and `observe`. Any attempt to reach the deck, the opponent's hidden cards or the engine's state disqualifies the bot.
- The environment already has what it needs: Python 3.13, `jax` 0.11, and `poker_bots` at engine commit `decafb9` (engine_sha256 `295477eff7637056`). The contract below describes this exact engine, so you do not need to read its source.

## What is already in your directory

```
/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_4_sea_monsters/kraken_gpt/
    PROMPT.md       # this prompt
    prompt.toml     # the values that filled it (ignore)
    __init__.py     # placeholder; you edit it at the end (see Deliverables)
    equity.py       # shared helpers (do not edit; see "Helpers" below)
    bench.py        # heads-up benchmark script (do not edit)
    opponents/      # frozen, self-contained opponent files (do not edit), one per benchmark bot except random:
                    # aggressor, alan_wake_claude, bassander_lin_glm, hadrian_marlowe_gpt, invoker_gpt,
                    # jack_joyce_gemini, jesse_faden_glm, max_payne_gpt, meepo_claude, valka_onderra_claude, windrunner_glm
```

## Deliverables

```
/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_4_sea_monsters/kraken_gpt/
    __init__.py     # one line: from .vN import Bot as DeployedBot  (N = your final version)
    v1.py           # each version is a separate file defining `class Bot`
    v2.py
    ...
    README.md
```

- Each `vN.py` defines exactly one class named `Bot`, may import from `.equity`, and must not import other versions. **Never edit a version after benchmarking it.** Changes go in a new file.
- Make `v1` a sound heads-up baseline. It may be stateless. Each later version changes one idea, or one closely related group of ideas, so results can be attributed. Add tracking as its own step, so its effect can be measured against the same strategy without it.
- Keep scratch files (benchmark logs, notes) inside your directory if you need them. Mention them in the README.

### README.md

1. **Header:** bot name, lineage, model (your own name and version, as best you know it), prompt version `botgen-v4`, engine (`decafb9`, engine_sha256 `295477eff7637056`), wall-clock time used, whether the code was executed, and the output of `bench.py check`.
2. **One section per version:**
   - *What it does:* two to four plain sentences describing the strategy.
   - *Why it exists:* what weakness in the parent it targets, or what hypothesis it tests.
   - *Parameters:* each with its value and unit, and how it was chosen (default, reasoning, or tuned against what).
   - *Results:* a table with columns opponent, seed, tables × hands, bb/100, SE, plus the weighted score when you ran `field`.
   - *Verdict:* kept as parent of the next version, or discarded, and why.
   - *Line count* of the decision code.
3. **Opponent tracking:** what the deployed version tracks, its priors, which decisions each statistic changes and by how much, and the evidence that it helps: the deployed version against the same strategy without tracking, and its first-50-hands versus later results (`bench.py` prints both).
4. **Final choice:** which version is deployed and why, its held-out results against every benchmark and its weighted score, and known weaknesses, including how an adapting opponent might exploit what your bot learns. If the best-scoring version is not the deployed one (for example because of clarity or a noisy win), say why.

## Metadata (provenance)

Every version's `metadata()` returns a JSON-serialisable dict (only str, int, float, bool, None, lists and dicts). It is written into every simulation log header:

```python
def metadata(self) -> dict:
    return {
        "bot_name": "kraken_gpt",
        "version": 2,                      # this file's N
        "parent": 1,                       # version this was derived from, or None for v1
        "lineage": "phase_4_sea_monsters/kraken_gpt",
        "model": "<your model name and version>",
        "prompt_version": "botgen-v4",
        "phase": "phase_4_sea_monsters",
        "engine": "decafb9 sha256:295477eff7637056",
        "params": {"raise_equity": self.raise_equity, ...},   # every constructor arg and its value
        "contributions": [
            # include each block below verbatim, in order
            {"role": "requester", "who": "a"},
            {"role": "generator", "interface": "GPT (Codex)"}
        ],
    }
```

## Benchmark protocol

Run `bench.py` from inside your bot directory:

```
cd /Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_4_sea_monsters/kraken_gpt
uv run python bench.py check                                   # first: environment and engine check
uv run python bench.py v1:Bot random --tables 2 --hands 20     # smoke test for each new version
uv run python bench.py v1:Bot field --seed 0                   # all twelve opponents, plus the weighted score
uv run python bench.py v2:Bot windrunner_glm --seed 1          # one opponent by name
uv run python bench.py v2:Bot v1:Bot --seed 2                  # against your own earlier version
```

- The format is heads-up (two seats). Your candidate sits in seat 0 on even tables and seat 1 on odd tables, so it plays both sides of the button equally. Stacks are 100bb with rebuys on bust and cash-out above 300bb, blinds 1/2.
- Each table is a 250-hand session (`--hands`), and state starts fresh at each table. Keep `--hands 250` for every result you report, since tracking depends on session length. Use `--tables` to change the sample size.
- The script prints bb/100 and a **table-clustered** standard error: the spread of per-table results divided by √tables. It also prints bb/100 over the first 50 hands of each session and over the rest, which shows whether tracking pays off as the session goes on. Defaults are 64 tables × 250 hands = 16,000 hands per opponent. `field` runs all twelve opponents in sequence.
- Expect heads-up SEs of roughly 4 to 20 bb/100 at the default size against the strong bots, larger against wild ones. Differences of under about 2 SE are noise. Do not keep a change because of one, and do not tune parameters by chasing differences inside the error bars. When two versions are close, use `--tables 128` or a second seed rather than more versions.
- Timing depends on the machine. Time your first `field` run and budget the hour from it. Fewer, informative runs beat many noisy ones. Single-opponent runs against the strong bots are a cheap way to test an idea before a full `field` run.
- **Iterate with seeds 0 to 9 only.**
- **Choose the final version on held-out seed 1000**, run once per finalist with `field`. Report it in the README.
- **The primary score is the weighted score against `field`.** Also check each finalist against its parent: a version that beats the benchmarks but loses to its own parent is a warning sign, since the other phase-4 lineages will not share the benchmarks' habits.

## Helpers in `equity.py`

Import them with `from .equity import ...`. They are tested against this engine.

| Helper | Returns |
|---|---|
| `equity(obs, key, num_samples=128)` | Monte Carlo share of the pot won at showdown against opponents still in the hand, each holding **uniformly random** cards. A float in [0, 1]. Works on every street. 128 samples is half your budget. Equity against a random hand overstates your chances against an opponent who has bet or raised, whose range is stronger than random. |
| `made_hand(obs)` | Strength of your best hand from hole cards plus the board so far (higher is better). `-1` preflop. Deterministic and cheap. |
| `hand_category(strength)` | Category of a `made_hand` value: 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 full house, 7 quads, 8 straight flush; `-1` preflop. |
| `to_call(obs)` | Chips needed to call (0 means checking is free), capped at your stack. |
| `pot_odds(obs)` | Equity needed to break even on a call: `to_call / (pot + to_call)`. |
| `num_live_opponents(obs)` | Opponents still in the hand (heads-up: 1 until someone folds). |
| `fold_or_check(obs)` | Folds only when there is something to call; otherwise checks. |
| `check_call()` | Check or call. |
| `raise_to(obs, total)` | Raise so this street's contribution becomes `total` chips, clipped to the legal range. Becomes a call when raising is not legal. |
| `choose(condition, if_true, if_false)` | Elementwise select between two actions (use instead of Python `if`). |

Note that `made_hand` only ranks your hand against other hands; it says nothing about how the board helps the opponent. A pair using only board cards, for example, is shared.

## Engine and bot contract (engine decafb9, engine_sha256 295477eff7637056)

You are writing a bot for a No-Limit Texas Hold'em simulator written in JAX. The engine runs whole tables under `jax.jit`, batches many tables with `jax.vmap`, and dispatches to bots with `jax.lax.switch`. Your bot's code is traced, not run as Python per decision.

### Interface

A bot is a plain Python class with three methods, plus an optional fourth:

```python
from poker_bots.poker_game import Action, ActionData, PlayerTableState, Round
from poker_bots.hand_eval import evaluate_hands  # optional

class MyBot:
    def __init__(self, ...):               # optional; hyperparameters only (Python constants)
    def init_state(self, key: jax.Array) -> S: ...
    def act(self, obs: PlayerTableState, state: S, key: jax.Array) -> tuple[ActionData, S]: ...
    def metadata(self) -> dict: ...          # plain Python dict, e.g. {"bot_type": "my_bot"}
    def observe(self, record: HandRecord, state: S, key: jax.Array) -> S: ...   # optional
```

- `init_state(key)` is called once per seat via `jax.vmap` over `max_players` keys, so every leaf of your state gets a leading seat axis in storage. You write it for one seat; `act` receives and returns one seat's slice.
- `act` is called only when it is your seat's turn to act. It must return `(ActionData, new_state)`.
- `metadata()` is called once, in Python, at env construction.
- Constructor arguments are compile-time constants. Use them for hyperparameters (aggression, thresholds, sample counts).
- `state` persists across hands for the whole session at a table: what `act` or `observe` returns is what the next call receives. It starts from `init_state` at every table.
- `observe(record, state, key)` is optional. If the class defines it, the engine calls it at the end of every hand, for every seat your bot occupies, with that seat's view of the finished hand (see "Observing hands" below), and stores the state it returns. It is traced like `act`, under the same hard constraints.

### Hard constraints (violating these fails to compile or silently misbehaves)

1. **Pure JAX in `act`, `observe` and `init_state`.** No Python `if`/`while`/`for` on traced values, no `.item()`, no `int(x)`, no NumPy on traced arrays, no Python randomness, no I/O. Use `jnp.where`, `jax.lax.cond`, `jax.lax.select`, `jax.lax.fori_loop`/`scan` with static trip counts. Python loops over static ranges (e.g. `range(n_samples)` with a constructor constant) are fine but unroll at trace time. `jax.debug.print` is allowed for debugging.
2. **Fixed-shape, fixed-dtype state.** `state` is a pytree of `jax.Array`s (or `None`). The `new_state` you return must have exactly the same tree structure, shapes and dtypes as the input `state`. No growing lists, no dicts with changing keys, no Python scalars. Use fixed-size buffers with counters if you need memory.
3. **Exact action dtypes.** Return `ActionData(type=..., value=...)` where both fields are scalar (`shape ()`) `jnp.int32` arrays. Cast explicitly: `jnp.asarray(x, dtype=jnp.int32)`. All bots in a table share one `lax.switch`, so a dtype or shape mismatch is a compile error.
4. **Randomness only from `key`.** The engine passes a fresh key every call. Split it with `jax.random.split`; do not store keys in state for future use and do not use a fixed seed.
5. **Cost.** Under `vmap` over tables, `lax.switch` evaluates every bot type's `act` every step. Your `act` runs on every action at every table, so keep it bounded: a few hundred to a few thousand cheap ops, or a Monte Carlo loop with a small constant sample count.
6. **No hard-coded table size.** Per-seat arrays have length `max_players`, which is an env parameter (default 9). Derive it from `obs.stacks.shape[0]`, which is a static Python int at trace time.

### Actions

```python
class Action:
    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2
    NO_OP = 3          # treated as FOLD; never return it
    SMALL_BLIND = 4    # history only
    BIG_BLIND = 5      # history only
```

- **FOLD**: folds, even when checking is free. The engine does not stop you folding for nothing; guard against it yourself (if `to_call == 0`, check instead).
- **CHECK_CALL**: checks if nothing is owed, otherwise calls (all-in if the stack is short). `value` is ignored.
- **RAISE**: `value` is the **raise-to total for this street**: the amount your `contributed` for this street will equal after the action, not the increment and not the amount added. The engine clips `value` into `[obs.raise_lower_bound, obs.raise_upper_bound]`. If `obs.can_raise` is false, a RAISE is silently converted to CHECK_CALL. Bets on a fresh street are RAISEs too (from `current_bet == 0`).
- Any other `type` value is silently treated as CHECK_CALL.
- All clamping is silent. An off-by-units bug will not error; it will just play badly. Get the units right.

### Chip units

All amounts are integer chips (`int32`). `obs.big_blind` gives the big blind in chips (default env: small blind 1, big blind 2). Express sizing in big blinds or pot fractions and convert. The small blind is not in `obs`; it appears in `obs.history` as the `SMALL_BLIND` record.

### Observation: `PlayerTableState`

All per-seat arrays are **rotated so index 0 is you**; index `i` is the seat `i` places to your left (clockwise, the direction of action). Seat references (`button`, `sb_seat`, `bb_seat`, `history.player_id`) use the same relative numbering. `N = max_players`.

| Field | Shape / dtype | Meaning |
|---|---|---|
| `hole_cards` | `(2,) int32` | Your two cards. |
| `board` | `(5,) int32` | Community cards; undealt slots are `-1`. Preflop all `-1`, flop 3 real, turn 4, river 5. |
| `history` | `ActionRecord`, each field `(max_actions,)` | This hand's actions so far, in order, including blinds (see below). Reset every hand. |
| `stacks` | `(N,) int32` | Chips behind (not yet committed) for each seat. |
| `seated` | `(N,) bool` | Seat was dealt into this hand (had chips at hand start). Empty or busted seats are `False`. |
| `in_hand` | `(N,) bool` | Still in the hand (not folded). Includes all-in players. Only meaningful where `seated`. |
| `button`, `sb_seat`, `bb_seat` | `() int32` | Relative seat of the dealer button, small blind, big blind. Heads-up, the button is also the small blind. |
| `contributed` | `(N,) int32` | Chips each seat has put in **on the current street only**. Resets to 0 each street. |
| `current_round` | `() int32` | `Round.PRE_FLOP=0, FLOP=1, TURN=2, RIVER=3`. |
| `current_bet` | `() int32` | Highest `contributed` on this street (the amount to match). |
| `min_raise` | `() int32` | Minimum legal raise **increment** over `current_bet` (the last full raise size, at least the big blind). |
| `pot` | `() int32` | Total chips committed by everyone this hand, all streets, **including** the current street's `contributed`. |
| `can_raise` | `() bool` | Whether a RAISE is legal for you now. False when you can only call all-in, when a short all-in did not reopen the action, or when nobody else can act. |
| `raise_lower_bound` | `() int32` | Minimum legal raise-to total (`current_bet + min_raise`, or your all-in total if smaller). Equals the call amount when `can_raise` is false. |
| `raise_upper_bound` | `() int32` | Maximum raise-to total: your all-in, `contributed[0] + stacks[0]`. Equals the call amount when `can_raise` is false. |
| `big_blind` | `() int32` | Big blind in chips. |
| `hand_count` | `() int32` | Hands completed at this table so far in this session (0 on the first hand). |

Useful derived values:

```python
to_call      = obs.current_bet - obs.contributed[0]          # 0 means you can check
all_in_total = obs.contributed[0] + obs.stacks[0]
call_cost    = jnp.minimum(to_call, obs.stacks[0])
pot_after_call = obs.pot + call_cost
is_all_in    = obs.seated & obs.in_hand & (obs.stacks == 0)  # per seat
num_opponents_in = jnp.sum(obs.in_hand & obs.seated) - 1
seats_after_button = (-obs.button) % N                       # 0 = you are the button, 1 = first seat after it (counts empty seats)
```

A pot-fraction bet of `f` on a street is a RAISE with `value = obs.current_bet + f * pot_after_call` (as int32), clipped to the bounds yourself if you want to know what will actually happen.

### History: `ActionRecord`

Fields, each shape `(max_actions,)` (default 100). The engine's own logs also store what each bot *requested*; in the bot's view `requested_type` and `requested` are just copies of `action_type` and `amount`, so opponents' raw requests are never visible.

| Field | dtype | Meaning |
|---|---|---|
| `player_id` | `int32` | Relative seat of the actor (0 = you); `-1` for unused slots. |
| `action_type` | `int32` | `FOLD`, `CHECK_CALL`, `RAISE`, `SMALL_BLIND`, `BIG_BLIND`. The recorded type is what the engine applied, not what the bot asked for. |
| `amount` | `int32` | The actor's street `contributed` total **after** the action (raise-to / call-to). For FOLD, their prior street total. Not the increment. |
| `requested_type` | `int32` | Same as `action_type` in the bot's view. |
| `requested` | `int32` | Same as `amount` in the bot's view. |
| `is_all_in` | `bool` | The action put the actor all-in. |
| `round` | `int32` | Street the action happened on. |
| `is_valid` | `bool` | Slot is filled. Always mask with this. |

Slots fill from index 0 in order; the first two valid records are the blinds. Count, e.g., this street's raises with `jnp.sum(h.is_valid & (h.round == obs.current_round) & (h.action_type == Action.RAISE))`. A CHECK_CALL is a check if the actor's `amount` did not increase relative to their previous record on the same street (0 postflop if none; the blind amount preflop), and a call otherwise.

### Observing hands: `observe` and `HandRecord`

`observe(record, state, key)` receives a `HandRecord` (importable from `poker_bots.poker_game`) for the hand that just ended, seat-relative like `obs` (index 0 is you, seat 1 the opponent heads-up). It is called once per hand, after the last action and the payout and before rebuys are applied, including hands in which you never acted (for example, the opponent folded the small blind).

| Field | Shape / dtype | Meaning |
|---|---|---|
| `hand_index` | `() int32` | Index of this hand in the session, from 0. |
| `button`, `sb_seat`, `bb_seat` | `() int32` | Relative seats, as in `obs`. |
| `seated` | `(N,) bool` | Seats dealt into the hand. |
| `seat_bot` | `(N,) int32` | Always `-1` in your view. |
| `starting_stacks` | `(N,) int32` | Stacks at the start of the hand. |
| `hole_cards` | `(N, 2) int32` | Your cards always. The opponent's cards only if the hand went to showdown and they had not folded; otherwise `-1`. |
| `board` | `(5,) int32` | Board cards actually dealt; `-1` for cards not dealt. |
| `board_dealt` | `() int32` | Number of board cards dealt: 0, 3, 4 or 5. |
| `history` | `ActionRecord` | The whole hand's actions, blinds included, in the same relative form as `obs.history`. |
| `action_count` | `() int32` | Number of valid history entries. |
| `in_hand` | `(N,) bool` | Seats that had not folded by the end. Heads-up, `seated[1] & ~in_hand[1]` means the opponent folded. |
| `went_to_showdown` | `() bool` | At least two seats were still in at the end (including all-in run-outs). |
| `hand_ranks` | `(N,) int32` | Evaluator strength of shown hands (higher wins); `-1` unless shown at showdown. |
| `payouts` | `(N,) int32` | Chips each seat won from the pot. |
| `ending_stacks` | `(N,) int32` | Stacks after the payout, before rebuys and cash-outs. `ending_stacks[0] - starting_stacks[0]` is your result for the hand. |
| `rebuys`, `cash_outs` | `(N,) int32` | Chips added (bust) or removed (above 300bb) after the hand. |

`observe` must return a state with exactly the same tree structure, shapes and dtypes as the one it received. Mask everything with `history.is_valid`, and remember that a statistic about, say, folds to your bets needs both "you bet" and "they then folded" from the history.

### Cards

Card id `c` in `0..51`: `rank = c % 13` (0 = deuce, ..., 8 = ten, 9 = jack, 10 = queen, 11 = king, 12 = ace), `suit = c // 13` (4 suits). `-1` means no card.

### Hand evaluator

`evaluate_hands(cards)` takes an int32 array of shape `(..., 5)`, `(..., 6)` or `(..., 7)` holding **only valid cards** (no `-1`) and returns an int32 strength per hand; higher is better, equal means a tie. Strengths are comparable across 5, 6 and 7 cards. Category is `strength // 13**5` (0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 full house, 7 quads, 8 straight flush). It is a table lookup and is cheap; batch it over a leading axis.

Because the number of dealt board cards depends on the street (a traced value), the easy way to get your current made hand is the `made_hand` helper in `equity.py`, which evaluates the 5-, 6- and 7-card cases and selects by street. For equity, complete the board by sampling (the `equity` helper does this). There is no preflop strength table. Passing `-1` padding to `evaluate_hands` gives garbage silently.

To sample unseen cards without Python control flow: build a `(52,)` boolean "known" mask from `hole_cards` and the valid entries of `board`, then e.g. take `jax.random.permutation` of `jnp.arange(52)`, stable-sort known cards to the end (`jnp.argsort(known[perm], stable=True)`), and read the first k.

### What the bot sees and never sees

- `act` sees the current hand up to its own decision. Everything that happens later in a hand, and every hand it does not act in, reaches the bot only through `observe`.
- `observe` sees the finished hand as described above: every action, the dealt board, the payouts, and the opponent's hole cards **only** if they were shown at showdown. Hands that end in a fold never reveal the folder's cards, so a bot that only learns from shown hands sees a biased sample.
- Never: the opponent's cards when not shown, undealt board cards, the deck, the opponent's identity, or the rest of the engine. Do not import or depend on `TableState`.

### Minimal valid example

```python
import jax
import jax.numpy as jnp
from poker_bots.poker_game import Action, ActionData, PlayerTableState

class CallingStation:
    def init_state(self, key):
        return None

    def act(self, obs: PlayerTableState, state, key):
        return ActionData(
            type=jnp.asarray(Action.CHECK_CALL, dtype=jnp.int32),
            value=jnp.asarray(0, dtype=jnp.int32),
        ), state

    def metadata(self):
        return {"bot_type": "calling_station"}
```

### A minimal tracking example

This counts how often the opponent ends a hand by folding and bluffs more against frequent folders. It shows the mechanics only; it is not a recommended strategy.

```python
class FoldCounter:
    def __init__(self, prior_hands: float = 4.0, prior_fold_rate: float = 0.3, bluff_fold_rate: float = 0.45):
        self.prior_hands = prior_hands              # pseudo-count hands behind the prior
        self.prior_fold_rate = prior_fold_rate      # assumed fold rate before any evidence
        self.bluff_fold_rate = bluff_fold_rate      # bluff when the estimated fold rate exceeds this

    def init_state(self, key):
        return {"hands_seen": jnp.zeros((), jnp.float32), "opponent_folds": jnp.zeros((), jnp.float32)}

    def observe(self, record, state, key):
        opponent_folded = record.seated[1] & ~record.in_hand[1]
        return {"hands_seen": state["hands_seen"] + 1,
                "opponent_folds": state["opponent_folds"] + opponent_folded.astype(jnp.float32)}

    def act(self, obs, state, key):
        fold_rate = (state["opponent_folds"] + self.prior_fold_rate * self.prior_hands) / (state["hands_seen"] + self.prior_hands)
        bluff = (fold_rate > self.bluff_fold_rate) & (to_call(obs) == 0)
        ...
        return action, state            # act may also return an updated state
```

Note the dtype discipline: the counts are `float32` in `init_state`, so `observe` adds `float32` values. Adding a `bool` or an `int32` would change the dtype and fail to compile.

### Heads-up specifics

Every table in this phase has `max_players = 2`, so per-seat arrays have length 2 and seat 1 is always your opponent, in `obs` and in `record`.
- The button posts the small blind, acts **first preflop** and **last on every later street**. The big blind acts last preflop and first postflop.
- `obs.button == 0` means you are the button (small blind); `obs.button == 1` means you are the big blind.
- Stacks are 100bb at the start, carry over between hands, are topped back up to 100bb after busting, and are capped at 300bb by cash-out. Effective stack is `jnp.minimum(obs.stacks[0] + obs.contributed[0], obs.stacks[1] + obs.contributed[1])`.
- Each table is one 250-hand session against one opponent, in the benchmark and in later tournaments. State starts fresh at every table.
- Score: bb/100 from the candidate's seat, `(ending stack - starting stack)` per hand, excluding rebuys and cash-outs.

---
