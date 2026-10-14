# Task: write a poker bot for a JAX No-Limit Hold'em simulator

You are designing and iterating on a poker bot for a research simulator. Everything you need is in this prompt. Read all of it before writing code.

## Your assignment

- **Bot name:** `alan_wake_claude` (snake_case; it is also your directory and Python package name)
- **Lineage:** phase_1_remedy/alan_wake_claude
- **Root directory:** `/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_1_remedy`. Create `/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_1_remedy/alan_wake_claude/` and put everything there.
- **Benchmark set:** RandomBot (`random`) and every earlier version of your own bot (`vK:Bot`).
- **Additional constraints:** None beyond the restrictions below.

Design a bot, benchmark it, improve it over a few versions, and pick one final version for deployment. Aim for the highest expected chip win rate (bb/100) you can reach under the restrictions below. Your final version will later be played against the final bots of other independently written lineages, so aim for robustly strong play. Do not overfit to exploiting the benchmark set. RandomBot is a weak sanity check; beating your own earlier versions is the stronger signal. Clarity is assessed as strictly as strength: a human reviewer reads every version and rejects any they cannot understand at a glance.

## Time box: 15 minutes of wall-clock time

- When you start, record the time. If your environment lets you schedule a reminder, timer or self-ping, set one for 12 minutes from now that says "wrap up now". Otherwise check elapsed time (e.g. `date`) between steps.
- At 12 minutes, stop iterating, whatever state you are in. Use the remaining time to pick the final version, finish the README and check the files.
- Record the time you actually used in the README.

## If you cannot run code

Some environments (browser chats) cannot execute code. Work out which kind you are at the start.
- **Can run code:** follow the benchmark protocol below. Code execution only counts if it runs on the machine that holds the root directory, with `poker_bots` importable. A remote sandbox where `import poker_bots` fails counts as not being able to run code.
- **Cannot run code:** still produce the full set of files. Iterate by reasoning: review each version for contract violations and strategic leaks, and justify each change. In the README, mark every benchmark field as `not run`. **Never invent benchmark numbers.** State clearly at the top of the README that the code has not been executed.

## Restrictions for this generation

1. **No persistent state.** `init_state` returns `None`, and `act` returns the `state` it was given unchanged. Decisions use only the current `obs` (including this hand's `obs.history`) and the random `key`. No opponent modelling across hands and no memory. Do not try to get around this with Python attributes, module globals or caches. They will not work under `jit`, and they break the rule.
2. **About 20 lines of decision code per version.** Treat it like an essay word count: around 20, give or take 10%, is fine. Count the non-blank, non-comment lines of `act` plus any helper methods or functions your version file defines. Do not count imports, docstrings, `__init__`, `init_state` or `metadata`. The shared `equity.py` helpers (below) do not count either. One statement per line. No semicolons, no stacking several decisions into one expression to save lines, no golfing.
3. **Readable at a glance.** Use descriptive names. Each line should say one poker idea ("call if equity beats pot odds"). Add a short comment where the poker reasoning is not obvious. The reviewer should understand what the bot does in under a minute.
4. **Interpretable parameters.** Every tunable number is a constructor argument with a default, a descriptive name and a unit in its name or comment (big blinds, pot fraction, equity probability). You may tune them with the benchmark. Keep the count small enough that each one means something.
5. **Compute cap.** At most 256 Monte Carlo samples per decision in total, and no loops longer than that. Under `vmap`, every bot's `act` runs at every step at every table.

## Sandbox rules (strict)

- Create and modify files **only** inside `/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_1_remedy/alan_wake_claude/`. Do not touch anything else: not the engine (`poker_bots` package), not its tests, not other bots, not git state. Do not commit.
- Import only `jax`, `jax.numpy`, the Python standard library (in `bench.py` only), `numpy` (in `bench.py` only), the `poker_bots` package, and your own package's files.
- No import-time side effects, no monkeypatching, no reading engine internals at runtime, no access to anything other than `obs` and `key` inside `act`. The bot's `act` only ever gets `obs`. Any attempt to reach the deck, other players' cards or the engine's state disqualifies the bot.
- Do not install packages. The environment already has what it needs: Python 3.13, `jax` 0.11, and `poker_bots` installed (editable) at engine commit `02abc49`.

## Deliverables

```
/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_1_remedy/alan_wake_claude/
    __init__.py     # one line: from .vN import Bot as DeployedBot  (N = your final version)
    equity.py       # shared helpers, copied verbatim from this prompt
    bench.py        # benchmark script, copied verbatim from this prompt
    v1.py           # each version is a separate file defining `class Bot`
    v2.py
    ...
    README.md
```

- Each `vN.py` defines exactly one class named `Bot`, may import from `.equity`, and must not import other versions. **Never edit a version after benchmarking it.** Changes go in a new file.
- Make `v1` a simple, sound baseline. Each later version changes one idea, so results can be attributed.

### README.md

1. **Header:** bot name, lineage, model (your own name and version, as best you know it), prompt version `botgen-v1`, engine commit `02abc49`, wall-clock time used, and whether the code was executed.
2. **One section per version:**
   - *What it does:* two or three plain sentences describing the strategy.
   - *Why it exists:* what weakness in the parent it targets, or what hypothesis it tests.
   - *Parameters:* each with its value and unit, and how it was chosen (default, reasoning, or tuned against what).
   - *Results:* a table with columns opponent, seed, tables × hands, bb/100, SE.
   - *Verdict:* kept as parent of the next version, or discarded, and why.
   - *Line count* of the decision code.
3. **Final choice:** which version is deployed and why, its held-out result, and known weaknesses. If the best-scoring version is not the deployed one (for example because of clarity or a noisy win), say why.

## Metadata (provenance)

Every version's `metadata()` returns a JSON-serialisable dict (only str, int, float, bool, None, lists and dicts). It is written into every simulation log header:

```python
def metadata(self) -> dict:
    return {
        "bot_name": "alan_wake_claude",
        "version": 2,                      # this file's N
        "parent": 1,                       # version this was derived from, or None for v1
        "lineage": "phase_1_remedy/alan_wake_claude",
        "model": "<your model name and version>",
        "prompt_version": "botgen-v1",
        "engine_commit": "02abc49",
        "params": {"raise_equity": self.raise_equity, ...},   # every constructor arg and its value
        "contributions": [
            # include each block below verbatim, in order
            {"role": "requester", "who": "a"},
            {"role": "generator", "interface": "Claude desktop (Code)"}
        ],
    }
```

## Benchmark protocol

Use `bench.py` (below) from inside your bot directory:

```
cd /Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_1_remedy/alan_wake_claude
python bench.py v1:Bot random --seed 0             # against RandomBot
python bench.py v2:Bot v1:Bot --seed 0             # against your own earlier version
```

If `python` doesn't see `poker_bots`, use the project's environment (`uv run python bench.py ...` from the repository, or `.venv/bin/python`).

- The format is 6-max, with your candidate in three alternating seats and the opponent in the other three. Stacks are 100bb with rebuys on bust and cash-out above 300bb, blinds 1/2. The script prints bb/100 with its standard error over seat-hands.
- Defaults are 32 tables × 250 hands per run. Expect roughly one to two minutes per run including compilation, depending on hardware. Budget your 15 minutes accordingly. Fewer, informative runs are better than many noisy ones.
- **Iterate with seeds 0 to 9 only.**
- **Choose the final version on held-out seed 1000**, run once per finalist against every opponent in the benchmark set. Report it in the README.
- When choosing the final version, weight results against your own earlier versions above results against RandomBot. A version that crushes RandomBot harder by playing exploitatively but loses to its parent is the wrong pick.
- A difference smaller than about 2 SE is noise. Do not keep a change because of it, and do not tune parameters by chasing differences inside the error bars.
- Before benchmarking any version, check that it imports and compiles: `python bench.py vN:Bot random --tables 2 --hands 20`.

---

## Engine and bot contract (engine commit 02abc49)

You are writing a bot for a No-Limit Texas Hold'em simulator written in JAX. The engine runs whole tables under `jax.jit`, batches many tables with `jax.vmap`, and dispatches to bots with `jax.lax.switch`. Your bot's code is traced, not run as Python per decision.

### Interface

A bot is a plain Python class with three methods:

```python
from poker_bots.poker_game import Action, ActionData, PlayerTableState, Round
from poker_bots.hand_eval import evaluate_hands  # optional

class MyBot:
    def __init__(self, ...):               # optional; hyperparameters only (Python constants)
    def init_state(self, key: jax.Array) -> S: ...
    def act(self, obs: PlayerTableState, state: S, key: jax.Array) -> tuple[ActionData, S]: ...
    def metadata(self) -> dict: ...          # plain Python dict, e.g. {"bot_type": "my_bot"}
```

- `init_state(key)` is called once per seat via `jax.vmap` over `max_players` keys, so every leaf of your state gets a leading seat axis in storage. You write it for one seat; `act` receives and returns one seat's slice.
- `act` is called only when it is your seat's turn to act. It must return `(ActionData, new_state)`.
- `metadata()` is called once, in Python, at env construction.
- Constructor arguments are compile-time constants. Use them for hyperparameters (aggression, thresholds, sample counts).

### Hard constraints (violating these fails to compile or silently misbehaves)

1. **Pure JAX in `act` and `init_state`.** No Python `if`/`while`/`for` on traced values, no `.item()`, no `int(x)`, no NumPy on traced arrays, no Python randomness, no I/O. Use `jnp.where`, `jax.lax.cond`, `jax.lax.select`, `jax.lax.fori_loop`/`scan` with static trip counts. Python loops over static ranges (e.g. `range(n_samples)` with a constructor constant) are fine but unroll at trace time. `jax.debug.print` is allowed for debugging.
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

### Cards

Card id `c` in `0..51`: `rank = c % 13` (0 = deuce, ..., 8 = ten, 9 = jack, 10 = queen, 11 = king, 12 = ace), `suit = c // 13` (4 suits). `-1` means no card.

### Hand evaluator

`evaluate_hands(cards)` takes an int32 array of shape `(..., 7)` holding **exactly seven valid cards** (no `-1`) and returns an int32 strength per hand; higher is better, equal means a tie. Category is `strength // 13**5` (0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 full house, 7 quads, 8 straight flush). It is a table lookup and is cheap; batch it over a leading axis.

There is no 5- or 6-card evaluator and no preflop table. To evaluate before the river, complete the board: sample the missing board cards (and opponents' hole cards, for equity) from the cards not in `hole_cards` or `board`, then call `evaluate_hands` on the resulting 7-card hands. Passing `-1` padding to it gives garbage silently.

To sample unseen cards without Python control flow: build a `(52,)` boolean "known" mask from `hole_cards` and the valid entries of `board`, then e.g. take `jax.random.permutation` of `jnp.arange(52)`, stable-sort known cards to the end (`jnp.argsort(known[perm], stable=True)`), and read the first k.

### What the bot never sees

- Anything after its own last decision in a hand: later actions, the showdown, opponents' hole cards, who won. The bot is only called when it must act.
- Hands it is not asked to act in at all.
- A hand counter, or rebuys and cash-outs (a stack can jump between hands). `state` persists across hands for the same seat (it is never reset by the engine), so a bot can keep running statistics, but it must detect a new hand itself (e.g. its stored `hole_cards` differ, or `current_round == PRE_FLOP` with no prior own action in `history`).
- The rest of the engine. Do not import or depend on `TableState`.

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

### How it is run and evaluated

```python
from poker_bots.poker_game import make_poker_env
from poker_bots.simulate import simulate

env = make_poker_env(bots=(BotA(), BotB()), max_players=6, small_blind=1, big_blind=2)
keys = jax.random.split(jax.random.PRNGKey(0), num_tables)
seated = jnp.ones((num_tables, 6), dtype=bool)
seat_bot = jnp.tile(jnp.array([0, 1, 0, 1, 0, 1]), (num_tables, 1))
tables = jax.vmap(env.init, in_axes=(0, 0, None, None, 0))(keys, seated, None, 0, seat_bot)
tables, summary = simulate(env, tables, total_hands=10_000, chunk_hands=1_000, out_path="run.jsonl")
```

- `seat_bot[seat]` is the index into `bots` for that seat. One stacked state exists per bot type.
- Many tables run in parallel under `jit(vmap(...))`; each table plays hands in a `lax.while_loop`.
- **Cash-game format.** Every seat starts with 100 big blinds. Stacks carry over between hands. A seat whose stack falls to `min_stack` (default 0, i.e. busting) is topped back up to 100bb before the next hand; chips above `max_stack` (default 300bb) are cashed out. So effective stacks are always between a few chips and 300bb, and busting costs only what was lost.
- **Metric: bb/100.** For each bot type, the score is the mean of `(ending stack - starting stack)` per seat-hand, in big blinds, times 100, with a standard error. Rebuys and cash-outs are excluded. Play to maximise expected chips, not to survive. Your opponents are described in the benchmark set above.
- Every hand is logged as JSON Lines with all hole cards, actions (requested and applied, so clamped or illegal requests are visible to the evaluator), board and payouts. Your `metadata()` dict is written into the log header.

---

## Shared helper file: `equity.py` (copy verbatim)

These helpers are tested against engine `02abc49`. `equity` estimates your share of the pot against the opponents still in the hand, assuming they hold uniformly random cards. It works on every street, because it samples the missing board cards before calling the 7-card evaluator. Calling `equity` once with 128 samples uses half your sample budget. `raise_to` takes a raise-to **total** for the street and clips it to the legal range. `fold_or_check` never folds when checking is free. `choose` replaces Python `if` for picking between actions.

```python
"""Shared helpers for bots. Copy verbatim; does not count toward the line budget."""
import jax
import jax.numpy as jnp

from poker_bots.hand_eval import evaluate_hands
from poker_bots.poker_game import Action, ActionData, PlayerTableState


def equity(obs: PlayerTableState, key: jax.Array, num_samples: int = 128) -> jax.Array:
    """Monte Carlo share of the pot won at showdown against the opponents still in the hand,
    each holding uniformly random cards. Returns a float32 scalar in [0, 1]."""
    num_opponents = obs.stacks.shape[0] - 1
    known = jnp.zeros(52, dtype=jnp.bool_).at[obs.hole_cards].set(True)
    known = known.at[jnp.where(obs.board >= 0, obs.board, 0)].max(obs.board >= 0)
    live = (obs.seated & obs.in_hand)[1:]

    def sample(key: jax.Array) -> jax.Array:
        perm = jax.random.permutation(key, 52)
        draws = perm[jnp.argsort(known[perm], stable=True)]
        board = jnp.where(obs.board >= 0, obs.board, draws[:5])
        opponents = draws[5:5 + 2 * num_opponents].reshape(num_opponents, 2)
        hero = evaluate_hands(jnp.concatenate([obs.hole_cards, board])[None])[0]
        villains = evaluate_hands(jnp.concatenate([opponents, jnp.broadcast_to(board, (num_opponents, 5))], axis=1))
        beaten = jnp.any(live & (villains > hero))
        ties = jnp.sum(live & (villains == hero))
        return jnp.where(beaten, 0.0, 1.0 / (1 + ties))

    return jnp.mean(jax.vmap(sample)(jax.random.split(key, num_samples)))


def to_call(obs: PlayerTableState) -> jax.Array:
    """Chips needed to call (0 means checking is free), capped at the stack."""
    return jnp.minimum(obs.current_bet - obs.contributed[0], obs.stacks[0])


def pot_odds(obs: PlayerTableState) -> jax.Array:
    """Equity needed to break even on a call: to_call / (pot + to_call)."""
    cost = to_call(obs)
    return cost / jnp.maximum(obs.pot + cost, 1)


def num_live_opponents(obs: PlayerTableState) -> jax.Array:
    return jnp.sum(obs.seated & obs.in_hand) - 1


def fold_or_check(obs: PlayerTableState) -> ActionData:
    """Folds only when there is something to call; otherwise checks."""
    kind = jnp.where(to_call(obs) > 0, Action.FOLD, Action.CHECK_CALL)
    return ActionData(type=jnp.asarray(kind, dtype=jnp.int32), value=jnp.asarray(0, dtype=jnp.int32))


def check_call() -> ActionData:
    return ActionData(type=jnp.asarray(Action.CHECK_CALL, dtype=jnp.int32), value=jnp.asarray(0, dtype=jnp.int32))


def raise_to(obs: PlayerTableState, total: jax.Array) -> ActionData:
    """Raise so this street's contribution becomes `total` chips, clipped to the legal range.
    Becomes a call when raising is not legal."""
    value = jnp.clip(jnp.round(total), obs.raise_lower_bound, obs.raise_upper_bound)
    return ActionData(type=jnp.asarray(Action.RAISE, dtype=jnp.int32), value=jnp.asarray(value, dtype=jnp.int32))


def choose(condition: jax.Array, if_true: ActionData, if_false: ActionData) -> ActionData:
    """Elementwise select between two actions (use instead of Python if)."""
    return jax.tree.map(lambda a, b: jnp.where(condition, a, b), if_true, if_false)
```

## Benchmark script: `bench.py` (copy verbatim)

```python
"""Benchmark a bot version. Copy verbatim into your bot directory; do not edit.

    python bench.py v2:Bot random --seed 0
    python bench.py v2:Bot v1:Bot --seed 0 --tables 32 --hands 250

Six-max, the two bots alternate seats (3 each), 100bb stacks with rebuys.
Prints the candidate's bb/100 and standard error over seat-hands.
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PACKAGE = Path(__file__).resolve().parent.name

from poker_bots.poker_game import RandomBot, make_poker_env  # noqa: E402


def load(spec: str):
    if spec == "random":
        return RandomBot()
    module, name = spec.split(":")
    return getattr(importlib.import_module(f"{PACKAGE}.{module}"), name)()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("opponent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--hands", type=int, default=250)
    args = parser.parse_args()

    env = make_poker_env((load(args.candidate), load(args.opponent)), max_players=6, small_blind=1, big_blind=2)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.tables)
    seated = jnp.ones((args.tables, 6), dtype=jnp.bool_)
    seat_bot = jnp.tile(jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32), (args.tables, 1))
    tables = jax.vmap(env.init, in_axes=(0, 0, None, None, 0))(keys, seated, None, 0, seat_bot)

    start = time.time()
    run = jax.jit(jax.vmap(lambda table: env.run_hands(table, args.hands)))
    _, records, num_recorded = jax.device_get(run(tables))
    recorded = np.arange(args.hands)[None, :] < num_recorded[:, None]
    net = (records.ending_stacks - records.starting_stacks) / env.big_blind
    for index, label in ((0, "candidate " + args.candidate), (1, "opponent " + args.opponent)):
        sample = net[records.seated & recorded[:, :, None] & (records.seat_bot == index)]
        mean, se = 100 * sample.mean(), 100 * sample.std(ddof=1) / np.sqrt(len(sample))
        print(f"{label:>24}: {mean:+8.2f} bb/100  (SE {se:.2f}, {len(sample)} seat-hands)")
    print(f"seed {args.seed}, {args.tables} tables x {args.hands} hands, {time.time() - start:.1f}s incl. compile")


if __name__ == "__main__":
    main()
```

## Example of the expected style (not a strong bot; do not submit it as-is)

```python
import jax
import jax.numpy as jnp

from poker_bots.poker_game import PlayerTableState
from .equity import check_call, choose, equity, fold_or_check, pot_odds, raise_to


class Bot:
    """Calls when equity beats pot odds, raises a pot fraction when equity is high."""

    def __init__(self, raise_equity: float = 0.65, call_margin: float = 0.0, bet_fraction: float = 0.75):
        self.raise_equity = raise_equity
        self.call_margin = call_margin
        self.bet_fraction = bet_fraction

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        eq = equity(obs, key, 128)
        bet = obs.current_bet + self.bet_fraction * (obs.pot + obs.current_bet - obs.contributed[0])
        action = choose(eq >= pot_odds(obs) + self.call_margin, check_call(), fold_or_check(obs))
        action = choose(obs.can_raise & (eq >= self.raise_equity), raise_to(obs, bet), action)
        return action, state

    def metadata(self) -> dict:
        return {"bot_name": "demo_bot", "version": 1}
```

This example has 5 lines of decision code. Yours can use up to about 20 lines for a better strategy: position, street, raise counts from `obs.history`, sizing, semi-bluffs and so on.
