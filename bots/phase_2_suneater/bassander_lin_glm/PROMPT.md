# Task: write a heads-up poker bot for a JAX No-Limit Hold'em simulator

You are designing and iterating on a poker bot for a research simulator. Everything you need is in this prompt and in your bot directory. Read all of it before writing code.

## Your assignment

- **Bot name:** `bassander_lin_glm` (snake_case; it is also your directory and Python package name)
- **Lineage:** phase_2_suneater/bassander_lin_glm
- **Bot directory:** `/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_2_suneater/bassander_lin_glm/`. It already exists and holds everything you need. All your work goes there.
- **Benchmark set:** The four phase-1 finals in `opponents/` (`finals`, or one by name), every earlier version of your own bot (`vK:Bot`), and RandomBot (`random`) as a sanity check only.
- **Additional constraints:** None beyond the restrictions below.

Design a bot, benchmark it, improve it over several versions, and pick one final version for deployment. Aim for the highest expected chip win rate (bb/100) you can reach **heads-up** under the restrictions below.

## Where this phase comes from

Phase 1 asked four independently written lineages for the strongest stateless bot each could write in about 20 lines and 15 minutes. Their final versions are now the benchmark: they are in `opponents/`, one file each, frozen. Phase 1 was played 6-max. In the 6-max round robin (32,000 hands per pair, bb/100 for the row bot, table-clustered SE):

| | vs alan_wake_claude | vs max_payne_gpt | vs jesse_faden_glm | vs jack_joyce_gemini | vs random |
|---|---|---|---|---|---|
| alan_wake_claude | | +21.3 ± 4.2 | +68.6 ± 6.1 | +84.9 ± 5.8 | +656 |
| max_payne_gpt | −21.3 | | +29.2 ± 6.5 | +47.5 ± 7.0 | +433 |
| jesse_faden_glm | −68.6 | −29.2 | | +36.3 ± 10.5 | +602 |
| jack_joyce_gemini | −84.9 | −47.5 | −36.3 | | +709 |

This phase is **heads-up only**, with more room: about 50 lines and an hour. The phase-1 bots were written for six players, and heads-up changes what good play looks like. You play every hand from one of the blinds, ranges are far wider, position alternates every hand, and folding too much simply bleeds blinds. Heads-up results for the same bots, from `bench.py` at seed 0 (16,000 hands per pair, bb/100 for the row bot, ± table-clustered SE):

| | vs alan_wake_claude | vs max_payne_gpt | vs jesse_faden_glm | vs jack_joyce_gemini | vs random | mean vs the other finals |
|---|---|---|---|---|---|---|
| alan_wake_claude | | +76.1 ± 14.3 | +172.7 ± 18.5 | +175.4 ± 18.8 | +334 ± 14 | +141 |
| max_payne_gpt | −76.1 | | −1.4 ± 22.7 | +128.8 ± 23.6 | +292 ± 14 | +17 |
| jesse_faden_glm | −172.7 | +1.4 | | +125.2 ± 36.8 | +692 ± 25 | −15 |
| jack_joyce_gemini | −175.4 | −128.8 | −125.2 | | +708 ± 25 | −143 |

The order matches 6-max, but the gaps are two to three times as wide, and beating RandomBot by a lot says little about strength against real opponents. jack_joyce_gemini beats random by the most and loses to everyone. Note the SEs: at 16,000 hands a difference needs roughly 30 to 75 bb/100 (2 SE) before it means anything.

Your final version will later play the other phase-2 lineages' finals, written independently from the same brief, as well as these benchmarks. So aim for play that is robustly strong heads-up, not a counter to four specific bots. Clarity is assessed as strictly as strength: a human reviewer reads every version and rejects any they cannot understand quickly.

## Time box: 60 minutes of wall-clock time

- When you start, record the time. If your environment lets you schedule a reminder, timer or self-ping, set one for 50 minutes from now that says "wrap up now". Otherwise check elapsed time (e.g. `date`) between steps.
- At 50 minutes, stop iterating, whatever state you are in. Use the remaining time to run the held-out evaluation if it is not done, pick the final version, finish the README and check the files.
- Record the time you actually used in the README.

## If you cannot run code

Some environments (browser chats) cannot execute code. Work out which kind you are at the start.
- **Can run code:** follow the benchmark protocol below. Code execution only counts if it runs on the machine that holds the bot directory, with `poker_bots` importable. A remote sandbox where `import poker_bots` fails counts as not being able to run code.
- **Cannot run code:** still produce the full set of files. Iterate by reasoning: review each version for contract violations and strategic leaks, and justify each change. In the README, mark every benchmark field as `not run`. **Never invent benchmark numbers.** State clearly at the top of the README that the code has not been executed.

## Restrictions for this generation

1. **No persistent state.** `init_state` returns `None`, `act` returns the `state` it was given unchanged, and the bot does not define `observe`. Decisions use only the current `obs` (including this hand's `obs.history`) and the random `key`. No opponent modelling across hands and no memory. Do not try to get around this with Python attributes, module globals or caches. They will not work under `jit`, and they break the rule. Reading the opponent's actions **within the current hand** from `obs.history` is allowed and encouraged.
2. **About 50 lines of decision code per version.** Treat it like an essay word count: around 50, give or take 10%, is fine. Count the non-blank, non-comment lines of `act` plus any helper methods or functions your version file defines. Do not count imports, docstrings, `__init__`, `init_state` or `metadata`. The provided `equity.py` helpers do not count either. One statement per line. No semicolons, no stacking several decisions into one expression to save lines, no golfing. More lines are room for more poker ideas, not for denser code.
3. **Readable at a glance.** Use descriptive names. Each line should say one poker idea ("call if equity beats pot odds"). Add a short comment where the poker reasoning is not obvious. Group the code by situation (preflop as button, preflop as big blind, postflop) so the reviewer can find each part. The reviewer should understand what the bot does in a few minutes.
4. **Interpretable parameters.** Every tunable number is a constructor argument with a default, a descriptive name and a unit in its name or comment (big blinds, pot fraction, equity probability). You may tune them with the benchmark. Keep the count small enough that each one means something. No lookup tables of fitted numbers (for example a 169-entry preflop chart); a short rule that a human can read is fine.
5. **Compute cap.** At most 256 Monte Carlo samples per decision in total, and no loops longer than that. Under `vmap`, every bot's `act` runs at every step at every table.
6. **No bot-specific exploits.** You may read the benchmark bots in `opponents/` to understand them, and you may borrow ideas from them (say so in the README). Do not detect or key on any particular opponent's patterns, such as its exact bet sizes. Your bot will face opponents you have not seen.

## Sandbox rules (strict)

- Your bot directory `/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_2_suneater/bassander_lin_glm/` is your whole workspace. Create, modify and read files **only** inside it. Do not read, list or modify anything outside it, including the engine source, other bots and other lineages, and git state. Do not commit.
- The single exception is the project's Python environment, which you use only by running `uv run python ...` from inside your directory (uv finds the project on its own). If `uv` is unavailable, use `../../../.venv/bin/python`. Do not install packages or change the environment.
- Do not edit `equity.py`, `bench.py` or anything in `opponents/`. They are shared with other lineages and must stay identical.
- Import only `jax`, `jax.numpy`, the `poker_bots` package, and your own package's files (`from .equity import ...`).
- No import-time side effects, no monkeypatching, no reading engine internals at runtime, no access to anything other than `obs` and `key` inside `act`. Any attempt to reach the deck, the opponent's cards or the engine's state disqualifies the bot.
- The environment already has what it needs: Python 3.13, `jax` 0.11, and `poker_bots` at engine commit `02abc49` plus working-tree changes (engine_sha256 `295477eff7637056`). The contract below describes this exact engine, so you do not need to read its source.

## What is already in your directory

```
/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_2_suneater/bassander_lin_glm/
    PROMPT.md       # this prompt
    prompt.toml     # the values that filled it (ignore)
    __init__.py     # placeholder; you edit it at the end (see Deliverables)
    equity.py       # shared helpers (do not edit; see "Helpers" below)
    bench.py        # heads-up benchmark script (do not edit)
    opponents/      # the four phase-1 finals, one self-contained file each (do not edit)
        alan_wake_claude.py
        jack_joyce_gemini.py
        jesse_faden_glm.py
        max_payne_gpt.py
```

## Deliverables

```
/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_2_suneater/bassander_lin_glm/
    __init__.py     # one line: from .vN import Bot as DeployedBot  (N = your final version)
    v1.py           # each version is a separate file defining `class Bot`
    v2.py
    ...
    README.md
```

- Each `vN.py` defines exactly one class named `Bot`, may import from `.equity`, and must not import other versions. **Never edit a version after benchmarking it.** Changes go in a new file.
- Make `v1` a simple, sound heads-up baseline. Each later version changes one idea, or one closely related group of ideas, so results can be attributed.
- Keep scratch files (benchmark logs, notes) inside your directory if you need them. Mention them in the README.

### README.md

1. **Header:** bot name, lineage, model (your own name and version, as best you know it), prompt version `botgen-v2`, engine (`02abc49` + working tree, engine_sha256 `295477eff7637056`), wall-clock time used, and whether the code was executed.
2. **One section per version:**
   - *What it does:* two to four plain sentences describing the strategy.
   - *Why it exists:* what weakness in the parent it targets, or what hypothesis it tests.
   - *Parameters:* each with its value and unit, and how it was chosen (default, reasoning, or tuned against what).
   - *Results:* a table with columns opponent, seed, tables × hands, bb/100, SE.
   - *Verdict:* kept as parent of the next version, or discarded, and why.
   - *Line count* of the decision code.
3. **Final choice:** which version is deployed and why, its held-out results against every benchmark, and known weaknesses. If the best-scoring version is not the deployed one (for example because of clarity or a noisy win), say why.

## Metadata (provenance)

Every version's `metadata()` returns a JSON-serialisable dict (only str, int, float, bool, None, lists and dicts). It is written into every simulation log header:

```python
def metadata(self) -> dict:
    return {
        "bot_name": "bassander_lin_glm",
        "version": 2,                      # this file's N
        "parent": 1,                       # version this was derived from, or None for v1
        "lineage": "phase_2_suneater/bassander_lin_glm",
        "model": "<your model name and version>",
        "prompt_version": "botgen-v2",
        "phase": "phase_2_suneater",
        "engine": "02abc49+wt sha256:295477eff7637056",
        "params": {"raise_equity": self.raise_equity, ...},   # every constructor arg and its value
        "contributions": [
            # include each block below verbatim, in order
            {"role": "requester", "who": "a"},
            {"role": "generator", "interface": "GLM"}
        ],
    }
```

## Benchmark protocol

Run `bench.py` from inside your bot directory:

```
cd /Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_2_suneater/bassander_lin_glm
uv run python bench.py v1:Bot random --tables 2 --hands 20     # smoke test: imports and compiles
uv run python bench.py v1:Bot finals --seed 0                  # against all four phase-1 finals, plus their mean
uv run python bench.py v2:Bot alan_wake_claude --seed 1        # against one phase-1 final
uv run python bench.py v2:Bot v1:Bot --seed 2                  # against your own earlier version
uv run python bench.py v1:Bot random --seed 0                  # sanity check
```

- The format is heads-up (two seats). Your candidate sits in seat 0 on even tables and seat 1 on odd tables, so it plays both sides of the button equally. Stacks are 100bb with rebuys on bust and cash-out above 300bb, blinds 1/2.
- The script prints bb/100 and a **table-clustered** standard error: the spread of per-table results divided by √tables. Defaults are 64 tables × 250 hands = 16,000 hands per opponent. `finals` runs four opponents in sequence.
- Expect heads-up SEs of roughly 20 to 40 bb/100 at the default size, much wider than in 6-max. Differences of under about 2 SE are noise. Do not keep a change because of one, and do not tune parameters by chasing differences inside the error bars. When two versions are close, use `--tables 128` or a second seed rather than more versions.
- Timing depends on the machine. Time your first `finals` run and budget the hour from it. Fewer, informative runs beat many noisy ones.
- **Iterate with seeds 0 to 9 only.**
- **Choose the final version on held-out seed 1000**, run once per finalist with `finals` (and `random` as a sanity check). Report it in the README.
- The primary score is the mean bb/100 against the four phase-1 finals. Also check each finalist against its parent: a version that beats the benchmarks by exploiting their 6-max habits but loses to its own parent is a warning sign, since the other phase-2 lineages will not share those habits.

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

## Engine and bot contract (engine 02abc49 + working-tree changes, engine_sha256 295477eff7637056)

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
- The engine also supports an optional `observe(record, state, key) -> state` method, called at the end of every hand. **Do not define `observe` in this phase.** The engine calls it whenever it exists, and it only makes sense with persistent state, which is not allowed here.

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
| `hand_count` | `() int32` | Hands completed at this table so far. Not useful without state; listed for completeness. |

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

`evaluate_hands(cards)` takes an int32 array of shape `(..., 5)`, `(..., 6)` or `(..., 7)` holding **only valid cards** (no `-1`) and returns an int32 strength per hand; higher is better, equal means a tie. Strengths are comparable across 5, 6 and 7 cards. Category is `strength // 13**5` (0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 full house, 7 quads, 8 straight flush). It is a table lookup and is cheap; batch it over a leading axis.

Because the number of dealt board cards depends on the street (a traced value), the easy way to get your current made hand is the `made_hand` helper in `equity.py`, which evaluates the 5-, 6- and 7-card cases and selects by street. For equity, complete the board by sampling (the `equity` helper does this). There is no preflop strength table. Passing `-1` padding to `evaluate_hands` gives garbage silently.

To sample unseen cards without Python control flow: build a `(52,)` boolean "known" mask from `hole_cards` and the valid entries of `board`, then e.g. take `jax.random.permutation` of `jnp.arange(52)`, stable-sort known cards to the end (`jnp.argsort(known[perm], stable=True)`), and read the first k.

### What the bot never sees

- Anything after its own last decision in a hand: later actions, the showdown, opponents' hole cards, who won. The bot is only called when it must act.
- Hands it is not asked to act in at all.
- Rebuys and cash-outs (a stack can jump between hands).
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

### Heads-up specifics

Every table in this phase has `max_players = 2`, so per-seat arrays have length 2 and seat 1 is always your opponent.
- The button posts the small blind, acts **first preflop** and **last on every later street**. The big blind acts last preflop and first postflop.
- `obs.button == 0` means you are the button (small blind); `obs.button == 1` means you are the big blind.
- Stacks are 100bb at the start, carry over between hands, are topped back up to 100bb after busting, and are capped at 300bb by cash-out. Effective stack is `jnp.minimum(obs.stacks[0] + obs.contributed[0], obs.stacks[1] + obs.contributed[1])`.
- Score: bb/100 from the candidate's seat, `(ending stack - starting stack)` per hand, excluding rebuys and cash-outs.

---
