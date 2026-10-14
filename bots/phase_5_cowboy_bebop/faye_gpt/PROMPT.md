# Task: write a heads-up poker bot for a JAX No-Limit Hold'em simulator

You are designing and iterating on a poker bot for a research simulator. Everything you need is in this prompt and in your bot directory. Read all of it before writing code.

## Your assignment

- **Bot name:** `faye_gpt` (snake_case; it is also your directory and Python package name)
- **Lineage:** phase_5_cowboy_bebop/faye_gpt
- **Bot directory:** `/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_5_cowboy_bebop/faye_gpt/`. It already exists and holds everything you need. All your work goes there.
- **Benchmark set:** The fifteen-bot field (`field`), which `bench.py` loads from `../field/` and you must not read, scored by the weighted score, which is your primary score; and every earlier version of your own bot (`vK:Bot`).
- **Additional constraints:** None beyond the restrictions below.

Design a bot, benchmark it, improve it over several versions, and pick one final version for deployment. Aim for the highest **weighted score** (defined below) you can reach **heads-up** under the restrictions below. This phase's requirement: your bot's postflop play must take **board texture and draws** into account.

The strategy is yours to design from first principles. This prompt deliberately does not describe how earlier bots play or which approaches worked, and the benchmark opponents' code is not available to you. Nothing here requires a particular method; the helpers below are optional conveniences.

## First two minutes: check that you can run code

Before anything else, run this from your bot directory:

```
uv run python bench.py check
```

It prints the Python, `jax` and engine it found, checks the engine hash against this contract, and plays a 40-hand smoke test.
- **If it ends with `OK`,** you can run code. Follow the benchmark protocol below.
- **If your shell runs in a Linux VM** and `uv` complains about the project's `.venv` (for example a Python link that does not exist, or a `.venv/.lock` it may not remove), you are pre-approved to build a separate environment outside the repository. Run every command as `UV_PROJECT_ENVIRONMENT=$HOME/p5venv uv run --frozen python bench.py ...` from your bot directory. This is the only installing you may do, it must not touch the repository's `.venv` or `uv.lock`, and you continue only if `check` then reports the engine hash `295477eff7637056`. Start your hour when `check` passes, and say in the README header that you used this environment.
- **If it still fails,** stop and tell the requester the exact output; do not spend the hour writing unbenchmarked code. Continue without execution only if the requester explicitly tells you to, and then follow "If you cannot run code" below.

## The benchmark field

The field is fifteen frozen bots from earlier phases: four phase-1, three phase-2, three phase-3 and three phase-4 finals, a probe bot called `aggressor`, and `random` (uniformly random legal actions). The three phase-4 bots keep state across hands and adapt to their opponent within a session; the others do not. Your final version will also face the other phase-5 lineages, written independently from this same prompt.

The opponents are black boxes: `bench.py` loads them from a shared folder outside your directory, which you must not read (see the sandbox rules). Learn about them only by playing them.

## The weighted score

Each opponent has a strength rank r from 1 (weakest) to 15 (strongest), from the order of its mean result against non-random opponents in the phase-4 round robin:

| rank | opponent | weight of a win | weight of a loss |
|---|---|---|---|
| 15 | kraken_gpt | 1.88 | 0.13 |
| 14 | leviathan_claude | 1.75 | 0.25 |
| 13 | megalodon_glm | 1.63 | 0.38 |
| 12 | windrunner_glm | 1.50 | 0.50 |
| 11 | invoker_gpt | 1.38 | 0.63 |
| 10 | meepo_claude | 1.25 | 0.75 |
| 9 | hadrian_marlowe_gpt | 1.13 | 0.88 |
| 8 | valka_onderra_claude | 1.00 | 1.00 |
| 7 | aggressor | 0.88 | 1.13 |
| 6 | alan_wake_claude | 0.75 | 1.25 |
| 5 | bassander_lin_glm | 0.63 | 1.38 |
| 4 | max_payne_gpt | 0.50 | 1.50 |
| 3 | jesse_faden_glm | 0.38 | 1.63 |
| 2 | jack_joyce_gemini | 0.25 | 1.75 |
| 1 | random | 0.13 | 1.88 |

A win (positive bb/100) against an opponent of rank r counts r / 8; a loss counts (16 − r) / 8. The weighted score is the mean of the fifteen weighted results, so beating strong bots is worth the most and losing to weak ones costs the most. `bench.py ... field` prints it. For calibration, the strongest existing bots score kraken_gpt +65.7, leviathan_claude +51.5, megalodon_glm +48.1 and windrunner_glm +43.3 on this measure.

## Board texture and draws (required)

Your postflop decisions must depend on the board and on what your hand can become, not only on how strong your hand is right now. At a minimum, consider:
- **Board texture:** what the community cards make possible for either player, for example paired boards, two or more cards of one suit, connected ranks, and high cards, and how that changes as the turn and river arrive.
- **Draws:** what your own hand can improve to, for example flush and straight draws, overcards, and how many cards improve it, and how that changes on later streets.

Which features you compute, how you measure them and which decisions they change is up to you. The card encoding is in "Cards" below. Report it in the README (see Deliverables), including evidence of what it is worth.

## Opponent tracking

As in phase 4, `state` persists across hands within a session and you may define `observe` (see the contract). Tracking is allowed, not required.

## Time box: 60 minutes of wall-clock time

- When you start, record the time. If your environment lets you schedule a reminder, timer or self-ping, set one for 50 minutes from now that says "wrap up now". Otherwise check elapsed time (e.g. `date`) between steps.
- At 50 minutes, stop iterating, whatever state you are in. Use the remaining time to run the held-out evaluation if it is not done, pick the final version, finish the README and check the files.
- Record the time you actually used in the README.

## If you cannot run code

Some environments (browser chats) cannot execute code at all. If `bench.py check` cannot run or fails, and the requester tells you to continue anyway:
- Still produce the full set of files. Iterate by reasoning: review each version for contract violations and strategic leaks, and justify each change.
- In the README, mark every benchmark field as `not run`. **Never invent benchmark numbers.** State clearly at the top of the README that the code has not been executed, and include the output of `bench.py check` if you have it.

## Restrictions for this generation

1. **About 100 lines of decision code per version.** Treat it like an essay word count: around 100, give or take 10%, is fine. Count the non-blank, non-comment lines of `act`, `observe` and any helper methods or functions your version file defines. Do not count imports, docstrings, `__init__`, `init_state` or `metadata`. The provided `equity.py` helpers do not count either. One statement per line. No semicolons, no stacking several decisions into one expression to save lines, no backslash-continued expressions, no golfing. You do not have to use all the lines. State the count in the README.
2. **Board texture and draws must change postflop decisions,** as described above.
3. **Readable at a glance.** Use descriptive names. Each line should say one poker idea. Add a short comment where the poker reasoning is not obvious. Group the code by situation and give each group a heading comment, so the reviewer can find each part. Name every feature and tracked statistic for what it measures (`flush_draw_outs`, not `f[2]`). The reviewer should understand what the bot does in a few minutes.
4. **Interpretable parameters.** Every tunable number is a constructor argument with a default, a descriptive name and a unit in its name or comment. You may tune them with the benchmark. Keep the count small enough that each one means something. No lookup tables of fitted numbers (for example a 169-entry preflop chart); a short rule that a human can read is fine.
5. **State, if you use it,** is a fixed-shape pytree of `jax.Array`s updated only in `observe` and `act`, holding summary statistics (counts, rates), not a log of hands. Do not store random keys. Do not use Python attributes, module globals or caches as memory.
6. **Compute cap.** At most 256 Monte Carlo samples per decision in total, and no loops longer than that. Keep `observe` cheap (counting, no Monte Carlo). Under `vmap`, every bot's `act` runs at every step at every table.
7. **Randomness only from `key`,** with any frequency given a named parameter.
8. **Adapt to behaviour, not identity.** Do not try to recognise which bot you are playing (from bet sizes, timing or any other fingerprint), and do not switch to a strategy written for a specific opponent.

## Sandbox rules (strict)

- Your bot directory `/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_5_cowboy_bebop/faye_gpt/` is your whole workspace. Create and modify files **only** inside it, and read only files inside it. Do not read, list or modify anything outside it: not the benchmark opponents in `/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_5_cowboy_bebop/field/`, not the engine source, other bots, other lineages, earlier phases, experiment results or git state. Do not commit.
- Your session may have been given access to the whole repository so that it can run code. That does not widen what you may read or change. If you read anything outside your directory by mistake, say so in the README.
- The single exception is the project's Python environment, which you use only by running `uv run python ...` from inside your directory (or the pre-approved separate environment above). If `uv` is unavailable, use `../../../.venv/bin/python`. Do not install packages or change the environment otherwise.
- Do not edit `equity.py` or `bench.py`. They are shared with other lineages and must stay identical.
- Import only `jax`, `jax.numpy`, the `poker_bots` package, and your own package's files (`from .equity import ...`).
- No import-time side effects, no monkeypatching, no reading engine internals at runtime, no access to anything other than `obs`, `record`, `state` and `key` inside `act` and `observe`. Any attempt to reach the deck, the opponent's hidden cards or the engine's state disqualifies the bot.
- The environment already has what it needs: Python 3.13, `jax` 0.11, and `poker_bots` at engine commit `decafb9` (engine_sha256 `295477eff7637056`). The contract below describes this exact engine, so you do not need to read its source.

## What is already in your directory

```
/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_5_cowboy_bebop/faye_gpt/
    PROMPT.md       # this prompt
    prompt.toml     # the values that filled it (ignore)
    __init__.py     # placeholder; you edit it at the end (see Deliverables)
    equity.py       # shared helpers (do not edit; see "Helpers" below)
    bench.py        # heads-up benchmark script (do not edit); loads the opponents from ../field/
```

## Deliverables

```
/Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_5_cowboy_bebop/faye_gpt/
    __init__.py     # one line: from .vN import Bot as DeployedBot  (N = your final version)
    v1.py           # each version is a separate file defining `class Bot`
    v2.py
    ...
    README.md
```

- Each `vN.py` defines exactly one class named `Bot`, may import from `.equity`, and must not import other versions. **Never edit a version after benchmarking it.** Changes go in a new file.
- Each later version changes one idea, or one closely related group of ideas, so results can be attributed.
- Keep scratch files (benchmark logs, notes) inside your directory if you need them. Mention them in the README.

### README.md

1. **Header:** bot name, lineage, model (your own name and version, as best you know it), prompt version `botgen-v5`, engine (`decafb9`, engine_sha256 `295477eff7637056`), wall-clock time used, whether the code was executed, the environment used, and the output of `bench.py check`.
2. **Design:** in a short paragraph, the overall approach of the deployed version and why you chose it.
3. **One section per version:**
   - *What it does:* two to four plain sentences describing the strategy.
   - *Why it exists:* what weakness in the parent it targets, or what hypothesis it tests.
   - *Parameters:* each with its value and unit, and how it was chosen (default, reasoning, or tuned against what).
   - *Results:* a table with columns opponent, seed, tables × hands, bb/100, SE, plus the weighted score when you ran `field`.
   - *Verdict:* kept as parent of the next version, or discarded, and why.
   - *Line count* of the decision code.
4. **Board texture and draws:** which features the deployed version computes, how, which decisions each one changes and by how much, and the evidence that it helps: the deployed version against the same strategy with the features switched off (a version or a parameter setting that disables them), on the field and head to head.
5. **Final choice:** which version is deployed and why, its held-out results against every benchmark and its weighted score, and known weaknesses. If the best-scoring version is not the deployed one (for example because of clarity or a noisy win), say why.

## Metadata (provenance)

Every version's `metadata()` returns a JSON-serialisable dict (only str, int, float, bool, None, lists and dicts). It is written into every simulation log header:

```python
def metadata(self) -> dict:
    return {
        "bot_name": "faye_gpt",
        "version": 2,                      # this file's N
        "parent": 1,                       # version this was derived from, or None for v1
        "lineage": "phase_5_cowboy_bebop/faye_gpt",
        "model": "<your model name and version>",
        "prompt_version": "botgen-v5",
        "phase": "phase_5_cowboy_bebop",
        "engine": "decafb9 sha256:295477eff7637056",
        "params": {"some_param": self.some_param, ...},   # every constructor arg and its value
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
cd /Users/alik/Documents/Projects/poker_bots/poker_bots/bots/phase_5_cowboy_bebop/faye_gpt
uv run python bench.py check                                   # first: environment and engine check
uv run python bench.py v1:Bot random --tables 2 --hands 20     # smoke test for each new version
uv run python bench.py v1:Bot field --seed 0                   # all fifteen opponents, plus the weighted score
uv run python bench.py v2:Bot kraken_gpt --seed 1              # one opponent by name
uv run python bench.py v2:Bot v1:Bot --seed 2                  # against your own earlier version
```

- The format is heads-up (two seats). Your candidate sits in seat 0 on even tables and seat 1 on odd tables, so it plays both sides of the button equally. Stacks are 100bb with rebuys on bust and cash-out above 300bb, blinds 1/2.
- Each table is a 250-hand session (`--hands`), and state starts fresh at each table. Keep `--hands 250` for every result you report. Use `--tables` to change the sample size.
- The script prints bb/100 and a **table-clustered** standard error: the spread of per-table results divided by √tables. It also prints bb/100 over the first 50 hands of each session and over the rest. Defaults are 64 tables × 250 hands = 16,000 hands per opponent. `field` runs all fifteen opponents in sequence.
- Expect heads-up SEs of roughly 4 to 20 bb/100 at the default size, larger against wild opponents. Differences of under about 2 SE are noise. Do not keep a change because of one, and do not tune parameters by chasing differences inside the error bars. When two versions are close, use `--tables 128` or a second seed rather than more versions.
- Timing depends on the machine. Time your first `field` run and budget the hour from it. Single-opponent runs are a cheap way to test an idea before a full `field` run.
- **Iterate with seeds 0 to 9 only.**
- **Choose the final version on held-out seed 1000**, run once per finalist with `field`. Report it in the README.
- **The primary score is the weighted score against `field`.** Also check each finalist against its parent: a version that beats the benchmarks but loses to its own parent is a warning sign, since the other phase-5 lineages will not share the benchmarks' habits.

## Helpers in `equity.py`

Import them with `from .equity import ...`. They are tested against this engine. Using them is optional.

| Helper | Returns |
|---|---|
| `equity(obs, key, num_samples=128)` | Monte Carlo share of the pot won at showdown against opponents still in the hand, each holding **uniformly random** cards. A float in [0, 1]. Works on every street. 128 samples is half your budget. |
| `made_hand(obs)` | Strength of your best hand from hole cards plus the board so far (higher is better). `-1` preflop. Deterministic and cheap. |
| `hand_category(strength)` | Category of a `made_hand` value: 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 full house, 7 quads, 8 straight flush; `-1` preflop. |
| `to_call(obs)` | Chips needed to call (0 means checking is free), capped at your stack. |
| `pot_odds(obs)` | Equity needed to break even on a call: `to_call / (pot + to_call)`. |
| `num_live_opponents(obs)` | Opponents still in the hand (heads-up: 1 until someone folds). |
| `fold_or_check(obs)` | Folds only when there is something to call; otherwise checks. |
| `check_call()` | Check or call. |
| `raise_to(obs, total)` | Raise so this street's contribution becomes `total` chips, clipped to the legal range. Becomes a call when raising is not legal. |
| `choose(condition, if_true, if_false)` | Elementwise select between two actions (use instead of Python `if`). |

There are no board-texture or draw helpers; computing those is part of this phase's task.

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
