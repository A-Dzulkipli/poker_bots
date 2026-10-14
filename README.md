# poker_bots

## Quickstart: play against the bots

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python 3.13 for you if it's missing.

    git clone <this repo> && cd poker_bots
    uv run --group ui python -m ui

The first run installs the dependencies, then opens http://127.0.0.1:8000. Pick opponents and seats, then
press Sit down. More options and the replay viewer are in [ui/README.md](ui/README.md).

## Bot generations

Each phase gives independently prompted LLM lineages the same brief (the `botgen_vN_prompt.md` in each phase folder). They design, benchmark and iterate inside their own folder, then freeze one `<lineage>_final.py`. Every phase after the first is benchmarked against all earlier finals. From phase 4 on, the score is a strength-weighted mean bb/100 over that field. In phases 1 to 6, `act` may draw at most 256 Monte Carlo samples per decision, and randomness must come only from the bot's `key`.

| Phase | Bots | Format and time | Instructions and restrictions |
|---|---|---|---|
| 1 `phase_1_remedy` | alan_wake_claude, jack_joyce_gemini, jesse_faden_glm, max_payne_gpt | 6-max, 15 minutes | Strongest stateless bot in about 20 lines of decision code. Readable at a glance; every tunable number is a named constructor argument. No memory across hands. |
| 2 `phase_2_suneater` | bassander_lin_glm, hadrian_marlowe_gpt, valka_onderra_claude | heads-up from here on, 60 minutes | About 50 lines and stateless. May read actions within the current hand. No fitted lookup tables such as a 169-hand chart. May read the phase-1 bots' code, but may not key on any opponent's patterns. |
| 3 `phase_3_dota` | invoker_gpt, meepo_claude, windrunner_glm | 60 minutes | About 100 lines and stateless. Code is grouped by situation. Randomised bluffing and mixing is allowed and expected. The `aggressor` probe joins the benchmark to punish over-folding. |
| 4 `phase_4_sea_monsters` | kraken_gpt, leviathan_claude, megalodon_glm | 60 minutes, 250-hand sessions | State may persist within a 250-hand session against one opponent, and `observe` sees every finished hand, so opponent tracking is allowed. State holds named summary statistics with priors, not a log of hands. Adapt to behaviour, never to identity. About 100 lines, tracking included. Scored by the weighted score. |
| 5 `phase_5_cowboy_bebop` | faye_gpt, spike_claude, vicious_glm | 60 minutes | Postflop decisions must use board texture and draws. Tracking is allowed but not required. Design from first principles: the prompt does not describe earlier bots, and their code is hidden. About 100 lines. |
| 6 `phase_6_trails` | estelle_claude, joshua_gpt, scherezard_glm | 60 minutes | Decisions must rest on an explicit opponent range built from the hand's actions and narrowed street by street. No memory between hands; `bench.py` enforces this. Range evaluations count toward the 256-sample cap. About 100 lines. |
| 7 `phase_7_lemonade_stand` | aiden_claude, doug_gpt | 12 hours, in progress | No limits on code length, readability, parameters or method; state and opponent modelling are optional. The one hard limit is speed: at most 6× the time random takes against kraken (`bench.py speed`; about 50 s per 16,000 hands on an M5 Pro). Memory under 8 GB. Precomputed data up to 50 MB, built by a kept script. Still adapts to behaviour, not identity. |

Other bots: `probes/aggressor` opens every button and bets whenever checking is free, and continues against bets only with equity. `rubick/` holds meta-bots that switch between finals by hole cards. `random` is the engine's RandomBot.

## Head to head

The phase-6 heads-up round robin covers every final through phase 6. Each pair plays 32,000 hands (seeds 0 and 1, 64 tables × 250 hands). Cells are bb/100 for the row bot against the column bot. Italic cells are within 2 standard errors (table-clustered) of zero. Rows are sorted by the phase-6 weighted score. Data and SEs are in `experiments/phase_6_trails/results/pairs.parquet`; per-phase write-ups are in each `experiments/` folder.

| # | bot | phase | weighted | kraken | faye | leviathan | megalodon | windrunner | spike | invoker | joshua | estelle | meepo | scherezard | hadrian | vicious | valka | aggressor | alan | bassander | max | jesse | jack | random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | kraken_gpt | 4 | +56.5 | · | _−5_ | +41 | +39 | +53 | +52 | +33 | +35 | +36 | +50 | +42 | +66 | +75 | +96 | +50 | +156 | +148 | +46 | +202 | +196 | +205 |
| 2 | faye_gpt | 5 | +50.9 | _+5_ | · | +60 | +58 | +63 | +44 | +38 | +35 | +60 | +47 | +27 | +51 | +92 | +59 | +21 | +69 | +97 | +47 | +132 | +121 | +106 |
| 3 | leviathan_claude | 4 | +38.1 | −41 | −60 | · | _−8_ | +13 | _+10_ | +19 | _−4_ | _+8_ | +17 | +18 | +57 | +61 | +84 | +51 | +111 | +168 | +51 | +274 | +344 | +531 |
| 4 | megalodon_glm | 4 | +36.4 | −39 | −58 | _+8_ | · | _+2_ | _+11_ | _+12_ | _−4_ | _+23_ | +22 | +41 | +48 | +68 | +101 | +49 | +108 | +165 | +55 | +213 | +302 | +406 |
| 5 | windrunner_glm | 3 | +32.6 | −53 | −63 | −13 | _−2_ | · | _+3_ | _+10_ | _−9_ | −26 | +22 | +28 | +61 | +77 | +86 | +37 | +96 | +154 | +50 | +203 | +281 | +399 |
| 6 | spike_claude | 5 | +31.5 | −52 | −44 | _−10_ | _−11_ | _−3_ | · | _+5_ | _+1_ | _+12_ | +35 | +20 | +31 | +87 | +99 | −41 | +111 | +152 | +97 | +277 | +317 | +369 |
| 7 | invoker_gpt | 3 | +22.9 | −33 | −38 | −19 | _−12_ | _−10_ | _−5_ | · | _+7_ | +30 | +35 | _+3_ | +20 | +44 | +26 | _+9_ | +71 | +120 | +65 | +244 | +245 | +367 |
| 8 | joshua_gpt | 6 | +21.9 | −35 | −35 | _+4_ | _+4_ | _+9_ | _−1_ | _−7_ | · | +46 | +19 | _+17_ | _+10_ | +60 | +57 | −45 | +72 | +106 | +80 | +266 | +292 | +356 |
| 9 | estelle_claude | 6 | +19.5 | −36 | −60 | _−8_ | _−23_ | +26 | _−12_ | −30 | −46 | · | −38 | −45 | _−4_ | +76 | _−27_ | _+7_ | +51 | +132 | +53 | +323 | +412 | +690 |
| 10 | meepo_claude | 3 | +17.4 | −50 | −47 | −17 | −22 | −22 | −35 | −35 | −19 | +38 | · | _−13_ | _+1_ | +57 | +46 | −24 | +59 | +123 | +59 | +278 | +337 | +457 |
| 11 | scherezard_glm | 6 | +13.5 | −42 | −27 | −18 | −41 | −28 | −20 | _−3_ | _−17_ | +45 | _+13_ | · | _−4_ | +50 | _+11_ | −19 | +57 | +46 | +58 | +241 | +259 | +448 |
| 12 | hadrian_marlowe_gpt | 2 | +13.4 | −66 | −51 | −57 | −48 | −61 | −31 | −20 | _−10_ | _+4_ | _−1_ | _+4_ | · | +28 | +43 | −22 | +119 | +59 | +67 | +279 | +298 | +334 |
| 13 | vicious_glm | 5 | +5.5 | −75 | −92 | −61 | −68 | −77 | −87 | −44 | −60 | −76 | −57 | −50 | −28 | · | _+2_ | _−3_ | +67 | +117 | +70 | +315 | +389 | +650 |
| 14 | valka_onderra_claude | 2 | +1.8 | −96 | −59 | −84 | −101 | −86 | −99 | −26 | −57 | _+27_ | −46 | _−11_ | −43 | _−2_ | · | +18 | +84 | +63 | +45 | +270 | +312 | +479 |
| 15 | aggressor | probe | −3.5 | −50 | −21 | −51 | −49 | −37 | +41 | _−9_ | +45 | _−7_ | +24 | +19 | +22 | _+3_ | −18 | · | −148 | +19 | +48 | +150 | +138 | +86 |
| 16 | alan_wake_claude | 1 | −24.5 | −156 | −69 | −111 | −108 | −96 | −111 | −71 | −72 | −51 | −59 | −57 | −119 | −67 | −84 | +148 | · | −55 | +75 | +163 | +180 | +327 |
| 17 | bassander_lin_glm | 2 | −29.6 | −148 | −97 | −168 | −165 | −154 | −152 | −120 | −106 | −132 | −123 | −46 | −59 | −117 | −63 | −19 | +55 | · | +48 | +243 | +333 | +692 |
| 18 | max_payne_gpt | 1 | −35.2 | −46 | −47 | −51 | −55 | −50 | −97 | −65 | −80 | −53 | −59 | −58 | −67 | −70 | −45 | −48 | −75 | −48 | · | _+5_ | +129 | +285 |
| 19 | jesse_faden_glm | 1 | −146.5 | −202 | −132 | −274 | −213 | −203 | −277 | −244 | −266 | −323 | −278 | −241 | −279 | −315 | −270 | −150 | −163 | −243 | _−5_ | · | +108 | +697 |
| 20 | jack_joyce_gemini | 1 | −195.4 | −196 | −121 | −344 | −302 | −281 | −317 | −245 | −292 | −412 | −337 | −259 | −298 | −389 | −312 | −138 | −180 | −333 | −129 | −108 | · | +718 |
| 21 | random | builtin | −439.0 | −205 | −106 | −531 | −406 | −399 | −369 | −367 | −356 | −690 | −457 | −448 | −334 | −650 | −479 | −86 | −327 | −692 | −285 | −697 | −718 | · |

## Engines

There are two engines with the same rules. Both run with the default no-limit hold'em cash-game settings.

- **`src/poker_bots/poker_game.py` (JAX):** the main engine. Many tables run at once under `jit` and `vmap`. Every phase, benchmark, tournament and the UI uses it.
- **`src/poker_bots/pure_game.py` (plain Python):** one table, stepped in ordinary Python, for people who want to write a bot without JAX. `tests/test_pure_game.py` drives it in lockstep with the JAX engine, on the same decks and actions. Every event, observation and hand record has to match exactly, across ring games, heads-up, busting, rebuy caps and short stacks. It runs about 600–800 hands/s for a 6-max table. Bots for one engine don't run in the other: a pure-Python bot has to be ported to JAX before it can join the benchmarks.

### Rules

- **Stacks:** blinds are 1/2 and everyone starts with 100bb. Stacks persist between hands. A seat that ends a hand at or below `min_stack` (default 0) rebuys to its buy-in, and anything above `max_stack` (default 300bb) is cashed out. `rebuy=False` turns rebuys and cash-outs off, so busted players leave and the table runs until one player is left.
- **Blinds:** the big blind moves to the next live seat, the old big blind posts the small blind, and the old small blind takes the button (TDA dead button). A seat that busted can therefore hold the button or the small blind, and its blind is then not posted. Heads-up, the button posts the small blind. A short blind posts what it has, but the amount to call is still the full big blind.
- **Actions:** each action is `ActionData(type, value)`.
  - `FOLD` always folds, even when a check is free. `NO_OP` is treated as a fold.
  - `CHECK_CALL` checks or calls, capped at the stack. Any unrecognised type is also treated as a call.
  - `RAISE` takes `value` as a raise-to total: the seat's contribution on this street after the raise. It is clipped to `[raise_lower_bound, raise_upper_bound]`, and becomes a call when `can_raise` is false.
  - A short all-in raise doesn't reopen the betting unless all-ins together add up to a full raise.
- **Showdown:**
  - Side pots are split by commitment level. An odd chip goes to the first winner after the button.
  - Every hand still live at showdown is revealed.
  - Hands are ranked with lookup tables (`hand_eval.py`): a rank-multiset table plus a flush-mask table, for 5 to 7 cards. A higher score wins, and `score // 13**5` is the hand category.
- **Logs:** both engines record the requested action next to the applied one (`requested_type` and `requested`, then `action_type` and `amount`), so clamped and converted actions are visible in the logs.

### Design (JAX engine)

- **Configuration:** `make_poker_env(bots, max_players, max_actions, small_blind, big_blind, rebuy)` is a factory. Everything passed to it is a compile-time constant held in a closure, and it returns a `PokerEnv` of functions.
- **State:** everything else lives in the `TableState` pytree. That includes the deck, a fixed-size action history, stacks, positions, the hand count, the PRNG key, and `seat_bot`, which maps each seat to a bot type.
- **Bot state:** each bot type's state is stacked over seats (leading `max_players` axis). `act` is dispatched per type with `lax.switch` on `seat_bot[current_player]`. Each branch reads its seat's slice, calls the bot, and writes the slice back.
- **Stepping:** `next_event` returns 0 (table finished), 1 (hand over), 2 (street over) or 3 (player to act). `step_with_record(state, action=None)` advances one event and returns `(state, hand_record, hand_ended)`. When `action` is given, it is used on a player-to-act step instead of calling the bot; the UI uses this for the human seat. `get_observation(state, seat)` gives any seat's view.
- **Batch runs:** `run_hands(state, n)` is a `while_loop` that collects `n` `HandRecord`s, each with every hole card, the full board, the history, payouts, rebuys and cash-outs. `simulate.py` writes these out as JSONL with a header, plus a `.summary.json` with bb/100 ± SE per bot type.
- **Observations** are seat-relative. Index 0 is the observing seat, and seat numbers count onward from it. Each bot sees only its own hole cards and the board dealt so far.

## Bot contract

### JAX bots (main lane)

```python
class Bot:
    def init_state(self, key) -> S: ...                                  # any fixed-shape pytree, or None
    def act(self, obs: PlayerTableState, state: S, key) -> tuple[ActionData, S]: ...
    def metadata(self) -> dict: ...                                      # e.g. {"bot_name": ..., "kind": "bot"}
    # optional, called at the end of every hand for each seat this bot occupies
    def observe(self, record: HandRecord, state: S, key) -> S: ...
```

- **Traced code:** `act` and `observe` run under `jit` and `vmap`. You can't use a Python `if` on array values; use `jnp.where` or `lax.cond` instead. Shapes must be static, and printing goes through `jax.debug.print`, or run the UI with `--eager`.
- **State:** the engine stores each seat's `S` and passes it back on the next call. Keep `self` for constants. `S` must keep the same shape and dtype on every call, and it persists across hands.
- **What `observe` sees:** the seat's relative `HandRecord`. Opponents' hole cards and every hand rank are -1 unless shown at showdown, undealt board cards are -1, and `seat_bot` is -1. Bots that don't define `observe` cost nothing, and their results are unchanged.
- **Observation fields (`PlayerTableState`):**
  - Cards: `hole_cards`, `board`.
  - Seats: `stacks`, `seated`, `in_hand`, `contributed` (this street), `button`, `sb_seat`, `bb_seat`.
  - Betting: `history` (an `ActionRecord` buffer, with `is_valid` marking real entries), `current_round`, `min_raise`, `current_bet`, `pot` (all chips put in so far), `can_raise`, `raise_lower_bound`, `raise_upper_bound`.
  - Table: `big_blind`, `hand_count`.
- **Cards** are ints 0–51 as `suit * 13 + rank`. Rank 0 is a 2 and 12 is an ace; suits run d, c, h, s. -1 means unknown.
- **Useful formulas:**
  - Chips to call: `to_call = min(obs.current_bet - obs.contributed[0], obs.stacks[0])`.
  - Hand strength: `hand_eval.evaluate_hands(cards[..., 5..7])` returns a comparable int.
- **Where bots live:** JAX bots go in `bots/` and plain-Python bots in `pure_bots/`. Each tree has its own registry with the same layout and names, so a name always says which engine the bot needs. `bots/registry.py` finds bots by file layout. `bots/<folder>/<lineage>/vN.py` with a class named `Bot` is registered as `<folder>/<lineage>/vN`, and a frozen `bots/<folder>/<lineage>_final.py` as `<folder>/<lineage>`. Any folder with an `__init__.py` is scanned, and the UI lists new bots straight away. To benchmark a bot, copy a lineage's `bench.py` and `opponents/` from the latest phase.

### Plain-Python bots (`pure_game.py`)

```python
from poker_bots.poker_game import Action, ActionData
from poker_bots.pure_game import Table, RandomBot

class Bot:
    def act(self, obs) -> ActionData: ...        # obs: pure_game.Observation, same fields as above as plain ints and lists
    def observe(self, record) -> None: ...       # optional, same masking as the JAX hook
    def metadata(self) -> dict: ...              # optional

table = Table([Bot(), RandomBot(seed=1), None, RandomBot(seed=2)], seed=0)   # None is an empty seat
records = table.run(1000)                                                     # list[HandRecord]
```

- **Where bots live:** `pure_bots/registry.py` scans `pure_bots/` with the same layout as `bots/`. `pure_bots/<folder>/<lineage>/vN.py` with a class named `Bot` is registered as `<folder>/<lineage>/vN`, and `pure_bots/<folder>/<lineage>_final.py` as `<folder>/<lineage>`. `builtin/random` is `pure_game.RandomBot`. Use it with `from pure_bots.registry import get_bot`. The UI, tournaments and `bench.py` only read `bots/`.
- **Instances and state:** each seat gets its own instance. State lives on `self`, and randomness comes from the bot's own `random.Random`.
- **Observation:** `Observation.history` is a list of `ActionRecord`s for the current hand, and `obs.to_call` is provided.
- **Actions and rules:** the same as the JAX engine above. `table.step(action)` and `table.observation(seat)` mirror `step_with_record` and `get_observation`, for stepping one event at a time.
