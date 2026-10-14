# Experiments

| Folder | Format | Contents |
|---|---|---|
| `phase_1_remedy/` | 1 (outcomes only) | 6-max round robin of the phase-1 finals. See its README. |
| `phase_2_suneater/` | 2 (full hand histories) | Heads-up round robin: phase-2 finals, phase-1 finals, RandomBot. |
| `phase_3_dota/` | 2 (full hand histories) | Heads-up round robin: phase-3 finals, phase-2 finals, the aggressor probe, phase-1 finals, RandomBot. Run `verify_finals.py` first. |
| `phase_4_sea_monsters/` | 2 (full hand histories) | Heads-up round robin: phase-4 finals (stateful, opponent tracking), phase-3, phase-2 and phase-1 finals, the aggressor probe, RandomBot, in 250-hand sessions. Run `verify_finals.py` first. Also writes `results/weighted.parquet`, the strength-weighted score over the phase-4 benchmark field. |
| `phase_5_cowboy_bebop/` | 2 (full hand histories) | Heads-up round robin: phase-5 finals plus the phase-4 roster, in 250-hand sessions. Run `build_finals.py` and `verify_finals.py` first. Writes `results/weighted.parquet` with the phase-5 ranks. |
| `phase_6_trails/` | 2 (full hand histories) | Heads-up round robin: phase-6 finals (stateless, opponent ranges) plus the phase-5 roster, in 250-hand sessions. Run `build_finals.py` and `verify_finals.py` first. Writes `results/weighted.parquet` with the phase-6 ranks. |
| `rubick_switch/` | 2 (full hand histories) | Rubick: builds the hole-card switching tables from phase-3 data (`build_tables.py`) and plays both Rubick variants against every finals bot on held-out seeds 2 and 3. |

Shared code:
- `hand_records.py` turns `env.run_hands` output into the three format-2 tables (`records_to_frames`).
- `tournament.py` holds the round-robin driver, `regenerate_block` and the summaries. A phase script only supplies its roster and `max_players`.

Both are meant to be imported by other consumers, such as the UI's play sessions and replay, so that every hand in the project is stored the same way.

## Storage format 2 (phase 2 onwards)

```
results/
    manifest.json                                format_version, config, engine hashes, bot sources and metadata, per-block timings
    seat_hands/<bot_a>__<bot_b>/seed_<s>.parquet  one row per seat per hand
    hands/<bot_a>__<bot_b>/seed_<s>.parquet       one row per hand
    actions/<bot_a>__<bot_b>/seed_<s>.parquet     one row per action, blinds included
    pairs.parquet, standings.parquet             derived summaries, rebuilt with --summarise
```

The rules are the same as format 1. Blocks are immutable zstd parquet files, and more hands means a new seed. Each block's `engine_sha256` is in the manifest, and pair names follow roster order. Every row starts with the block's key columns, then `table` and `hand`. A tournament's key columns are `pair` and `seed`. A play session uses `session` instead, with the same columns after it. `(key columns, table, hand)` identifies a hand in all three tables, and adding `seat` or `idx` identifies a row. Seats are **absolute** table seats, not the bot-relative seats of `PlayerTableState`. Chip amounts are integer chips; the big blind is `format.big_blind` in the manifest (2). Cards are strings such as `"AsKh"` (rank `23456789TJQKA`, suit `d c h s`), the same as `poker_bots.simulate.card_to_str`.

Hidden information is stored in full: every hole card and the whole runout, including cards no one saw. That is the point. A viewer that should hide cards hides them itself.

### `seat_hands`

| Column | Type | Meaning |
|---|---|---|
| `table`, `hand`, `seat` | u16, u32, u8 | |
| `bot`, `by` | str | Roster name of the seat's bot (`"human"` in a play session), and `"bot"` or `"human"` (from the bot's `metadata()["kind"]`). |
| `button`, `seats_from_button` | u8 | The button's seat, and this seat's distance clockwise from it. Heads-up: 0 is the button/small blind, 1 the big blind. |
| `starting_stack` | i32 | Stack at the start of the hand. |
| `net_chips` | i32 | Ending minus starting stack, before rebuys and cash-outs. This is the score. |
| `rebuy`, `cash_out` | i32 | Chips added after the hand (bust) or removed (above 300bb). |
| `showdown` | bool | Seat reached showdown. |
| `hole_cards`, `hand_class` | str | e.g. `"Kh7h"`, `"K7s"`; pairs are `"77"`, offsuit `"K7o"`. |
| `folded` | bool | Seat folded at some point in the hand. |
| `invested`, `payout` | i32 | Chips put in over the hand, and chips won back. `net_chips = payout - invested`. |
| `hand_rank` | i32, null | Evaluator strength at showdown (higher wins; `hand_rank // 13**5` is the category 0 high card … 8 straight flush). Null unless the seat reached showdown. |

The columns up to `showdown` are format 1's, so `pl.concat(..., how="diagonal")` with phase-1 data works.

### `hands`

| Column | Type | Meaning |
|---|---|---|
| `num_seated`, `button`, `sb_seat`, `bb_seat` | u8 | |
| `board`, `runout`, `board_dealt` | str, str, u8 | Board cards actually dealt, all five runout cards (dealt or not), and how many were dealt (0, 3, 4, 5). |
| `num_actions` | u16 | History entries, blinds included. |
| `pot` | i32 | Final pot (sum of all `invested`). |
| `showdown` | bool | |

### `actions`

| Column | Type | Meaning |
|---|---|---|
| `idx` | u16 | Order within the hand, from 0 (blinds are 0 and 1). |
| `seat`, `bot`, `by` | u8, str, str | Who acted. |
| `street` | u8 | 0 preflop, 1 flop, 2 turn, 3 river. |
| `type` | enum | What the engine applied: `fold`, `check_call`, `raise`, `small_blind`, `big_blind`. |
| `value` | i32 | The seat's contribution **this street after the action**: the raise-to total for a raise, the matched bet for a call, unchanged for a fold or check. |
| `type_requested`, `value_requested` | enum, i32 | What the bot asked for before clamping. They differ when, for example, a raise was converted to a call or an out-of-range size was clipped. Unknown type codes read as `invalid`. |
| `is_all_in` | bool | The action put the seat all in. |
| `pot_before`, `current_bet_before`, `to_call`, `contributed_before`, `stack_before` | i32 | Table state just before the action, rebuilt from the history: total pot (**including** any bet this seat now faces), the street's bet to match, chips this seat owed (0 means it could check), its street contribution so far and its stack behind. |

The betting replay that produces the `_before` columns is checked against the engine's ending stacks for every hand when the block is written, and a mismatch raises an error. Sizes on the usual scale: a raise is `(value - current_bet_before) / (pot_before + to_call)` of the pot after calling; a bet being faced is `to_call / (pot_before - to_call)` of the pot before it was made (a pot-sized bet is 1.0; note that `to_call / pot_before` would give 0.5).

### Reading and examples

```python
import polars as pl
R = "experiments/phase_2_suneater/results"
seat_hands = pl.scan_parquet(f"{R}/seat_hands/**/*.parquet", hive_partitioning=False)
actions = pl.scan_parquet(f"{R}/actions/**/*.parquet", hive_partitioning=False)

# Heads-up: the button's first preflop decision (fold, limp or raise), by bot
first = actions.filter(pl.col("street") == 0, pl.col("idx") == 2)          # idx 0 and 1 are the blinds
first.group_by("bot", "type").len().collect()

# Postflop: fold rate facing a bet, by bot and by the bet's size as a fraction of the pot before it
facing = actions.filter(pl.col("street") > 0, pl.col("to_call") > 0, pl.col("type").is_in(["fold", "check_call", "raise"]))
(facing.with_columns((pl.col("to_call") / (pl.col("pot_before") - pl.col("to_call"))).alias("bet_to_pot"))
       .group_by("bot", (pl.col("bet_to_pot") * 4).round() / 4)
       .agg((pl.col("type") == "fold").mean().alias("fold_rate"), pl.len()).sort("bot", "bet_to_pot").collect())

# Win rate by starting hand class and position
seat_hands.group_by("bot", "seats_from_button", "hand_class").agg((pl.col("net_chips").mean() * 50).alias("bb_per_100"), pl.len()).collect()
```

Statistics are as in format 1: bb/100 with a table-clustered SE (`std` of per-table bb/100 divided by √tables). Leak statistics sliced finely (one hand class at one position) have far fewer hands than the block. Check `len()` before reading anything into a rate.

### Size and speed

About 100 bytes per heads-up hand across the three tables, so a 16,000-hand block is around 1.6 MB. Building the frames takes well under a second per block; the simulation dominates.

### Replaying a stored block

`tournament.regenerate_block(bot_a, bot_b, seed, tables, hands, max_players)` replays a block exactly as the tournament played it, using the same key split and seat pattern. It returns `(env, records, num_recorded)`, and `records_to_frames` turns these into the stored tables. With format 2 you rarely need it, since the trajectories are on disk. It is still how to recover phase-1 hands, which stored outcomes only: `regenerate_block(..., max_players=6)` reproduces a phase-1 block's `seat_hands` rows exactly on the same engine (`engine_sha256` 295477eff7637056). Results can differ across platforms, so compare `net_chips` with the stored rows before trusting a replay.

### Play sessions (UI)

A session writes the same three tables under `ui/sessions/<session>/`, with `session` as the only key column. One way to do it is to stack the session's `HandRecord`s along a table axis of length 1 and call `records_to_frames(records, num_recorded, bot_names, {"session": name}, bot_kinds)`, with `bot_kinds[i] = "human"` for the human seat's bot type. Human hands then go through the same analysis code as tournament hands.
