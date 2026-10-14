import json
import subprocess
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import numpy as np

from poker_bots.poker_game import Action, HandRecord, ID_TO_CARD, PokerEnv, TableState

SCHEMA_VERSION = 1
ACTION_NAMES = {
    Action.FOLD: "fold",
    Action.CHECK_CALL: "check_call",
    Action.RAISE: "raise",
    Action.NO_OP: "no_op",
    Action.SMALL_BLIND: "small_blind",
    Action.BIG_BLIND: "big_blind"
}
STREET_NAMES = ("preflop", "flop", "turn", "river")
SUIT_LETTERS = {"Diamonds": "d", "Clubs": "c", "Hearts": "h", "Spades": "s"}

def card_to_str(card: int) -> str:
    rank, suit = ID_TO_CARD[card]
    return ("T" if rank == "10" else rank) + SUIT_LETTERS[suit]

def get_engine_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=Path(__file__).parent
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def get_kind(metadata: dict) -> str:
    return metadata.get("kind", "bot")

def make_header(env: PokerEnv, table_states: TableState, seed: int | None) -> dict:
    seated = np.atleast_2d(np.asarray(table_states.seated))
    seat_bot = np.atleast_2d(np.asarray(table_states.seat_bot))
    return {
        "schema_version": SCHEMA_VERSION,
        "engine_commit": get_engine_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_players": env.max_players,
        "small_blind": env.small_blind,
        "big_blind": env.big_blind,
        "seed": seed,
        "bots": [{"index": i, "metadata": metadata} for i, metadata in enumerate(env.player_metadata)],
        "tables": [
            {
                "table": table,
                "seats": [
                    {"seat": seat, "kind": get_kind(env.player_metadata[seat_bot[table, seat]]), "bot_index": int(seat_bot[table, seat])} if seated[table, seat]
                    else {"seat": seat, "kind": "empty"}
                    for seat in range(env.max_players)
                ]
            }
            for table in range(seated.shape[0])
        ]
    }

def hand_to_json(hand: HandRecord, table: int, player_metadata: tuple) -> dict:
    seated = hand.seated.tolist()
    seat_kinds = [get_kind(player_metadata[bot]) for bot in hand.seat_bot.tolist()]
    shown = (hand.went_to_showdown & hand.in_hand).tolist()
    history = hand.history
    return {
        "table": table,
        "hand_index": int(hand.hand_index),
        "button": int(hand.button),
        "sb_seat": int(hand.sb_seat),
        "bb_seat": int(hand.bb_seat),
        "seated": seated,
        "seat_bot": hand.seat_bot.tolist(),
        "starting_stacks": hand.starting_stacks.tolist(),
        "hole_cards": [
            [card_to_str(card) for card in cards] if is_seated else None
            for cards, is_seated in zip(hand.hole_cards.tolist(), seated)
        ],
        "board": [card_to_str(card) for card in hand.board.tolist()],
        "board_dealt": int(hand.board_dealt),
        "actions": [
            {
                "seat": int(history.player_id[i]),
                "street": STREET_NAMES[int(history.round[i])],
                "type": ACTION_NAMES[int(history.action_type[i])],
                "type_requested": ACTION_NAMES[int(history.requested_type[i])],
                "value_requested": int(history.requested[i]),
                "value_applied": int(history.amount[i]),
                "is_all_in": bool(history.is_all_in[i]),
                "by": seat_kinds[int(history.player_id[i])]
            }
            for i in range(int(hand.action_count)) if history.is_valid[i]
        ],
        "went_to_showdown": bool(hand.went_to_showdown),
        "hand_ranks": [int(rank) if is_shown else None for rank, is_shown in zip(hand.hand_ranks.tolist(), shown)],
        "payouts": hand.payouts.tolist(),
        "ending_stacks": hand.ending_stacks.tolist(),
        "rebuys": hand.rebuys.tolist(),
        "cash_outs": hand.cash_outs.tolist()
    }

def summarise(net_by_bot: list[list[np.ndarray]], env: PokerEnv) -> list[dict]:
    summary = []
    for i, metadata in enumerate(env.player_metadata):
        net = np.concatenate(net_by_bot[i]) if net_by_bot[i] else np.zeros(0)
        num = len(net)
        summary.append({
            "bot_index": i,
            "metadata": metadata,
            "seat_hands": num,
            "net_chips": int(net.sum()),
            "bb_per_100": float(100 * net.mean() / env.big_blind) if num else None,
            "bb_per_100_se": float(100 * net.std(ddof=1) / np.sqrt(num) / env.big_blind) if num > 1 else None
        })
    return summary

def simulate(
    env: PokerEnv,
    table_states: TableState,
    total_hands: int,
    chunk_hands: int,
    out_path: str | Path,
    seed: int | None = None
) -> tuple[TableState, list[dict]]:
    out_path = Path(out_path)
    runners: dict[int, Callable] = {}
    net_by_bot: list[list[np.ndarray]] = [[] for _ in env.player_metadata]

    with open(out_path, "w") as log:
        log.write(json.dumps(make_header(env, table_states, seed)) + "\n")
        hands_done = 0
        while hands_done < total_hands:
            num_hands = min(chunk_hands, total_hands - hands_done)
            if num_hands not in runners:
                runners[num_hands] = jax.jit(jax.vmap(partial(env.run_hands, num_hands=num_hands)))
            table_states, records, num_recorded = runners[num_hands](table_states)
            records, num_recorded = jax.device_get((records, num_recorded))

            for table in range(len(num_recorded)):
                for hand in range(int(num_recorded[table])):
                    log.write(json.dumps(hand_to_json(jax.tree.map(lambda x: x[table, hand], records), table, env.player_metadata)) + "\n")

            recorded = np.arange(num_hands)[None, :] < num_recorded[:, None]
            seat_mask = records.seated & recorded[:, :, None]
            net = records.ending_stacks - records.starting_stacks
            for i in range(len(env.player_metadata)):
                net_by_bot[i].append(net[seat_mask & (records.seat_bot == i)])

            hands_done += num_hands
            if int(num_recorded.max()) < num_hands:
                break

    summary = summarise(net_by_bot, env)
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return table_states, summary
