"""Full hand histories as parquet tables (storage format 2, used from phase 2 onwards).

    frames = records_to_frames(records, num_recorded, bot_names, keys={"pair": "a__b", "seed": 0})
    for name, frame in frames.items():        # "seat_hands", "hands", "actions"
        frame.write_parquet(..., compression="zstd")

`records` is the `HandRecord` from `env.run_hands`, vmapped over tables and moved to host
(`jax.device_get`), so every field has leading axes (table, hand). Play sessions with a single
table pass records with a table axis of length 1. Seats are absolute table seats throughout.
See experiments/README.md for the column reference.
"""
import numpy as np
import polars as pl

from poker_bots.simulate import ACTION_NAMES, card_to_str

FORMAT_VERSION = 2
ACTION_TYPES = pl.Enum(list(ACTION_NAMES.values()) + ["invalid"])
RANK_CHARS = "23456789TJQKA"
CARD_STRINGS = np.array([card_to_str(card) for card in range(52)])


def cards_to_str(cards: np.ndarray) -> np.ndarray:
    """Concatenate card strings along the last axis, e.g. [[51, 38]] -> ["AsKh"]. -1 entries are skipped."""
    parts = np.where(cards >= 0, CARD_STRINGS[np.clip(cards, 0, 51)], "")
    return np.array(["".join(row) for row in parts.reshape(-1, cards.shape[-1])]).reshape(cards.shape[:-1])


def hand_class(hole_cards: np.ndarray) -> np.ndarray:
    """Starting-hand class from (..., 2) card ids: "AKs", "T9o", "77"."""
    ranks, suits = hole_cards % 13, hole_cards // 13
    high, low = np.maximum(ranks[..., 0], ranks[..., 1]), np.minimum(ranks[..., 0], ranks[..., 1])
    chars = np.array(list(RANK_CHARS))
    suffix = np.where(high == low, "", np.where(suits[..., 0] == suits[..., 1], "s", "o"))
    return np.char.add(np.char.add(chars[high], chars[low]), suffix)


def action_names(codes: np.ndarray) -> np.ndarray:
    names = np.array([ACTION_NAMES.get(code, "invalid") for code in range(max(ACTION_NAMES) + 1)] + ["invalid"])
    return names[np.where((codes >= 0) & (codes <= max(ACTION_NAMES)), codes, len(names) - 1)]


def replay_betting(records) -> dict[str, np.ndarray]:
    """Table state just before each action, rebuilt from the history: pot, street bet to match,
    chips owed, this seat's street contribution and stack. Shapes (table, hand, action)."""
    history = records.history
    tables, hands, num_actions = history.player_id.shape
    stacks = records.starting_stacks.astype(np.int64).copy()
    contributed = np.zeros_like(stacks)
    pot = np.zeros((tables, hands), dtype=np.int64)
    street = np.zeros((tables, hands), dtype=np.int64)
    out = {name: np.zeros((tables, hands, num_actions), dtype=np.int64)
           for name in ("pot_before", "current_bet_before", "to_call", "contributed_before", "stack_before")}
    for k in range(num_actions):
        valid = history.is_valid[:, :, k]
        seat = np.clip(history.player_id[:, :, k], 0, None)[..., None]
        new_street = valid & (history.round[:, :, k] != street)
        contributed = np.where(new_street[..., None], 0, contributed)
        street = np.where(valid, history.round[:, :, k], street)
        own_contribution = np.take_along_axis(contributed, seat, -1)[..., 0]
        own_stack = np.take_along_axis(stacks, seat, -1)[..., 0]
        current_bet = contributed.max(-1)
        out["pot_before"][:, :, k] = pot
        out["current_bet_before"][:, :, k] = current_bet
        out["to_call"][:, :, k] = np.minimum(current_bet - own_contribution, own_stack)
        out["contributed_before"][:, :, k] = own_contribution
        out["stack_before"][:, :, k] = own_stack
        delta = np.where(valid, history.amount[:, :, k] - own_contribution, 0)
        np.put_along_axis(contributed, seat, np.where(valid, history.amount[:, :, k], own_contribution)[..., None], -1)
        np.put_along_axis(stacks, seat, (own_stack - delta)[..., None], -1)
        pot += delta
    out["stacks_after"] = stacks
    out["pot_after"] = pot
    return out


def records_to_frames(records, num_recorded: np.ndarray, bot_names: list[str], keys: dict,
                      bot_kinds: list[str] | None = None) -> dict[str, pl.DataFrame]:
    """Build the seat_hands, hands and actions tables for one block of hands.

    bot_names[i] names bot type i of the env (the values of `seat_bot`); bot_kinds[i] is its
    metadata "kind" ("bot" or "human", default "bot"). `keys` are constant identifying columns
    put first in every table: {"pair": ..., "seed": ...} for tournaments, {"session": ...} for play.
    Raises if the replayed betting does not reproduce the engine's ending stacks.
    """
    tables, hands, max_players = records.seated.shape
    names = np.array(bot_names)
    kinds = np.array(bot_kinds or ["bot"] * len(bot_names))
    recorded = np.arange(hands)[None, :] < num_recorded[:, None]
    betting = replay_betting(records)
    if not np.array_equal((betting["stacks_after"] + records.payouts)[recorded], records.ending_stacks[recorded]):
        raise ValueError("replayed betting does not match the engine's ending stacks")

    def key_columns(length: int) -> dict:
        return {name: pl.Series(name, [value] * length) for name, value in keys.items()}

    # seat_hands: one row per seated seat per hand
    table, hand, seat = np.meshgrid(np.arange(tables), np.arange(hands), np.arange(max_players), indexing="ij")
    keep = records.seated & recorded[..., None]
    showdown = np.broadcast_to(records.went_to_showdown[..., None], keep.shape) & records.in_hand
    button = np.broadcast_to(records.button[..., None], keep.shape)
    seat_hands = pl.DataFrame({
        **key_columns(int(keep.sum())),
        "table": table[keep].astype(np.uint16),
        "hand": hand[keep].astype(np.uint32),
        "seat": seat[keep].astype(np.uint8),
        "bot": names[records.seat_bot[keep]],
        "by": kinds[records.seat_bot[keep]],
        "button": button[keep].astype(np.uint8),
        "seats_from_button": ((seat - button) % max_players)[keep].astype(np.uint8),
        "starting_stack": records.starting_stacks[keep].astype(np.int32),
        "net_chips": (records.ending_stacks - records.starting_stacks)[keep].astype(np.int32),
        "rebuy": records.rebuys[keep].astype(np.int32),
        "cash_out": records.cash_outs[keep].astype(np.int32),
        "showdown": showdown[keep],
        "hole_cards": cards_to_str(records.hole_cards)[keep],
        "hand_class": hand_class(records.hole_cards)[keep],
        "folded": ~records.in_hand[keep],
        "invested": (records.starting_stacks - betting["stacks_after"])[keep].astype(np.int32),
        "payout": records.payouts[keep].astype(np.int32),
        "hand_rank": pl.Series(np.where(showdown, records.hand_ranks, -1)[keep]).replace(-1, None).cast(pl.Int32),
    })

    # hands: one row per hand
    table, hand = np.meshgrid(np.arange(tables), np.arange(hands), indexing="ij")
    board_dealt = records.board_dealt
    dealt_board = np.where(np.arange(5) < board_dealt[..., None], records.board, -1)
    hands_frame = pl.DataFrame({
        **key_columns(int(recorded.sum())),
        "table": table[recorded].astype(np.uint16),
        "hand": hand[recorded].astype(np.uint32),
        "num_seated": records.seated.sum(-1)[recorded].astype(np.uint8),
        "button": records.button[recorded].astype(np.uint8),
        "sb_seat": records.sb_seat[recorded].astype(np.uint8),
        "bb_seat": records.bb_seat[recorded].astype(np.uint8),
        "board": cards_to_str(dealt_board)[recorded],
        "runout": cards_to_str(records.board)[recorded],
        "board_dealt": board_dealt[recorded].astype(np.uint8),
        "num_actions": records.action_count[recorded].astype(np.uint16),
        "pot": betting["pot_after"][recorded].astype(np.int32),
        "showdown": records.went_to_showdown[recorded],
    })

    # actions: one row per recorded action, blinds included
    history = records.history
    num_actions = history.player_id.shape[-1]
    table, hand, idx = np.meshgrid(np.arange(tables), np.arange(hands), np.arange(num_actions), indexing="ij")
    keep = history.is_valid & recorded[..., None]
    seat = np.clip(history.player_id, 0, None)
    acting_bot = np.take_along_axis(records.seat_bot, seat, -1)
    actions = pl.DataFrame({
        **key_columns(int(keep.sum())),
        "table": table[keep].astype(np.uint16),
        "hand": hand[keep].astype(np.uint32),
        "idx": idx[keep].astype(np.uint16),
        "seat": seat[keep].astype(np.uint8),
        "bot": names[acting_bot[keep]],
        "by": kinds[acting_bot[keep]],
        "street": history.round[keep].astype(np.uint8),
        "type": pl.Series(action_names(history.action_type[keep]), dtype=ACTION_TYPES),
        "value": history.amount[keep].astype(np.int32),
        "type_requested": pl.Series(action_names(history.requested_type[keep]), dtype=ACTION_TYPES),
        "value_requested": history.requested[keep].astype(np.int32),
        "is_all_in": history.is_all_in[keep],
        **{name: betting[name][keep].astype(np.int32) for name in
           ("pot_before", "current_bet_before", "to_call", "contributed_before", "stack_before")},
    })
    return {"seat_hands": seat_hands, "hands": hands_frame, "actions": actions}
