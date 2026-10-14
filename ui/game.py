"""Drives one unbatched table for a person playing against registry bots.

The engine is used read-only through `next_event`, `step_with_record` (with an action override for
the human seat) and `get_observation`. Everything here is presentation: un-rotating observations to
absolute seats, building snapshots for the page, and handing finished hands to the session writer.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from bots import registry
from poker_bots.hand_eval import CATEGORY_BASE
from poker_bots.poker_game import Action, ActionData, HandRecord, HumanSeat, PlayerTableState, make_poker_env
from poker_bots.simulate import ACTION_NAMES, STREET_NAMES, card_to_str, hand_to_json
from ui.storage import SessionWriter, short_name

HUMAN_SEAT = 0
CATEGORY_NAMES = ("high card", "pair", "two pair", "three of a kind", "straight", "flush", "full house",
                  "four of a kind", "straight flush")
ACTION_TYPES = {"fold": Action.FOLD, "check_call": Action.CHECK_CALL, "raise": Action.RAISE}


def rank_name(rank: int | None) -> str | None:
    return None if rank is None or rank < 0 else CATEGORY_NAMES[rank // CATEGORY_BASE]


def cards(ids) -> list[str | None]:
    return [card_to_str(int(c)) if c >= 0 else None for c in ids]


def unrotate(obs: PlayerTableState, seat: int, max_players: int) -> dict:
    """Observer-relative indices (0 = the observer) back to absolute seats."""
    order = (np.arange(max_players) - seat) % max_players   # absolute seat s sits at relative index order[s]
    absolute = lambda seat_value: (int(seat_value) + seat) % max_players
    return {
        "stacks": np.asarray(obs.stacks)[order].tolist(),
        "contributed": np.asarray(obs.contributed)[order].tolist(),
        "seated": np.asarray(obs.seated)[order].tolist(),
        "in_hand": np.asarray(obs.in_hand)[order].tolist(),
        "button": absolute(obs.button),
        "sb_seat": absolute(obs.sb_seat),
        "bb_seat": absolute(obs.bb_seat),
    }


def action_log(history, action_count: int, max_players: int, observer: int) -> list[dict]:
    """The observation's history, with seats back in absolute terms."""
    log = []
    for i in range(action_count):
        if not history.is_valid[i]:
            continue
        log.append({
            "seat": (int(history.player_id[i]) + observer) % max_players,
            "street": STREET_NAMES[int(history.round[i])],
            "type": ACTION_NAMES[int(history.action_type[i])],
            "value_applied": int(history.amount[i]),
            "is_all_in": bool(history.is_all_in[i]),
        })
    return log


class PlaySession:
    """One human seat (seat 0) plus bots. `seat_bots` names a registry bot per other seat, in seat order."""

    def __init__(self, seat_bots: list[str], small_blind: int = 1, big_blind: int = 2, starting_bb: int = 100,
                 seed: int | None = None, flush_every: int = 10, sessions_dir=None, jit: bool = True):
        self.seed = int(seed if seed is not None else time.time_ns() % 2 ** 31)
        self.max_players = len(seat_bots) + 1
        names = list(dict.fromkeys(seat_bots))
        classes = {name: registry.get_bot(name) for name in names}
        self.env = make_poker_env((HumanSeat(), *(classes[name]() for name in names)),
                                  max_players=self.max_players, small_blind=small_blind, big_blind=big_blind)
        self.seat_names = ["human"] + seat_bots
        seat_bot = jnp.asarray([0] + [1 + names.index(name) for name in seat_bots], dtype=jnp.int32)
        stacks = jnp.full((self.max_players,), starting_bb * big_blind, dtype=jnp.int32)
        self.state = self.env.init(jax.random.PRNGKey(self.seed), jnp.ones((self.max_players,), dtype=jnp.bool_),
                                   stacks, 0, seat_bot)
        wrap = jax.jit if jit else (lambda f: f)
        self._step = wrap(self.env.step_with_record)
        self._next_event = wrap(self.env.next_event)
        self._observe = wrap(self.env.get_observation)
        self.last_hand: dict | None = None
        self.finished = False
        self.writer = SessionWriter(
            config={"max_players": self.max_players, "small_blind": small_blind, "big_blind": big_blind,
                    "starting_stack_bb": starting_bb, "rebuy": True, "human_seat": HUMAN_SEAT,
                    "seats": self.seat_names},
            seed=self.seed, flush_every=flush_every, root=sessions_dir,
            bot_types=[("human", None, HumanSeat, "human")] + [(short_name(name), name, classes[name], "bot") for name in names])
        self.human_net = 0

    @property
    def session_id(self) -> str:
        return self.writer.session_id

    def event(self) -> int:
        return int(self._next_event(self.state))

    def human_to_act(self) -> bool:
        return self.event() == 3 and int(self.state.current_player) == HUMAN_SEAT

    def advance(self) -> str:
        """Run engine steps up to the next thing worth drawing. Returns what happened:
        "human" (waiting for input), "bot" (a bot acted), "street", "hand_over" or "finished"."""
        if self.finished:
            return "finished"
        self.last_hand = None
        event = self.event()
        if event == 0:
            self.close()
            return "finished"
        if event == 3 and int(self.state.current_player) == HUMAN_SEAT:
            return "human"
        self.state, record, ended = self._step(self.state)
        if bool(ended):
            self._record(record)
            return "hand_over"
        return "bot" if event == 3 else "street"

    def act(self, kind: str, value: int = 0) -> None:
        if not self.human_to_act():
            raise ValueError("it is not your turn")
        action = ActionData(type=jnp.asarray(ACTION_TYPES[kind], dtype=jnp.int32), value=jnp.asarray(value, dtype=jnp.int32))
        self.state, _, _ = self._step(self.state, action)

    def _record(self, record: HandRecord) -> None:
        record = jax.device_get(record)
        self.writer.add_hand(record)
        self.human_net += int(record.ending_stacks[HUMAN_SEAT] - record.starting_stacks[HUMAN_SEAT])
        self.last_hand = hand_to_json(record, 0, self.env.player_metadata)

    def close(self) -> None:
        self.finished = True
        self.writer.close()

    def snapshot(self, show_bot_cards: bool = False) -> dict:
        state = self.state
        obs = jax.device_get(self._observe(state, jnp.asarray(HUMAN_SEAT)))
        table = unrotate(obs, HUMAN_SEAT, self.max_players)
        hole = np.asarray(state.deck.hole_cards)
        seats = []
        for seat in range(self.max_players):
            visible = seat == HUMAN_SEAT or show_bot_cards
            seats.append({
                "seat": seat,
                "name": self.seat_names[seat],
                "stack": table["stacks"][seat],
                "contributed": table["contributed"][seat],
                "seated": table["seated"][seat],
                "in_hand": table["in_hand"][seat],
                "cards": cards(hole[seat]) if visible and table["seated"][seat] else None,
            })
        event = self.event()
        to_act = event == 3 and int(state.current_player) == HUMAN_SEAT
        controls = None
        if to_act:
            to_call = int(obs.current_bet) - int(obs.contributed[0])
            controls = {
                "to_call": min(to_call, int(obs.stacks[0])),
                "can_raise": bool(obs.can_raise),
                "raise_min": int(obs.raise_lower_bound),
                "raise_max": int(obs.raise_upper_bound),
                "current_bet": int(obs.current_bet),
            }
        return {
            "session": self.session_id,
            "hand": int(state.hand_count),
            "street": STREET_NAMES[int(state.current_round)],
            "board": [c for c in cards(obs.board) if c],
            "pot": int(obs.pot),
            "big_blind": self.env.big_blind,
            "button": table["button"],
            "sb_seat": table["sb_seat"],
            "bb_seat": table["bb_seat"],
            "current_player": int(state.current_player) if event == 3 else None,
            "human_seat": HUMAN_SEAT,
            "seats": seats,
            "actions": action_log(obs.history, int(state.action_count), self.max_players, HUMAN_SEAT),
            "controls": controls,
            "finished": self.finished,
            "last_hand": self._hand_summary(self.last_hand, show_bot_cards) if self.last_hand else None,
            "hands_played": self.writer.hands,
            "net": self.human_net,
        }

    def _hand_summary(self, hand: dict, show_bot_cards: bool) -> dict:
        """End-of-hand view: cards are shown for the human, for seats that reached showdown, or all with the toggle."""
        shown = [rank is not None for rank in hand["hand_ranks"]]
        folded = {action["seat"] for action in hand["actions"] if action["type"] == "fold"}
        return {
            "hand": hand["hand_index"],
            "board": hand["board"][:hand["board_dealt"]],
            "showdown": hand["went_to_showdown"],
            "pot": sum(hand["payouts"]),
            "button": hand["button"],
            "seats": [
                {
                    "seat": seat,
                    "name": self.seat_names[seat],
                    "cards": hand["hole_cards"][seat] if (seat == HUMAN_SEAT or shown[seat] or show_bot_cards) else None,
                    "rank": rank_name(hand["hand_ranks"][seat]),
                    "payout": hand["payouts"][seat],
                    "net": hand["ending_stacks"][seat] - hand["starting_stacks"][seat],
                    "stack": hand["ending_stacks"][seat],
                    "in_hand": seat not in folded,
                    "rebuy": hand["rebuys"][seat],
                    "cash_out": hand["cash_outs"][seat],
                }
                for seat in range(self.max_players) if hand["seated"][seat]
            ],
        }
