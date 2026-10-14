import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from poker_bots.poker_game import Action, ActionData, HumanSeat, make_poker_env
from poker_bots.pure_game import RandomBot, Table, evaluate_hand
from poker_bots.hand_eval import evaluate_hands


class DeckFeed:
    """Stands in for Table's rng so each hand is dealt the JAX engine's deck."""
    deck: list[int]

    def shuffle(self, cards: list[int]):
        cards[:] = self.deck


def jax_deck(state) -> list[int]:
    hole = np.asarray(state.deck.hole_cards).tolist()
    dealt = [cards[0] for cards in hole] + [cards[1] for cards in hole] + np.asarray(state.deck.runout).tolist()
    return dealt + [card for card in range(52) if card not in dealt]


def valid_history(history) -> list[tuple]:
    if isinstance(history, list):
        return [(r.player_id, r.action_type, r.amount, r.requested_type, r.requested, r.is_all_in, r.round) for r in history]
    h = jax.tree.map(np.asarray, history)
    return [
        (int(h.player_id[i]), int(h.action_type[i]), int(h.amount[i]), int(h.requested_type[i]), int(h.requested[i]), bool(h.is_all_in[i]), int(h.round[i]))
        for i in range(len(h.is_valid)) if h.is_valid[i]
    ]


def as_plain(value):
    value = np.asarray(value)
    return value.tolist()


def assert_same(jax_tuple, pure_obj, fields):
    for name in fields:
        if name == "history":
            assert valid_history(getattr(jax_tuple, name)) == valid_history(getattr(pure_obj, name)), name
        else:
            assert as_plain(getattr(jax_tuple, name)) == as_plain(getattr(pure_obj, name)), name


OBS_FIELDS = (
    "hole_cards", "board", "history", "stacks", "seated", "in_hand", "button", "sb_seat", "bb_seat", "contributed",
    "current_round", "min_raise", "current_bet", "pot", "can_raise", "raise_lower_bound", "raise_upper_bound",
    "big_blind", "hand_count"
)
RECORD_FIELDS = (
    "hand_index", "button", "sb_seat", "bb_seat", "seated", "starting_stacks", "hole_cards", "board", "board_dealt",
    "history", "in_hand", "went_to_showdown", "hand_ranks", "payouts", "ending_stacks", "rebuys", "cash_outs"
)


def random_action(rng: random.Random, obs) -> ActionData:
    kind = rng.choices([Action.FOLD, Action.CHECK_CALL, Action.RAISE, Action.NO_OP, 7], weights=[15, 50, 30, 3, 2])[0]
    value = rng.choice([
        rng.randint(-5, obs.raise_upper_bound + 10),
        obs.raise_lower_bound,
        obs.raise_upper_bound,
        rng.randint(obs.raise_lower_bound, obs.raise_lower_bound + 3 * obs.big_blind),
    ])
    return ActionData(type=jnp.asarray(kind, dtype=jnp.int32), value=jnp.asarray(value, dtype=jnp.int32))


CONFIGS = {
    "six_max_rebuy": dict(max_players=6, seated=[1] * 6, stacks=None, rebuy=True, min_stack=0, max_stack=None, hands=60),
    "ring_busting": dict(max_players=9, seated=[1, 0, 1, 1, 0, 1, 0, 0, 1], stacks=[400, 0, 250, 600, 0, 90, 0, 0, 300], rebuy=False, min_stack=0, max_stack=None, hands=200),
    "short_stacks": dict(max_players=7, seated=[1, 1, 1, 1, 0, 1, 1], stacks=[12, 30, 7, 400, 0, 25, 3], rebuy=False, min_stack=0, max_stack=None, hands=200),
    "caps": dict(max_players=5, seated=[1] * 5, stacks=[30, 200, 90, 15, 300], rebuy=True, min_stack=20, max_stack=150, hands=60),
    "heads_up": dict(max_players=2, seated=[1, 1], stacks=[150, 50], rebuy=False, min_stack=0, max_stack=None, hands=200),
}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", CONFIGS)
def test_lockstep_with_jax_engine(name, seed):
    config = CONFIGS[name]
    n = config["max_players"]
    env = make_poker_env((HumanSeat(),), max_players=n, rebuy=config["rebuy"])
    stacks = config["stacks"] or [200] * n
    max_stack = config["max_stack"] or 600
    state = env.init(
        jax.random.PRNGKey(seed), jnp.asarray(config["seated"], dtype=jnp.bool_), jnp.asarray(stacks, dtype=jnp.int32),
        n - 1, None, None, config["min_stack"], max_stack
    )
    feed = DeckFeed()
    feed.deck = jax_deck(state)
    bots = [RandomBot() if s else None for s in config["seated"]]
    table = Table(bots, starting_stacks=stacks, button=n - 1, min_stack=config["min_stack"], max_stack=max_stack, rebuy=config["rebuy"], rng=feed)

    step = jax.jit(env.step_with_record)
    next_event = jax.jit(env.next_event)
    get_observation = jax.jit(env.get_observation)
    rng = random.Random(seed)
    hands = 0
    while hands < config["hands"]:
        event = int(next_event(state))
        assert event == table.next_event()
        if event == 0:
            break
        action = None
        if event == 3:
            seat = int(state.current_player)
            assert seat == table.current_player
            observer = rng.randrange(n)
            assert_same(get_observation(state, observer), table.observation(observer), OBS_FIELDS)
            jax_obs = get_observation(state, seat)
            assert_same(jax_obs, table.observation(seat), OBS_FIELDS)
            action = random_action(rng, table.observation(seat))
        state, jax_record, ended = step(state, action)
        if bool(ended):
            feed.deck = jax_deck(state)
        pure_record = table.step(None if action is None else ActionData(int(action.type), int(action.value)))
        assert bool(ended) == (pure_record is not None)
        if pure_record is not None:
            assert_same(jax_record, pure_record, RECORD_FIELDS)
            hands += 1
        assert as_plain(state.stacks) == table.stacks
        assert as_plain(state.seated) == table.seated
    # Ran the full count, or played until one player had every chip
    assert hands == config["hands"] or (event == 0 and hands > 0)


def test_evaluate_hand_matches_jax():
    rng = random.Random(0)
    for num_cards in (5, 6, 7):
        hands = [rng.sample(range(52), num_cards) for _ in range(300)]
        expected = np.asarray(evaluate_hands(jnp.asarray(hands))).tolist()
        assert [evaluate_hand(hand) for hand in hands] == expected


class CountingBot(RandomBot):
    def __init__(self, seed: int):
        super().__init__(seed)
        self.hands = 0
        self.net = 0
        self.leaked = False

    def observe(self, record):
        self.hands += 1
        self.net += record.ending_stacks[0] - record.starting_stacks[0]
        shown = [record.went_to_showdown and in_hand for in_hand in record.in_hand]
        self.leaked |= any(cards != [-1, -1] and not shown[s] for s, cards in enumerate(record.hole_cards) if s > 0)
        self.leaked |= any(card >= 0 for card in record.board[record.board_dealt:])


def test_observe_and_conservation():
    bots = [CountingBot(seed) for seed in range(6)]
    table = Table(bots, seed=1)
    records = table.run(300)
    assert len(records) == 300
    for record in records:
        assert sum(record.ending_stacks) == sum(record.starting_stacks)
    for seat, bot in enumerate(bots):
        assert bot.hands == 300 and not bot.leaked
        assert bot.net == sum(r.ending_stacks[seat] - r.starting_stacks[seat] for r in records)
