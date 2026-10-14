import json

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_equal

from poker_bots.poker_game import Action, ActionData, HandRecord, HumanSeat, RandomBot, make_poker_env
from poker_bots.simulate import hand_to_json, make_header


class ObservingBot(RandomBot):
    def init_state(self, key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)

    def act(self, obs, state, key):
        return super().act(obs, None, key)[0], state

    def observe(self, record: HandRecord, state, key: jax.Array):
        hands, net, leaked = state
        shown = record.went_to_showdown & record.in_hand
        hidden_hole = (record.hole_cards[1:] >= 0) & ~shown[1:, None]
        hidden_board = (jnp.arange(5) >= record.board_dealt) & (record.board >= 0)
        leaked = leaked | jnp.any(hidden_hole) | jnp.any(hidden_board) | jnp.any(record.hole_cards[0] < 0)
        net = net + record.ending_stacks[0] - record.starting_stacks[0]
        return hands + 1, net, leaked


def test_observe_sees_every_hand_from_own_seat():
    num_hands = 30
    env = make_poker_env((ObservingBot(), RandomBot()), max_players=6)
    seat_bot = jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    state = env.init(jax.random.PRNGKey(0), jnp.ones((6,), dtype=jnp.bool_), None, 0, seat_bot)
    state, records, num_recorded = jax.jit(lambda state: env.run_hands(state, num_hands))(state)
    hands, net, leaked = state.player_states[0]
    observer = np.asarray(seat_bot == 0)

    assert int(num_recorded) == num_hands
    assert_array_equal(hands[observer], num_hands)
    assert_array_equal(hands[~observer], 0)
    assert not bool(jnp.any(leaked))
    assert_array_equal(net[observer], jnp.sum(records.ending_stacks - records.starting_stacks, axis=0)[observer])


def test_human_seat_driven_by_actions():
    env = make_poker_env((HumanSeat(), RandomBot()), max_players=3)
    seat_bot = jnp.array([0, 1, 1], dtype=jnp.int32)
    state = env.init(jax.random.PRNGKey(3), jnp.ones((3,), dtype=jnp.bool_), None, 0, seat_bot)
    step = jax.jit(env.step_with_record)
    next_event = jax.jit(env.next_event)
    raise_to = ActionData(type=jnp.asarray(Action.RAISE, dtype=jnp.int32), value=jnp.asarray(10 ** 6, dtype=jnp.int32))

    hands = []
    while len(hands) < 10:
        if int(next_event(state)) == 3 and int(state.current_player) == 0:
            obs = env.get_observation(state, state.current_player)
            assert int(obs.hand_count) == int(state.hand_count)
            state, record, ended = step(state, raise_to)
        else:
            state, record, ended = step(state)
        if bool(ended):
            hands.append(jax.device_get(record))

    for hand in hands:
        valid = hand.history.is_valid & (hand.history.player_id == 0) & (hand.history.requested_type == Action.RAISE)
        assert bool(np.any(valid))
        assert int(np.sum(hand.ending_stacks)) == int(np.sum(hand.starting_stacks))

    header = make_header(env, state, seed=3)
    assert [seat["kind"] for seat in header["tables"][0]["seats"]] == ["human", "bot", "bot"]
    actions = hand_to_json(hands[0], 0, env.player_metadata)["actions"]
    assert all(action["by"] == ("human" if action["seat"] == 0 else "bot") for action in actions)
    json.dumps(header)


def test_observation_for_any_seat():
    env = make_poker_env((RandomBot(),), max_players=4)
    state = env.init(jax.random.PRNGKey(1), jnp.ones((4,), dtype=jnp.bool_))
    for seat in range(4):
        obs = env.get_observation(state, jnp.asarray(seat))
        assert_array_equal(obs.hole_cards, state.deck.hole_cards[seat])
        assert int(obs.stacks[0]) == int(state.stacks[seat])
        assert int(obs.button) == (int(state.button) - seat) % 4
