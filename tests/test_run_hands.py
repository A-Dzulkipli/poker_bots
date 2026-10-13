import json

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_equal

from poker_bots.poker_game import Action, ActionData, RandomBot, make_poker_env
from poker_bots.simulate import simulate


class OverShoveBot:
    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs, state: None, key: jax.Array) -> tuple[ActionData, None]:
        action = ActionData(
            type=jnp.asarray(Action.RAISE, dtype=jnp.int32),
            value=jnp.asarray(10 ** 6, dtype=jnp.int32)
        )
        return action, state

    def metadata(self) -> dict:
        return {"bot_type": "over_shove_bot"}


def make_tables(env, num_tables: int, seed: int = 0):
    keys = jax.random.split(jax.random.PRNGKey(seed), num_tables)
    seated = jnp.ones((num_tables, env.max_players), dtype=jnp.bool_)
    return jax.vmap(env.init, in_axes=(0, 0))(keys, seated)


def test_default_stacks_and_caps():
    env = make_poker_env((RandomBot(),), max_players=6, big_blind=2)
    state = env.init(jax.random.PRNGKey(0), jnp.ones((6,), dtype=jnp.bool_))
    assert_array_equal(state.buy_in, [200] * 6)
    assert int(state.min_stack) == 0
    assert int(state.max_stack) == 600


def test_run_hands_cash_accounting():
    num_hands = 50
    env = make_poker_env((RandomBot(),), max_players=6)
    states = make_tables(env, num_tables=16)
    run = jax.jit(jax.vmap(lambda state: env.run_hands(state, num_hands)))
    states, records, num_recorded = run(states)

    # Rebuys keep every table going, so each records exactly num_hands
    assert_array_equal(num_recorded, num_hands)
    assert_array_equal(records.hand_index, jnp.broadcast_to(jnp.arange(num_hands), records.hand_index.shape))
    assert_array_equal(states.hand_count, num_hands)

    # Chips are conserved within each hand
    assert_array_equal(jnp.sum(records.ending_stacks, axis=-1), jnp.sum(records.starting_stacks, axis=-1))
    # Next hand starts from this hand's end, after rebuys and cash-outs
    adjusted = records.ending_stacks + records.rebuys - records.cash_outs
    assert_array_equal(records.starting_stacks[:, 1:], adjusted[:, :-1])
    assert int(jnp.max(records.starting_stacks[:, 1:])) <= int(states.max_stack[0])
    # Rebuys only for busted players, back to the buy-in
    assert bool(jnp.all((records.rebuys == 0) | (records.ending_stacks == 0)))
    assert bool(jnp.all((records.rebuys == 0) | (records.ending_stacks + records.rebuys == states.buy_in[:, None, :])))


def test_requested_value_recorded_separately():
    env = make_poker_env((OverShoveBot(),), max_players=3)
    states = make_tables(env, num_tables=4)
    _, records, _ = jax.jit(jax.vmap(lambda state: env.run_hands(state, 5)))(states)

    raises = records.history.is_valid & (records.history.action_type == Action.RAISE)
    assert bool(jnp.any(raises))
    assert bool(jnp.all(jnp.where(raises, records.history.requested == 10 ** 6, True)))
    assert bool(jnp.all(jnp.where(raises, records.history.amount < 10 ** 6, True)))


def test_simulate_writes_logs_and_summary(tmp_path):
    env = make_poker_env((RandomBot(), RandomBot()), max_players=6)
    num_tables = 8
    keys = jax.random.split(jax.random.PRNGKey(0), num_tables)
    seated = jnp.ones((num_tables, 6), dtype=jnp.bool_)
    seat_bot = jnp.tile(jnp.array([0, 1, 0, 1, 0, 1]), (num_tables, 1))
    states = jax.vmap(env.init, in_axes=(0, 0, None, None, 0))(keys, seated, None, 0, seat_bot)

    out_path = tmp_path / "run.jsonl"
    _, summary = simulate(env, states, total_hands=25, chunk_hands=10, out_path=out_path, seed=0)

    lines = out_path.read_text().splitlines()
    header = json.loads(lines[0])
    hands = [json.loads(line) for line in lines[1:]]
    assert header["max_players"] == 6 and len(header["tables"]) == num_tables
    assert len(hands) == num_tables * 25
    assert sorted(hand["hand_index"] for hand in hands if hand["table"] == 0) == list(range(25))
    assert all(sum(hand["ending_stacks"]) == sum(hand["starting_stacks"]) for hand in hands)

    # Zero-sum across bot types
    assert sum(entry["net_chips"] for entry in summary) == 0
    assert sum(entry["seat_hands"] for entry in summary) == num_tables * 25 * 6
    assert json.loads(out_path.with_suffix(".summary.json").read_text()) == summary
