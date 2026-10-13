import jax
import jax.numpy as jnp
from numpy.testing import assert_array_equal

from poker_bots.poker_game import Round, TestBot, make_poker_env


def test_init_chip_conservation_multiway():
    num_players = 3
    bots = tuple(TestBot() for _ in range(num_players))
    env = make_poker_env(bots, max_players=num_players, small_blind=1, big_blind=2)

    key = jax.random.PRNGKey(0)
    seated = jnp.array([True, True, True])
    starting_stacks = jnp.array([100, 100, 100], dtype=jnp.int32)
    button = jnp.int32(0)

    state = env.init(key, seated, starting_stacks, button)

    # 1. Total chips on table + stacks must equal total initial chips
    assert int(jnp.sum(state.stacks) + jnp.sum(state.contributed)) == 300

    # 2. SB is seat 1 (1 chip), BB is seat 2 (2 chips)
    assert int(state.contributed[1]) == 1
    assert int(state.contributed[2]) == 2
    assert int(state.stacks[1]) == 99
    assert int(state.stacks[2]) == 98

    # 3. UTG acts first pre-flop multi-way -> Seat 0 (button)
    # Clockwise from button: Seat 1 (SB), Seat 2 (BB), Seat 0 (UTG)
    assert int(state.current_player) == 0
    assert int(state.current_bet) == 2


def test_init_heads_up_positions():
    num_players = 2
    bots = tuple(TestBot() for _ in range(num_players))
    env = make_poker_env(bots, max_players=num_players, small_blind=1, big_blind=2)

    key = jax.random.PRNGKey(42)
    seated = jnp.array([True, True])
    starting_stacks = jnp.array([50, 50], dtype=jnp.int32)
    button = jnp.int32(0)

    state = env.init(key, seated, starting_stacks, button)

    # Heads-up: Button (Seat 0) posts SB, other player (Seat 1) posts BB
    assert int(state.contributed[0]) == 1
    assert int(state.contributed[1]) == 2
    assert int(state.stacks[0]) == 49
    assert int(state.stacks[1]) == 48

    # Heads-up rule: Button/SB acts first pre-flop
    assert int(state.current_player) == 0
    assert int(state.current_bet) == 2


def test_init_short_stack_big_blind():
    num_players = 3
    bots = tuple(TestBot() for _ in range(num_players))
    env = make_poker_env(bots, max_players=num_players, small_blind=1, big_blind=2)

    key = jax.random.PRNGKey(7)
    seated = jnp.array([True, True, True])
    # BB (Seat 2) only has 1 chip
    starting_stacks = jnp.array([100, 100, 1], dtype=jnp.int32)
    button = jnp.int32(0)

    state = env.init(key, seated, starting_stacks, button)

    # BB should only contribute 1 chip and have 0 left
    assert int(state.contributed[2]) == 1
    assert int(state.stacks[2]) == 0

    # Others must still call the full big blind
    assert int(state.current_bet) == 2


def test_init_jitted_execution():
    num_players = 2
    bots = tuple(TestBot() for _ in range(num_players))
    env = make_poker_env(bots, max_players=num_players, small_blind=1, big_blind=2)

    key = jax.random.PRNGKey(123)
    seated = jnp.array([True, True])
    starting_stacks = jnp.array([100, 100], dtype=jnp.int32)

    # Compile env.init under XLA to guarantee no dynamic control flow failures
    jitted_init = jax.jit(env.init)
    state = jitted_init(key, seated, starting_stacks, jnp.int32(0))

    # Blind posts are the first two history entries
    assert int(state.action_count) == 2
    assert int(state.current_round) == int(Round.PRE_FLOP)