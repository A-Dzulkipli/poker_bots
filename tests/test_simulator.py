import jax
import jax.numpy as jnp
from numpy.testing import assert_array_equal

from poker_bots.poker_game import RandomBot, Round, TestBot, make_poker_env


def test_init_chip_conservation_multiway():
    num_players = 3
    bots = (TestBot(),)
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
    bots = (TestBot(),)
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
    bots = (TestBot(),)
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
    bots = (TestBot(),)
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

def make_showdown_state(env, starting_stacks, stacks, in_hand, button=0):
    num_players = len(starting_stacks)
    state = env.init(jax.random.PRNGKey(0), jnp.ones((num_players,), dtype=jnp.bool_), jnp.array(starting_stacks), button)
    return state._replace(
        starting_stacks=jnp.array(starting_stacks, dtype=jnp.int32),
        stacks=jnp.array(stacks, dtype=jnp.int32),
        in_hand=jnp.array(in_hand)
    )


def assert_chips_conserved(state, payouts):
    assert int(jnp.sum(payouts)) == int(jnp.sum(state.starting_stacks - state.stacks))


def test_payouts_three_way_all_in_side_pot():
    env = make_poker_env((TestBot(),), max_players=3)
    # A all-in 50, B all-in 100, C covers with 150
    state = make_showdown_state(env, [50, 100, 150], [0, 0, 50], [True, True, True])
    payouts = env.get_payouts(state, jnp.array([3, 2, 1]))

    # A wins main pot 3 x 50, B wins side pot 2 x 50
    assert_array_equal(payouts, [150, 100, 0])
    assert_chips_conserved(state, payouts)


def test_payouts_folded_chips_stay_in_pot():
    env = make_poker_env((TestBot(),), max_players=3)
    # B put in 20 then folded with the best cards
    state = make_showdown_state(env, [100, 100, 100], [60, 80, 60], [True, False, True])
    payouts = env.get_payouts(state, jnp.array([3, 5, 1]))

    assert_array_equal(payouts, [100, 0, 0])
    assert_chips_conserved(state, payouts)


def test_payouts_split_odd_chips_left_of_button():
    env = make_poker_env((TestBot(),), max_players=4)
    # Three-way tie for 15, plus 2 from folded D: 17 splits 5/5/5 with 2 odd chips
    state = make_showdown_state(env, [100, 100, 100, 100], [95, 95, 95, 98], [True, True, True, False], button=0)
    payouts = env.get_payouts(state, jnp.array([4, 4, 4, 9]))

    # Odd chips go to the first winners after the button: seats 1 and 2
    assert_array_equal(payouts, [5, 6, 6, 0])
    assert_chips_conserved(state, payouts)


def test_payouts_uncalled_bet_returned():
    env = make_poker_env((TestBot(),), max_players=2)
    # A shoves 100 into B's 30 and loses
    state = make_showdown_state(env, [100, 30], [0, 0], [True, True])
    payouts = env.get_payouts(state, jnp.array([1, 2]))

    assert_array_equal(payouts, [70, 60])
    assert_chips_conserved(state, payouts)


def test_random_play_invariants():
    max_players = 6
    max_actions = 100
    num_tables = 64
    num_steps = 3000
    env = make_poker_env((RandomBot(),), max_players=max_players, max_actions=max_actions)

    keys = jax.random.split(jax.random.PRNGKey(0), num_tables)
    starting_stacks = jax.random.randint(jax.random.PRNGKey(1), (num_tables, max_players), 20, 200)
    seated = jnp.ones((num_tables, max_players), dtype=jnp.bool_)
    states = jax.vmap(env.init, in_axes=(0, 0, 0, None))(keys, seated, starting_stacks, 0)

    def scan_step(state, _):
        state = env.step(state)
        committed = state.starting_stacks - state.stacks
        checks = (
            jnp.sum(state.starting_stacks),
            jnp.min(state.stacks),
            jnp.all(state.contributed <= committed),
            state.action_count,
            ~jnp.any(state.to_act_mask) | state.to_act_mask[state.current_player],
        )
        return state, checks

    run = jax.jit(jax.vmap(lambda state: jax.lax.scan(scan_step, state, None, length=num_steps)))
    final_states, (total_chips, min_stack, contributed_ok, action_count, current_player_ok) = run(states)

    assert_array_equal(total_chips, jnp.broadcast_to(jnp.sum(starting_stacks, axis=1)[:, None], total_chips.shape))
    assert int(jnp.min(min_stack)) >= 0
    assert bool(jnp.all(contributed_ok))
    # Action cap: history writes past max_actions are silently dropped
    assert int(jnp.max(action_count)) < max_actions
    assert bool(jnp.all(current_player_ok))
    # Random shoving should bust every table down to one player
    assert bool(jnp.all(jnp.sum(final_states.seated, axis=1) == 1))


def advance_to_next_hand(env, state):
    # Fold everyone but one player out so step goes straight to next_hand
    in_hand = jnp.zeros_like(state.in_hand).at[state.current_player].set(True)
    return env.step(state._replace(in_hand=in_hand))


def test_heads_up_positions_alternate_between_hands():
    env = make_poker_env((TestBot(),), max_players=2)
    state = env.init(jax.random.PRNGKey(0), jnp.array([True, True]), jnp.array([100, 100]), 0)

    for _ in range(4):
        old_button = int(state.button)
        state = advance_to_next_hand(env, state)
        # The button posts the small blind and acts first preflop
        assert int(state.button) == 1 - old_button
        assert int(state.sb_seat) == int(state.button)
        assert int(state.bb_seat) == 1 - int(state.button)
        assert int(state.current_player) == int(state.button)


def test_heads_up_big_blind_acts_first_postflop():
    env = make_poker_env((TestBot(),), max_players=2)
    state = env.init(jax.random.PRNGKey(0), jnp.array([True, True]), jnp.array([100, 100]), 0)
    state = advance_to_next_hand(env, state)

    # Button completes, big blind checks, then the flop is dealt
    state = env.step(state)
    state = env.step(state)
    state = env.step(state)
    assert int(state.current_round) == int(Round.FLOP)
    assert int(state.current_player) == int(state.bb_seat)


def test_three_to_two_players_big_blind_moves_forward():
    env = make_poker_env((TestBot(),), max_players=3)
    state = env.init(jax.random.PRNGKey(0), jnp.array([True, True, True]), jnp.array([100, 100, 100]), 0)
    # Button 0, SB 1, BB 2. Seat 1 busts this hand.
    state = state._replace(starting_stacks=state.starting_stacks.at[1].set(1), stacks=state.stacks.at[1].set(0))
    state = advance_to_next_hand(env, state._replace(current_player=jnp.int32(0)))

    # Seat 0 is the next big blind, and seat 2 becomes button and small blind
    assert int(state.num_seated) == 2
    assert int(state.bb_seat) == 0
    assert int(state.button) == 2
    assert int(state.sb_seat) == 2
    assert int(state.current_player) == 2
