"""Read only public action/outcome records from fresh benchmark games."""
import sys
import jax
import jax.numpy as jnp
import bench
from poker_bots.poker_game import make_poker_env

candidate, opponent = sys.argv[1:3]
tables, hands = 64, 250
env = make_poker_env((bench.load(candidate), bench.load(opponent)), max_players=2, small_blind=1, big_blind=2)
keys = jax.random.split(jax.random.PRNGKey(1), tables)
seated = jnp.ones((tables, 2), dtype=jnp.bool_)
seat_bot = jnp.where((jnp.arange(tables) % 2 == 0)[:, None], jnp.array([0, 1]), jnp.array([1, 0])).astype(jnp.int32)
states = jax.vmap(env.init, in_axes=(0, 0, None, None, 0))(keys, seated, None, 0, seat_bot)
_, records, _ = jax.device_get(jax.jit(jax.vmap(lambda state: env.run_hands(state, hands)))(states))
hero = records.seat_bot == 0
net = jnp.sum(jnp.where(hero, records.ending_stacks - records.starting_stacks, 0), axis=-1) / 2
hero_seat = jnp.argmax(hero, axis=-1)
h = records.history
hero_action = h.is_valid & (h.player_id == hero_seat[..., None])
opponent_action = h.is_valid & ~hero_action
for title, mask in [('all', jnp.ones_like(net, dtype=bool)), ('preflop ended', records.board_dealt == 0), ('flop ended', records.board_dealt == 3), ('turn ended', records.board_dealt == 4), ('river ended', records.board_dealt == 5), ('showdown', records.went_to_showdown), ('no showdown', ~records.went_to_showdown)]:
    print(title, 'hands', int(mask.sum()), 'net bb/100 of all hands', float(100 * jnp.where(mask, net, 0).mean()))
for title, mask in [('hero', hero_action), ('opponent', opponent_action)]:
    for street in range(4):
        valid = mask & (h.round == street) & (h.action_type < 3)
        print(title, 'street', street, 'actions', int(valid.sum()), 'fold/call/raise fractions', [float(jnp.sum(valid & (h.action_type == a)) / jnp.maximum(valid.sum(), 1)) for a in range(3)])
    previous_other_raise = jnp.roll(h.is_valid & ~mask & (h.action_type == 2), 1, axis=-1)
    facing_bet = mask & previous_other_raise & (h.round == jnp.roll(h.round, 1, axis=-1)) & (jnp.arange(h.is_valid.shape[-1]) > 0)
    print(title, 'folds facing bet', float(jnp.sum(facing_bet & (h.action_type == 0)) / jnp.maximum(facing_bet.sum(), 1)))
