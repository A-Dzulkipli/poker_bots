"""Scratch diagnostic: where does the candidate win/lose chips vs one opponent? (seat 0 tables only)"""
import sys, numpy as np, jax, jax.numpy as jnp
import bench
from poker_bots.poker_game import make_poker_env
cand, opp, tables = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16
env = make_poker_env((bench.load(cand), bench.load(opp)), max_players=2, small_blind=1, big_blind=2)
keys = jax.random.split(jax.random.PRNGKey(7), tables)
seated = jnp.ones((tables, 2), dtype=jnp.bool_)
seat_bot = jnp.tile(jnp.array([0, 1], jnp.int32), (tables, 1))
states = jax.vmap(env.init, in_axes=(0, 0, None, None, 0))(keys, seated, None, 0, seat_bot)
_, r, n = jax.device_get(jax.jit(jax.vmap(lambda s: env.run_hands(s, 250)))(states))
net = (r.ending_stacks - r.starting_stacks)[..., 0] / 2.0
sd = r.went_to_showdown
i_folded = ~r.in_hand[..., 0]
opp_folded = ~r.in_hand[..., 1]
streets = r.board_dealt
H = net.size
def rep(name, m):
    print(f"{name:28s} freq {m.mean():6.3f}  bb/hand-contrib {net[m].sum()/H:+7.3f}  avg {net[m].mean() if m.any() else 0:+7.2f}")
for s, nm in [(0, "pre"), (3, "flop"), (4, "turn"), (5, "river")]:
    rep(f"I fold {nm}", i_folded & (streets == s))
    rep(f"opp folds {nm}", opp_folded & (streets == s))
rep("showdown", sd)
# opponent action frequency: raises per hand, by street
h = r.history
for st in range(4):
    opp_raise = ((h.player_id == 1) & (h.action_type == 2) & h.is_valid & (h.round == st)).sum(-1)
    my_raise = ((h.player_id == 0) & (h.action_type == 2) & h.is_valid & (h.round == st)).sum(-1)
    print(f"street {st}: opp raises/hand {opp_raise.mean():.2f}  my raises/hand {my_raise.mean():.2f}")
allin = (h.is_all_in & h.is_valid & (h.player_id == 1)).any(-1)
rep("opp went all-in", allin)
print("total bb/100", 100 * net.sum() / H)
