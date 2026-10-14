"""Scratch: ablation evidence for the range model.

Plays the deployed Bot (range on) against Bot(uniform_range=True) — the identical strategy
deciding against a uniformly random opponent hand — head to head, and plays the uniform
variant against a subset of the field at the same seeds/sizes as the spot runs.
Usage: uv run python scratch_ablation.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
import jax.numpy as jnp
import numpy as np
from poker_bots.poker_game import make_poker_env
from scherezard_glm.v5 import Bot
import bench


def match(cand, opp, seed, tables, hands):
    env = make_poker_env((cand, opp), max_players=2, small_blind=1, big_blind=2)
    keys = jax.random.split(jax.random.PRNGKey(seed), tables)
    seated = jnp.ones((tables, 2), jnp.bool_)
    seat = jnp.where((jnp.arange(tables) % 2 == 0)[:, None], jnp.array([0, 1]), jnp.array([1, 0])).astype(jnp.int32)
    states = jax.vmap(env.init, in_axes=(0, 0, None, None, 0))(keys, seated, None, 0, seat)
    _, recs, n = jax.device_get(jax.jit(jax.vmap(lambda s: env.run_hands(s, hands)))(states))
    idx = np.arange(hands)[None, :, None]
    mine = (idx < n[:, None, None]) & recs.seated & (recs.seat_bot == 0)
    net = np.where(mine, recs.ending_stacks - recs.starting_stacks, 0)
    per_table = 100 * net.sum(axis=(1, 2)) / mine.sum(axis=(1, 2)) / env.big_blind
    return float(per_table.mean()), float(per_table.std(ddof=1) / np.sqrt(tables))


if __name__ == "__main__":
    mean, se = match(Bot(), Bot(uniform_range=True), 0, 32, 250)
    print(f"v5(range) vs v5(uniform) head-to-head: {mean:+.2f} bb/100 (SE {se:.2f})")
    for name in ["aggressor", "windrunner_glm", "jesse_faden_glm"]:
        mean, se = match(Bot(uniform_range=True), bench.load(name), 0, 32, 250)
        print(f"v5(uniform) vs {name:>18}: {mean:+8.2f} bb/100 (SE {se:.2f})")
