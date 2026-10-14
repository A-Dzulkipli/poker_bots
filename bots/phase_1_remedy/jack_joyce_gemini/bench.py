"""Benchmark a bot version. Copy verbatim into your bot directory; do not edit.

    python bench.py v2:Bot random --seed 0
    python bench.py v2:Bot v1:Bot --seed 0 --tables 32 --hands 250

Six-max, the two bots alternate seats (3 each), 100bb stacks with rebuys.
Prints the candidate's bb/100 and standard error over seat-hands.
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PACKAGE = Path(__file__).resolve().parent.name

from poker_bots.poker_game import RandomBot, make_poker_env  # noqa: E402


def load(spec: str):
    if spec == "random":
        return RandomBot()
    module, name = spec.split(":")
    return getattr(importlib.import_module(f"{PACKAGE}.{module}"), name)()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("opponent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--hands", type=int, default=250)
    args = parser.parse_args()

    env = make_poker_env((load(args.candidate), load(args.opponent)), max_players=6, small_blind=1, big_blind=2)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.tables)
    seated = jnp.ones((args.tables, 6), dtype=jnp.bool_)
    seat_bot = jnp.tile(jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32), (args.tables, 1))
    tables = jax.vmap(env.init, in_axes=(0, 0, None, None, 0))(keys, seated, None, 0, seat_bot)

    start = time.time()
    run = jax.jit(jax.vmap(lambda table: env.run_hands(table, args.hands)))
    _, records, num_recorded = jax.device_get(run(tables))
    recorded = np.arange(args.hands)[None, :] < num_recorded[:, None]
    net = (records.ending_stacks - records.starting_stacks) / env.big_blind
    for index, label in ((0, "candidate " + args.candidate), (1, "opponent " + args.opponent)):
        sample = net[records.seated & recorded[:, :, None] & (records.seat_bot == index)]
        mean, se = 100 * sample.mean(), 100 * sample.std(ddof=1) / np.sqrt(len(sample))
        print(f"{label:>24}: {mean:+8.2f} bb/100  (SE {se:.2f}, {len(sample)} seat-hands)")
    print(f"seed {args.seed}, {args.tables} tables x {args.hands} hands, {time.time() - start:.1f}s incl. compile")


if __name__ == "__main__":
    main()