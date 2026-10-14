"""Heads-up benchmark for phase 2. Do not edit.

    uv run python bench.py v2:Bot finals --seed 0      # against every phase-1 final in opponents/
    uv run python bench.py v2:Bot alan_wake_claude     # against one phase-1 final
    uv run python bench.py v2:Bot v1:Bot --seed 0      # against your own earlier version
    uv run python bench.py v2:Bot random

Two-player tables, 100bb stacks with rebuys, blinds 1/2. The candidate sits in seat 0 on even
tables and seat 1 on odd tables. Prints the candidate's bb/100 with a table-clustered standard
error (standard deviation of per-table bb/100 divided by sqrt(tables)).
"""
import argparse
import importlib
import importlib.util
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
PACKAGE = HERE.name
OPPONENTS = HERE / "opponents"
FINALS = sorted(p.stem for p in OPPONENTS.glob("*.py"))

from poker_bots.poker_game import RandomBot, make_poker_env  # noqa: E402


def load(spec: str):
    if spec == "random":
        return RandomBot()
    if spec in FINALS:
        module_spec = importlib.util.spec_from_file_location(f"opponents_{spec}", OPPONENTS / f"{spec}.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module.Bot()
    module, name = spec.split(":")
    return getattr(importlib.import_module(f"{PACKAGE}.{module}"), name)()


def play(candidate: str, opponent: str, seed: int, tables: int, hands: int) -> tuple[float, float, int]:
    env = make_poker_env((load(candidate), load(opponent)), max_players=2, small_blind=1, big_blind=2)
    keys = jax.random.split(jax.random.PRNGKey(seed), tables)
    seated = jnp.ones((tables, 2), dtype=jnp.bool_)
    seat_bot = jnp.where((jnp.arange(tables) % 2 == 0)[:, None], jnp.array([0, 1]), jnp.array([1, 0])).astype(jnp.int32)
    states = jax.vmap(env.init, in_axes=(0, 0, None, None, 0))(keys, seated, None, 0, seat_bot)
    _, records, num_recorded = jax.device_get(jax.jit(jax.vmap(lambda state: env.run_hands(state, hands)))(states))
    recorded = np.arange(hands)[None, :, None] < num_recorded[:, None, None]
    mine = recorded & records.seated & (records.seat_bot == 0)
    net = np.where(mine, records.ending_stacks - records.starting_stacks, 0).sum(axis=(1, 2))
    per_table = 100 * net / mine.sum(axis=(1, 2)) / env.big_blind
    return float(per_table.mean()), float(per_table.std(ddof=1) / np.sqrt(tables)), int(mine.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("opponent", help="'finals', a phase-1 final's name, 'random', or vK:Bot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--hands", type=int, default=250)
    args = parser.parse_args()

    opponents = FINALS if args.opponent == "finals" else [args.opponent]
    results = []
    for opponent in opponents:
        start = time.time()
        mean, se, hands = play(args.candidate, opponent, args.seed, args.tables, args.hands)
        results.append(mean)
        print(f"{args.candidate} vs {opponent:>20}: {mean:+8.2f} bb/100  (SE {se:.2f}, {hands} hands, seed {args.seed}, {time.time() - start:.0f}s)", flush=True)
    if len(results) > 1:
        print(f"{args.candidate} mean vs finals: {np.mean(results):+8.2f} bb/100")


if __name__ == "__main__":
    main()
