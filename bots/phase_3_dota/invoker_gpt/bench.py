"""Heads-up benchmark for phase 3. Do not edit.

    uv run python bench.py check                       # environment check: run this first
    uv run python bench.py v2:Bot core --seed 0        # against the core benchmarks (phase-2 finals + aggressor)
    uv run python bench.py v2:Bot legacy --seed 0      # against the phase-1 finals
    uv run python bench.py v2:Bot all --seed 0         # both groups
    uv run python bench.py v2:Bot aggressor            # against one opponent by name
    uv run python bench.py v2:Bot v1:Bot --seed 0      # against your own earlier version
    uv run python bench.py v2:Bot random

Two-player tables, 100bb stacks with rebuys, blinds 1/2. The candidate sits in seat 0 on even
tables and seat 1 on odd tables. Prints the candidate's bb/100 with a table-clustered standard
error (standard deviation of per-table bb/100 divided by sqrt(tables)).
"""
import argparse
import hashlib
import importlib
import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
PACKAGE = HERE.name
OPPONENTS = HERE / "opponents"
GROUPS = {group: sorted(p.stem for p in (OPPONENTS / group).glob("*.py")) for group in ("core", "legacy")}
GROUPS["all"] = GROUPS["core"] + GROUPS["legacy"]
ENGINE_SHA256 = "295477eff7637056"


def check() -> None:
    """Fail fast, with a message a human can act on, if the project environment is not reachable."""
    print(f"python      {sys.executable} ({sys.version.split()[0]})")
    try:
        import jax
        import poker_bots
    except ImportError as error:
        sys.exit(f"FAILED: {error}.\nThis shell cannot reach the project's Python environment. Stop and tell the "
                 "requester: the session must run on the machine that holds the repository, with the repository "
                 "readable (writes can stay limited to this folder), and be started with `uv run` from here.")
    engine = Path(poker_bots.__file__).parent
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in sorted(engine.glob("*.py")))).hexdigest()[:16]
    print(f"jax         {jax.__version__} ({jax.default_backend()})")
    print(f"poker_bots  {engine}")
    print(f"engine      sha256 {digest} ({'matches the contract' if digest == ENGINE_SHA256 else 'DIFFERS from the contract ' + ENGINE_SHA256 + '; tell the requester'})")
    start = time.time()
    mean, _, hands = play("random", "aggressor", 0, 2, 20)
    print(f"smoke test  random vs aggressor, {hands} hands in {time.time() - start:.0f}s: OK")


def load(spec: str):
    from poker_bots.poker_game import RandomBot
    if spec == "random":
        return RandomBot()
    for group in ("core", "legacy"):
        if spec in GROUPS[group]:
            module_spec = importlib.util.spec_from_file_location(f"opponents_{spec}", OPPONENTS / group / f"{spec}.py")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            return module.Bot()
    module, name = spec.split(":")
    return getattr(importlib.import_module(f"{PACKAGE}.{module}"), name)()


def play(candidate: str, opponent: str, seed: int, tables: int, hands: int) -> tuple[float, float, int]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from poker_bots.poker_game import make_poker_env

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
    if sys.argv[1:] == ["check"]:
        return check()
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", help="vK:Bot, or 'check' on its own")
    parser.add_argument("opponent", help="'core', 'legacy', 'all', an opponent's name, 'random', or vK:Bot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--hands", type=int, default=250)
    args = parser.parse_args()

    opponents = GROUPS.get(args.opponent, [args.opponent])
    results = {}
    for opponent in opponents:
        start = time.time()
        mean, se, hands = play(args.candidate, opponent, args.seed, args.tables, args.hands)
        results[opponent] = mean
        print(f"{args.candidate} vs {opponent:>20}: {mean:+8.2f} bb/100  (SE {se:.2f}, {hands} hands, seed {args.seed}, {time.time() - start:.0f}s)", flush=True)
    for group in ("core", "legacy"):
        scored = [results[name] for name in GROUPS[group] if name in results]
        if len(scored) == len(GROUPS[group]):
            print(f"{args.candidate} mean vs {group}: {sum(scored) / len(scored):+8.2f} bb/100")


if __name__ == "__main__":
    main()
