"""Heads-up benchmark for phase 6. Do not edit.

    uv run python bench.py check                       # environment check: run this first
    uv run python bench.py v2:Bot field --seed 0       # all eighteen benchmark opponents, plus the weighted score
    uv run python bench.py v2:Bot kraken_gpt           # one opponent by name
    uv run python bench.py v2:Bot v1:Bot --seed 0      # against your own earlier version
    uv run python bench.py v2:Bot random

Two-player tables, 100bb stacks with rebuys, blinds 1/2. The candidate sits in seat 0 on even
tables and seat 1 on odd tables. Each table is one session of --hands hands against one opponent,
and bot state (if any) starts fresh at every table. Prints the candidate's bb/100 with a
table-clustered standard error (standard deviation of per-table bb/100 divided by sqrt(tables)),
and the bb/100 over the first EARLY_HANDS hands of each session against the rest. Opponents are
loaded from ../field/, which bots must not read. Phase 6 is stateless: a vK:Bot candidate whose
init_state does not return None, or that defines observe, is refused.

The weighted score: each opponent has a strength rank (STRENGTH_RANK, 1 = weakest). A win counts
rank / mean_rank and a loss counts (N + 1 - rank) / mean_rank, so beating the strongest bots and not
losing to the weakest matter most. The score is the mean of the weighted results over the field.
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
OPPONENTS = HERE.parent / "field"
ENGINE_SHA256 = "295477eff7637056"
EARLY_HANDS = 50
# Order of mean bb/100 against non-random opponents in the phase-5 round robin, weakest first.
STRENGTH_RANK = {name: rank for rank, name in enumerate([
    "random", "jack_joyce_gemini", "jesse_faden_glm", "bassander_lin_glm", "max_payne_gpt", "alan_wake_claude",
    "aggressor", "valka_onderra_claude", "vicious_glm", "hadrian_marlowe_gpt", "meepo_claude", "invoker_gpt",
    "windrunner_glm", "faye_gpt", "spike_claude", "megalodon_glm", "leviathan_claude", "kraken_gpt",
], start=1)}
FIELD = sorted(STRENGTH_RANK, key=STRENGTH_RANK.get, reverse=True)


def weight(opponent: str, result: float) -> float:
    rank, mean_rank = STRENGTH_RANK[opponent], (len(STRENGTH_RANK) + 1) / 2
    return (rank if result > 0 else len(STRENGTH_RANK) + 1 - rank) / mean_rank


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
    play("random", "aggressor", 0, 2, 20)
    print(f"smoke test  random vs aggressor, 40 hands in {time.time() - start:.0f}s: OK")


def load(spec: str):
    from poker_bots.poker_game import RandomBot
    if spec == "random":
        return RandomBot()
    if (OPPONENTS / f"{spec}.py").exists():
        module_spec = importlib.util.spec_from_file_location(f"field_{spec}", OPPONENTS / f"{spec}.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module.Bot()
    import jax
    module, name = spec.split(":")
    bot = getattr(importlib.import_module(f"{PACKAGE}.{module}"), name)()
    if hasattr(bot, "observe") or bot.init_state(jax.random.PRNGKey(0)) is not None:
        sys.exit(f"REFUSED: {spec} keeps memory. Phase 6 bots must return None from init_state and not define observe.")
    return bot


def play(candidate: str, opponent: str, seed: int, tables: int, hands: int) -> tuple[float, float, float, float]:
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
    hand_index = np.arange(hands)[None, :, None]
    mine = (hand_index < num_recorded[:, None, None]) & records.seated & (records.seat_bot == 0)
    net = np.where(mine, records.ending_stacks - records.starting_stacks, 0)

    def bb_per_100(mask):
        return 100 * np.where(mask, net, 0).sum() / max(mask.sum(), 1) / env.big_blind

    per_table = 100 * net.sum(axis=(1, 2)) / mine.sum(axis=(1, 2)) / env.big_blind
    early = mine & (hand_index < EARLY_HANDS)
    return (float(per_table.mean()), float(per_table.std(ddof=1) / np.sqrt(tables)),
            float(bb_per_100(early)), float(bb_per_100(mine & ~early)))


def main() -> None:
    if sys.argv[1:] == ["check"]:
        return check()
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", help="vK:Bot, or 'check' on its own")
    parser.add_argument("opponent", help="'field', an opponent's name, 'random', or vK:Bot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--hands", type=int, default=250, help="hands per table, i.e. session length")
    args = parser.parse_args()

    opponents = FIELD if args.opponent == "field" else [args.opponent]
    results = {}
    for opponent in opponents:
        start = time.time()
        mean, se, early, late = play(args.candidate, opponent, args.seed, args.tables, args.hands)
        results[opponent] = mean
        print(f"{args.candidate} vs {opponent:>20}: {mean:+8.2f} bb/100  (SE {se:.2f}; first {EARLY_HANDS} hands "
              f"{early:+.1f}, rest {late:+.1f}; {args.tables}x{args.hands}, seed {args.seed}, {time.time() - start:.0f}s)", flush=True)
    if args.opponent == "field":
        weighted = sum(weight(name, value) * value for name, value in results.items()) / len(results)
        plain = sum(value for name, value in results.items() if name != "random") / (len(results) - 1)
        print(f"{args.candidate} weighted score: {weighted:+8.2f}   (plain mean vs non-random: {plain:+.2f} bb/100)")


if __name__ == "__main__":
    main()
