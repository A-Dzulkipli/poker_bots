"""Round-robin tournaments that store full hand histories (storage format 2, phase 2 onwards).

A phase script supplies the roster and the table size, then calls `main`:

    ROSTER = {"alan_wake_claude": ("phase_1_remedy/alan_wake_claude", get_bot("phase_1_remedy/alan_wake_claude")), ...}
    main(Path(__file__).parent / "results", ROSTER, max_players=2)

Each (pair, seed) block is `tables x hands` hands. Bot A sits in the even seats on even tables and
the odd seats on odd tables (heads-up: seat 0 on even tables, seat 1 on odd), bot B in the others.
Stacks are 100bb with rebuys, blinds 1/2. Blocks are written as they finish and skipped if already
present, so runs can be resumed or extended with new seeds. See experiments/README.md for the layout.
"""
import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import polars as pl

import poker_bots
from poker_bots.poker_game import make_poker_env

from hand_records import FORMAT_VERSION, records_to_frames

REPO = Path(__file__).resolve().parents[1]
TABLES = ("seat_hands", "hands", "actions")
SMALL_BLIND, BIG_BLIND = 1, 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def engine_digest() -> str:
    """One hash over all engine source files, recorded per block so mixed-engine results are detectable."""
    files = sorted(Path(poker_bots.__file__).parent.glob("*.py"))
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16]


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_file(cls: type) -> Path:
    return Path(sys.modules[cls.__module__].__file__)


def repo_relative(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def seat_assignment(tables: int, max_players: int) -> jax.Array:
    """seat_bot per table: bot 0 in even seats on even tables and odd seats on odd tables."""
    pattern = jnp.arange(max_players) % 2
    return jnp.where((jnp.arange(tables) % 2 == 0)[:, None], pattern, 1 - pattern).astype(jnp.int32)


def regenerate_block(bot_a, bot_b, seed: int, tables: int, hands: int, max_players: int):
    """Play one block exactly as the tournament does. Takes bot instances; returns (env, records,
    num_recorded) on the host, with records shaped (table, hand, ...). Deterministic per engine
    and platform, so stored blocks can be replayed and checked against their parquet rows."""
    env = make_poker_env((bot_a, bot_b), max_players=max_players, small_blind=SMALL_BLIND, big_blind=BIG_BLIND)
    keys = jax.random.split(jax.random.PRNGKey(seed), tables)
    seated = jnp.ones((tables, max_players), dtype=jnp.bool_)
    states = jax.vmap(env.init, in_axes=(0, 0, None, None, 0))(keys, seated, None, 0, seat_assignment(tables, max_players))
    _, records, num_recorded = jax.device_get(jax.jit(jax.vmap(lambda state: env.run_hands(state, hands)))(states))
    return env, records, num_recorded


def block_path(results: Path, table: str, pair: str, seed: int) -> Path:
    return results / table / pair / f"seed_{seed}.parquet"


def summarise(results: Path) -> None:
    seat_hands = pl.scan_parquet(results / "seat_hands" / "**" / "*.parquet", hive_partitioning=False)
    per_table = (
        seat_hands.group_by("pair", "seed", "table", "bot")
        .agg(pl.col("net_chips").sum(), pl.len().alias("seat_hands"), pl.col("hand").n_unique().alias("hands"))
        .with_columns((100 * pl.col("net_chips") / pl.col("seat_hands") / BIG_BLIND).alias("bb_per_100"))
        .collect()
    )
    pairs = (
        per_table.group_by("pair", "bot")
        .agg(
            pl.col("hands").sum(),
            pl.col("seat_hands").sum(),
            pl.len().alias("tables"),
            (100 * pl.col("net_chips").sum() / pl.col("seat_hands").sum() / BIG_BLIND).alias("bb_per_100"),
            (pl.col("bb_per_100").std() / pl.len().sqrt()).alias("se"),
        )
        .with_columns(
            pl.col("pair").str.split("__").list.get(0).alias("bot_a"),
            pl.col("pair").str.split("__").list.get(1).alias("bot_b"),
        )
        .with_columns(pl.when(pl.col("bot") == pl.col("bot_a")).then(pl.col("bot_b")).otherwise(pl.col("bot_a")).alias("opponent"))
        .with_columns((pl.col("bb_per_100") / pl.col("se")).alias("z"))
        .select("bot", "opponent", "bb_per_100", "se", "z", "hands", "seat_hands", "tables", "pair")
        .sort("bot", "opponent")
    )
    standings = (
        pairs.group_by("bot")
        .agg(
            pl.col("bb_per_100").mean().alias("mean_bb_per_100"),
            (pl.col("se").pow(2).sum().sqrt() / pl.len()).alias("se"),
            pl.col("bb_per_100").filter(pl.col("opponent") != "random").mean().alias("mean_vs_non_random"),
            (pl.col("z") > 2).sum().alias("clear_wins"),
            (pl.col("z") < -2).sum().alias("clear_losses"),
        )
        .sort("mean_vs_non_random", descending=True, nulls_last=True)
    )
    pairs.write_parquet(results / "pairs.parquet")
    standings.write_parquet(results / "standings.parquet")
    with pl.Config(tbl_rows=200, tbl_cols=20, float_precision=1, tbl_hide_dataframe_shape=True):
        print(pairs.drop("pair", "seat_hands"))
        print(standings)


def write_manifest(results: Path, roster: dict, max_players: int, blocks: dict) -> None:
    path = results / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"blocks": {}}
    engine_dir = Path(poker_bots.__file__).parent
    manifest.update({
        "format_version": FORMAT_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "format": {"max_players": max_players, "small_blind": SMALL_BLIND, "big_blind": BIG_BLIND,
                   "starting_stack_bb": 100, "rebuy": True,
                   "seating": "A in even seats on even tables, odd seats on odd tables"},
        "engine": {
            "git_head": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("--no-optional-locks", "status", "--porcelain", "--", "src")),
            "files_sha256": {p.name: sha256(p) for p in sorted(engine_dir.glob("*.py"))},
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
        },
        "bots": {
            name: {"registry_name": registry_name, "metadata": cls().metadata(),
                   "source": repo_relative(source_file(cls)), "source_sha256": sha256(source_file(cls))}
            for name, (registry_name, cls) in roster.items()
        },
    })
    manifest["blocks"].update(blocks)
    path.write_text(json.dumps(manifest, indent=2, default=str))


def main(results: Path, roster: dict[str, tuple[str, type]], max_players: int, focus: tuple[str, ...] = ()) -> None:
    """roster maps a short, unique bot name to (registry name, bot class); pairs follow its order.
    With `focus`, only pairs with exactly one focus bot are played (focus bots against the field)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--hands", type=int, default=250, help="hands per table per seed block")
    parser.add_argument("--bots", nargs="+", default=list(roster), choices=list(roster))
    parser.add_argument("--summarise", action="store_true", help="only rebuild pairs/standings from stored blocks")
    args = parser.parse_args()

    if not args.summarise:
        pairs = [(a, b) for a, b in itertools.combinations(args.bots, 2) if not focus or (a in focus) != (b in focus)]
        for seed, (bot_a, bot_b) in itertools.product(args.seeds, pairs):
            pair = f"{bot_a}__{bot_b}"
            if all(block_path(results, table, pair, seed).exists() for table in TABLES):
                continue
            start = time.time()
            bots = (roster[bot_a][1](), roster[bot_b][1]())
            _, records, num_recorded = regenerate_block(*bots, seed, args.tables, args.hands, max_players)
            kinds = [bot.metadata().get("kind", "bot") for bot in bots]
            frames = records_to_frames(records, num_recorded, [bot_a, bot_b], {"pair": pair, "seed": seed}, kinds)
            for table, frame in frames.items():
                path = block_path(results, table, pair, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.write_parquet(path, compression="zstd")
            seconds = time.time() - start
            write_manifest(results, roster, max_players, {f"{pair}/seed={seed}": {
                "tables": args.tables, "hands": args.hands, "seconds": round(seconds, 1),
                "finished_at": datetime.now(timezone.utc).isoformat(), "engine_sha256": engine_digest()}})
            print(f"{bot_a} vs {bot_b} seed {seed}: {args.tables * args.hands} hands in {seconds:.0f}s", flush=True)
    summarise(results)
