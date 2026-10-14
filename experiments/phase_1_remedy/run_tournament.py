"""Phase 1 (remedy) round robin: every pair of final bots plus RandomBot, heads-up by lineage at 6-max.

    uv run python experiments/phase_1_remedy/run_tournament.py                 # seeds 0 1
    uv run python experiments/phase_1_remedy/run_tournament.py --seeds 2 3     # add more hands
    uv run python experiments/phase_1_remedy/run_tournament.py --summarise     # rebuild summaries only

Each (pair, seed) block is `tables x hands` hands. Bot A sits in seats 0/2/4 on even tables and
1/3/5 on odd tables; bot B takes the others. Stacks are 100bb with rebuys, blinds 1/2.
Blocks are written as they finish and skipped if already present, so runs can be resumed or
extended with new seeds. See README.md in this directory for the storage layout.
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
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import poker_bots  # noqa: E402
from bots.registry import get_bot, list_bots  # noqa: E402
from poker_bots.poker_game import make_poker_env  # noqa: E402

RESULTS = HERE / "results"
SEAT_HANDS = RESULTS / "seat_hands"
PHASE = "phase_1_remedy"
ROSTER = {name.split("/")[-1]: get_bot(name) for name in list_bots(PHASE, finals_only=True) + ["builtin/random"]}
MAX_PLAYERS, SMALL_BLIND, BIG_BLIND = 6, 1, 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def engine_digest() -> str:
    """One hash over all engine source files, recorded per block so mixed-engine results are detectable."""
    files = sorted(Path(poker_bots.__file__).parent.glob("*.py"))
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16]


def repo_relative(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def block_path(bot_a: str, bot_b: str, seed: int) -> Path:
    return SEAT_HANDS / f"{bot_a}__{bot_b}" / f"seed_{seed}.parquet"


def run_block(bot_a: str, bot_b: str, seed: int, tables: int, hands: int) -> pl.DataFrame:
    env = make_poker_env((ROSTER[bot_a](), ROSTER[bot_b]()), max_players=MAX_PLAYERS, small_blind=SMALL_BLIND, big_blind=BIG_BLIND)
    keys = jax.random.split(jax.random.PRNGKey(seed), tables)
    seated = jnp.ones((tables, MAX_PLAYERS), dtype=jnp.bool_)
    pattern = jnp.arange(MAX_PLAYERS) % 2
    seat_bot = jnp.where((jnp.arange(tables) % 2 == 0)[:, None], pattern, 1 - pattern).astype(jnp.int32)
    states = jax.vmap(env.init, in_axes=(0, 0, None, None, 0))(keys, seated, None, 0, seat_bot)
    _, records, num_recorded = jax.device_get(jax.jit(jax.vmap(lambda state: env.run_hands(state, hands)))(states))

    table, hand, seat = np.meshgrid(np.arange(tables), np.arange(hands), np.arange(MAX_PLAYERS), indexing="ij")
    keep = records.seated & (hand < num_recorded[:, None, None])
    names = np.array([bot_a, bot_b])
    return pl.DataFrame({
        "pair": f"{bot_a}__{bot_b}",
        "seed": np.full(keep.sum(), seed, dtype=np.uint32),
        "table": table[keep].astype(np.uint16),
        "hand": hand[keep].astype(np.uint32),
        "seat": seat[keep].astype(np.uint8),
        "bot": names[records.seat_bot[keep]],
        "button": np.broadcast_to(records.button[:, :, None], keep.shape)[keep].astype(np.uint8),
        "starting_stack": records.starting_stacks[keep].astype(np.int32),
        "net_chips": (records.ending_stacks - records.starting_stacks)[keep].astype(np.int32),
        "rebuy": records.rebuys[keep].astype(np.int32),
        "cash_out": records.cash_outs[keep].astype(np.int32),
        "showdown": (np.broadcast_to(records.went_to_showdown[:, :, None], keep.shape) & records.in_hand)[keep],
    })


def summarise() -> None:
    seat_hands = pl.scan_parquet(SEAT_HANDS / "**" / "*.parquet", hive_partitioning=False)
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
            pl.col("bot").cast(pl.String),
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
            pl.col("bb_per_100").filter(pl.col("opponent") != "random").mean().alias("mean_vs_finals"),
            (pl.col("z") > 2).sum().alias("clear_wins"),
            (pl.col("z") < -2).sum().alias("clear_losses"),
        )
        .sort("mean_vs_finals", descending=True, nulls_last=True)
    )
    pairs.write_parquet(RESULTS / "pairs.parquet")
    standings.write_parquet(RESULTS / "standings.parquet")
    with pl.Config(tbl_rows=100, tbl_cols=20, float_precision=1, tbl_hide_dataframe_shape=True):
        print(pairs.drop("pair", "seat_hands"))
        print(standings)


def write_manifest(args: argparse.Namespace, timings: dict) -> None:
    manifest_path = RESULTS / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"blocks": {}}
    engine_dir = Path(poker_bots.__file__).parent
    manifest.update({
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "format": {"max_players": MAX_PLAYERS, "small_blind": SMALL_BLIND, "big_blind": BIG_BLIND,
                   "starting_stack_bb": 100, "rebuy": True, "seating": "A in even seats on even tables, odd seats on odd tables"},
        "engine": {
            "git_head": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("--no-optional-locks", "status", "--porcelain", "--", "src")),
            "files_sha256": {p.name: sha256(p) for p in sorted(engine_dir.glob("*.py"))},
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
        },
        "bots": {
            name: {"metadata": cls().metadata(), "source": repo_relative(Path(sys.modules[cls.__module__].__file__)),
                   "source_sha256": sha256(Path(sys.modules[cls.__module__].__file__))}
            for name, cls in ROSTER.items()
        },
    })
    for key, value in timings.items():
        manifest["blocks"][key] = {"tables": args.tables, "hands": args.hands, **value}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--hands", type=int, default=250, help="hands per table per seed block")
    parser.add_argument("--bots", nargs="+", default=list(ROSTER), choices=list(ROSTER))
    parser.add_argument("--summarise", action="store_true", help="only rebuild pairs/standings from stored blocks")
    args = parser.parse_args()

    if not args.summarise:
        timings = {}
        for (bot_a, bot_b), seed in itertools.product(itertools.combinations(args.bots, 2), args.seeds):
            path = block_path(bot_a, bot_b, seed)
            if path.exists():
                continue
            start = time.time()
            frame = run_block(bot_a, bot_b, seed, args.tables, args.hands)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(path, compression="zstd")
            seconds = time.time() - start
            timings[f"{bot_a}__{bot_b}/seed={seed}"] = {"seconds": round(seconds, 1), "finished_at": datetime.now(timezone.utc).isoformat(),
                                                    "engine_sha256": engine_digest()}
            write_manifest(args, timings)
            print(f"{bot_a} vs {bot_b} seed {seed}: {args.tables * args.hands} hands in {seconds:.0f}s", flush=True)
    summarise()


if __name__ == "__main__":
    main()
