"""Hands for the replay viewer, read from format-2 hand histories (play sessions and phase 2+ tournaments),
or for format-1 tournaments (outcomes only) regenerated with experiments/tournament.regenerate_block.

Everything is turned into one viewer shape per table: seat names plus a list of hands.
"""
import json
from functools import lru_cache
from pathlib import Path

import polars as pl

from bots import registry
from ui.storage import EXPERIMENTS, SESSIONS, TABLES

from hand_records import records_to_frames  # noqa: E402  (on sys.path via ui.storage)
from tournament import engine_digest, regenerate_block  # noqa: E402

STREETS = ("preflop", "flop", "turn", "river")


def split_cards(text: str) -> list[str]:
    return [text[i:i + 2] for i in range(0, len(text), 2)]


def frames_to_view(seat_hands: pl.DataFrame, hands: pl.DataFrame, actions: pl.DataFrame, max_players: int) -> dict:
    """Format-2 rows of a single table to the viewer's shape."""
    seat_rows = seat_hands.sort("hand", "seat").partition_by("hand", as_dict=True)
    action_rows = actions.sort("hand", "idx").partition_by("hand", as_dict=True)
    seat_names = ["" for _ in range(max_players)]
    out = []
    for row in hands.sort("hand").iter_rows(named=True):
        hand = {
            "hand_index": row["hand"], "button": row["button"], "sb_seat": row["sb_seat"], "bb_seat": row["bb_seat"],
            "board": split_cards(row["runout"]), "board_dealt": row["board_dealt"], "went_to_showdown": row["showdown"],
            "seated": [False] * max_players, "hole_cards": [None] * max_players, "hand_ranks": [None] * max_players,
            "starting_stacks": [0] * max_players, "ending_stacks": [0] * max_players, "payouts": [0] * max_players,
            "rebuys": [0] * max_players, "cash_outs": [0] * max_players,
        }
        for seat in seat_rows[(row["hand"],)].iter_rows(named=True):
            s = seat["seat"]
            seat_names[s] = seat["bot"]
            hand["seated"][s] = True
            hand["hole_cards"][s] = split_cards(seat["hole_cards"])
            hand["hand_ranks"][s] = seat["hand_rank"]
            hand["starting_stacks"][s] = seat["starting_stack"]
            hand["ending_stacks"][s] = seat["starting_stack"] + seat["net_chips"]
            hand["payouts"][s] = seat["payout"]
            hand["rebuys"][s] = seat["rebuy"]
            hand["cash_outs"][s] = seat["cash_out"]
        acts = action_rows.get((row["hand"],))
        hand["actions"] = [] if acts is None else [
            {"seat": a["seat"], "street": STREETS[a["street"]], "type": a["type"], "type_requested": a["type_requested"],
             "value_requested": a["value_requested"], "value_applied": a["value"], "is_all_in": a["is_all_in"], "by": a["by"]}
            for a in acts.iter_rows(named=True)
        ]
        out.append(hand)
    return {"seat_names": seat_names, "max_players": max_players, "hands": out}


# play sessions

def list_sessions(root: Path = SESSIONS) -> list[dict]:
    sessions = []
    for manifest_path in sorted(root.glob("*/manifest.json"), reverse=True):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format_version") != 2:
            continue   # sessions saved before the format-2 tables existed
        seat_hands = pl.read_parquet(manifest_path.parent / "seat_hands.parquet")
        sessions.append({
            "session": manifest["session"],
            "created_at": manifest["created_at"],
            "seats": manifest["format"]["seats"],
            "hands": manifest.get("hands", 0),
            "closed": "closed_at" in manifest,
            "human_net": int(seat_hands.filter(pl.col("by") == "human")["net_chips"].sum()),
        })
    return sessions


def load_session(session: str, root: Path = SESSIONS) -> dict:
    folder = root / session
    manifest = json.loads((folder / "manifest.json").read_text())
    frames = {name: pl.read_parquet(folder / f"{name}.parquet") for name in TABLES}
    view = frames_to_view(frames["seat_hands"], frames["hands"], frames["actions"], manifest["format"]["max_players"])
    return {**view, "big_blind": manifest["format"]["big_blind"], "warning": None}


# tournaments

def tournament_manifests() -> dict[str, dict]:
    return {path.parents[1].name: json.loads(path.read_text()) for path in sorted(EXPERIMENTS.glob("*/results/manifest.json"))}


def list_tournaments() -> list[dict]:
    """Every block of every experiment. Format 2 is read from disk; format 1 needs the engine it ran on."""
    current = engine_digest()
    out = []
    for experiment, manifest in tournament_manifests().items():
        version = manifest.get("format_version", 1)
        for key, block in manifest["blocks"].items():
            pair, seed = key.split("/seed=")
            out.append({
                "experiment": experiment, "pair": pair, "seed": int(seed), "tables": block["tables"],
                "hands": block["hands"], "format_version": version,
                "replayable": version >= 2 or block.get("engine_sha256") == current,
            })
    return out


def registry_name(experiment: str, manifest: dict, name: str) -> str:
    """Format-2 manifests record it; for format 1, the phase folder's final or a builtin of that name."""
    recorded = manifest["bots"].get(name, {}).get("registry_name")
    if recorded:
        return recorded
    return f"{experiment}/{name}" if f"{experiment}/{name}" in registry.REGISTRY else f"builtin/{name}"


@lru_cache(maxsize=2)
def regenerated_block(experiment: str, pair: str, seed: int) -> tuple[dict, list[int]]:
    """A format-1 block replayed with the tournament's own regenerate_block and turned into format-2 tables.
    Returns the frames and the hands (as (table, hand)) whose net_chips differ from the stored rows."""
    manifest = tournament_manifests()[experiment]
    block = manifest["blocks"][f"{pair}/seed={seed}"]
    if block.get("engine_sha256") != engine_digest():
        raise ValueError(f"block was run on engine {block.get('engine_sha256')}, the current engine is "
                         f"{engine_digest()}: its hands cannot be regenerated with this code")
    names = pair.split("__")
    bots = [registry.get_bot(registry_name(experiment, manifest, name))() for name in names]
    _, records, num_recorded = regenerate_block(*bots, seed, block["tables"], block["hands"], manifest["format"]["max_players"])
    frames = records_to_frames(records, num_recorded, names, {"pair": pair, "seed": seed})
    stored = pl.read_parquet(EXPERIMENTS / experiment / "results" / "seat_hands" / pair / f"seed_{seed}.parquet")
    joined = frames["seat_hands"].join(stored, on=["table", "hand", "seat"], how="full", suffix="_stored")
    mismatched = (joined.filter(pl.col("net_chips").ne_missing(pl.col("net_chips_stored")))
                  .select(pl.coalesce("table", "table_stored"), pl.coalesce("hand", "hand_stored")).unique().rows())
    return frames, mismatched


def tournament_table(experiment: str, pair: str, seed: int, table: int) -> dict:
    manifest = tournament_manifests()[experiment]
    max_players = manifest["format"]["max_players"]
    warning, mismatched = None, []
    if manifest.get("format_version", 1) >= 2:
        results = EXPERIMENTS / experiment / "results"
        frames = {name: pl.read_parquet(results / name / pair / f"seed_{seed}.parquet") for name in TABLES}
    else:
        frames, block_mismatched = regenerated_block(experiment, pair, seed)
        mismatched = sorted(hand for t, hand in block_mismatched if t == table)
        if mismatched:
            warning = (f"{len(mismatched)} regenerated hands on this table do not match the stored results "
                       f"(first: {[h + 1 for h in mismatched[:5]]}). This usually means a platform or float "
                       "difference from the machine that ran the block. Treat those hands as approximate.")
    frames = {name: frame.filter(pl.col("table") == table) for name, frame in frames.items()}
    view = frames_to_view(frames["seat_hands"], frames["hands"], frames["actions"], max_players)
    return {**view, "big_blind": manifest["format"]["big_blind"], "warning": warning, "mismatched_hands": mismatched}
