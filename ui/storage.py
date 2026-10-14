"""Play-session files in the project's hand-history format (storage format 2, experiments/README.md).

ui/sessions/<session>/
    seat_hands.parquet, hands.parquet, actions.parquet   built by experiments/hand_records.records_to_frames,
                                                         with `session` as the key column
    manifest.json                                        format_version, config, engine hashes, bot sources, seed

A session's HandRecords are kept in memory and the files rewritten every `flush_every` hands and on
close, so a crash loses at most that many hands. After close the files are not touched again. A session
closed before its first hand finishes writes nothing.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np

import poker_bots

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
sys.path.insert(0, str(EXPERIMENTS))   # hand_records and tournament import each other by bare module name

from hand_records import FORMAT_VERSION, records_to_frames  # noqa: E402
from tournament import engine_digest, repo_relative, sha256, source_file  # noqa: E402

SESSIONS = Path(__file__).resolve().parent / "sessions"
TABLES = ("seat_hands", "hands", "actions")


def short_name(registry_name: str) -> str:
    """Roster-style name: the lineage for a final or builtin ("alan_wake_claude", "random"), lineage/vN for a version."""
    parts = registry_name.split("/")
    return "/".join(parts[1:]) if len(parts) > 2 else parts[-1]


class SessionWriter:
    """`bot_types` are the env's bots in order, as (name in the tables, registry name or None, class, kind)."""

    def __init__(self, config: dict, seed: int, bot_types: list[tuple[str, str | None, type, str]],
                 flush_every: int = 10, root: Path | None = None, player_name: str | None = None):
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.dir = Path(root or SESSIONS) / self.session_id
        self.bot_names = [name for name, _, _, _ in bot_types]
        self.bot_kinds = [kind for _, _, _, kind in bot_types]
        self.flush_every = flush_every
        self.records = []
        self.closed = False
        engine_dir = Path(poker_bots.__file__).parent
        self.manifest = {
            "format_version": FORMAT_VERSION,
            "session": self.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "player_name": player_name,   # typed by the player when sitting down; not verified
            "format": config,
            "engine": {
                "files_sha256": {p.name: sha256(p) for p in sorted(engine_dir.glob("*.py"))},
                "jax_version": jax.__version__,
                "backend": jax.default_backend(),
            },
            "engine_sha256": engine_digest(),
            "bots": {
                name: {"registry_name": registry_name, "kind": kind, "metadata": cls().metadata(),
                       "source": repo_relative(source_file(cls)), "source_sha256": sha256(source_file(cls))}
                for name, registry_name, cls, kind in bot_types
            },
        }

    @property
    def hands(self) -> int:
        return len(self.records)

    def add_hand(self, record) -> None:
        """`record` is one finished hand's HandRecord, on the host."""
        self.records.append(record)
        if self.hands % self.flush_every == 0:
            self.flush()

    def frames(self) -> dict:
        stacked = jax.tree.map(lambda *xs: np.stack(xs)[None], *self.records)   # (table=1, hand, ...)
        return records_to_frames(stacked, np.array([self.hands]), self.bot_names, {"session": self.session_id},
                                 self.bot_kinds)

    def flush(self) -> None:
        if not self.records:   # nothing to keep: a session with no finished hand leaves no folder
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        for name, frame in self.frames().items():
            path = self.dir / f"{name}.parquet"
            tmp = path.with_suffix(".parquet.tmp")
            frame.write_parquet(tmp, compression="zstd")
            os.replace(tmp, path)
        self.manifest["hands"] = self.hands
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2, default=str))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.manifest["closed_at"] = datetime.now(timezone.utc).isoformat()
        self.flush()
