"""Check that each phase-5 `<lineage>_final.py` plays exactly like the version its README deploys.

    uv run python experiments/phase_5_cowboy_bebop/verify_finals.py

The deployed version is read from each lineage's `__init__.py` (`from .vN import Bot as DeployedBot`).
Plays each final and its source version against the same opponent on the same seed and compares
every stored row. Exits non-zero on any difference.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(HERE.parent)]

from bots.registry import get_bot, list_bots  # noqa: E402
from hand_records import records_to_frames  # noqa: E402
from tournament import regenerate_block  # noqa: E402

PHASE = "phase_5_cowboy_bebop"
OPPONENT = "probes/aggressor"

failed = False
finals = list_bots(PHASE, finals_only=True)
assert finals, f"no finals found in bots/{PHASE}"
for name in finals:
    lineage = name.split("/")[-1]
    init = (REPO / "bots" / PHASE / lineage / "__init__.py").read_text()
    version = re.search(r"from \.(v\d+) import Bot as DeployedBot", init).group(1)
    frames = []
    for bot in (name, f"{name}/{version}"):
        _, records, num_recorded = regenerate_block(get_bot(bot)(), get_bot(OPPONENT)(), 0, 8, 100, 2)
        frames.append(records_to_frames(records, num_recorded, ["candidate", "opponent"], {"pair": "check", "seed": 0}))
    same = all(frames[0][table].equals(frames[1][table]) for table in frames[0])
    failed |= not same
    print(f"{lineage}: final vs {version}.py {'identical' if same else 'DIFFERENT'} over {frames[0]['hands'].height} hands")
sys.exit(1 if failed else 0)
