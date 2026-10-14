"""Check that each phase-2 `<lineage>_final.py` plays exactly like the version file it was copied from.

    uv run python experiments/phase_2_suneater/verify_finals.py

Plays each final and its source version against the same opponent on the same seed and compares
every stored row. Exits non-zero on any difference.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent)]

from bots.registry import get_bot, list_bots  # noqa: E402
from hand_records import records_to_frames  # noqa: E402
from tournament import regenerate_block  # noqa: E402

DEPLOYED = {"valka_onderra_claude": "v5", "hadrian_marlowe_gpt": "v6", "bassander_lin_glm": "v8"}
OPPONENT = "phase_1_remedy/alan_wake_claude"

failed = False
assert sorted(list_bots("phase_2_suneater", finals_only=True)) == sorted(f"phase_2_suneater/{n}" for n in DEPLOYED)
for lineage, version in DEPLOYED.items():
    frames = []
    for name in (f"phase_2_suneater/{lineage}", f"phase_2_suneater/{lineage}/{version}"):
        _, records, num_recorded = regenerate_block(get_bot(name)(), get_bot(OPPONENT)(), 0, 8, 100, 2)
        frames.append(records_to_frames(records, num_recorded, ["candidate", "opponent"], {"pair": "check", "seed": 0}))
    same = all(frames[0][table].equals(frames[1][table]) for table in frames[0])
    failed |= not same
    print(f"{lineage}: final vs {version}.py {'identical' if same else 'DIFFERENT'} over {frames[0]['hands'].height} hands")
sys.exit(1 if failed else 0)
