"""Phase 2 (suneater) heads-up round robin with full hand histories.

    uv run python experiments/phase_2_suneater/run_tournament.py                 # seeds 0 1
    uv run python experiments/phase_2_suneater/run_tournament.py --seeds 2 3     # add more hands
    uv run python experiments/phase_2_suneater/run_tournament.py --summarise     # rebuild summaries only

Roster: the phase-2 finals (`bots/phase_2_suneater/<lineage>_final.py`, once they exist), the
phase-1 finals, and RandomBot. Storage format 2, documented in experiments/README.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent)]

from bots.registry import get_bot, list_bots  # noqa: E402
from tournament import main  # noqa: E402

FINALS = list_bots("phase_2_suneater", finals_only=True) + list_bots("phase_1_remedy", finals_only=True)
ROSTER = {name.split("/")[-1]: (name, get_bot(name)) for name in FINALS + ["builtin/random"]}

if __name__ == "__main__":
    main(HERE / "results", ROSTER, max_players=2)
