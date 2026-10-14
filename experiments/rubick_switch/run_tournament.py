"""Rubick (per-hand-class bot switching) against every existing bot, heads-up, full hand histories.

    uv run python experiments/rubick_switch/run_tournament.py --seeds 2 3     # fresh seeds: the tables used 0 and 1
    uv run python experiments/rubick_switch/run_tournament.py --summarise

Plays rubick/rubick (argmax table) and rubick/rubick_strict against the phase-3 roster
(phase-3, phase-2 and phase-1 finals, the aggressor probe and RandomBot). Pairs between two
non-Rubick bots are not replayed; take those from experiments/phase_3_dota. Tables are built by
experiments/rubick_switch/build_tables.py from experiments/phase_3_dota/results seeds 0 and 1.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent)]

from bots.registry import get_bot, list_bots  # noqa: E402
from tournament import main  # noqa: E402

RUBICK = list_bots("rubick", finals_only=True)
FIELD = (list_bots("phase_3_dota", finals_only=True) + list_bots("phase_2_suneater", finals_only=True)
         + list_bots("probes", finals_only=True) + list_bots("phase_1_remedy", finals_only=True) + ["builtin/random"])
ROSTER = {name.split("/")[-1]: (name, get_bot(name)) for name in RUBICK + FIELD}

if __name__ == "__main__":
    main(HERE / "results", ROSTER, max_players=2, focus=tuple(name.split("/")[-1] for name in RUBICK))
