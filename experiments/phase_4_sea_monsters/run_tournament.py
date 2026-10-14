"""Phase 4 (sea monsters) heads-up round robin with full hand histories.

    uv run python experiments/phase_4_sea_monsters/verify_finals.py                 # first: finals match their sources
    uv run python experiments/phase_4_sea_monsters/run_tournament.py                # seeds 0 1
    uv run python experiments/phase_4_sea_monsters/run_tournament.py --seeds 2 3    # add more hands
    uv run python experiments/phase_4_sea_monsters/run_tournament.py --summarise    # rebuild summaries only

Roster: the phase-4 finals (`bots/phase_4_sea_monsters/<lineage>_final.py`), the phase-3, phase-2
and phase-1 finals, the aggressor probe, and RandomBot. Storage format 2, documented in
experiments/README.md. Each table is one 250-hand session, so stateful bots learn within a table.

After the summaries it writes `results/weighted.parquet`: each bot's phase-4 weighted score over
the twelve-bot benchmark field (the same strength ranks and weights as the phase-4 bench.py).
"""
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent)]

from bots.registry import get_bot, list_bots  # noqa: E402
from tournament import main  # noqa: E402

FINALS = (list_bots("phase_4_sea_monsters", finals_only=True) + list_bots("phase_3_dota", finals_only=True)
          + list_bots("phase_2_suneater", finals_only=True) + list_bots("probes", finals_only=True)
          + list_bots("phase_1_remedy", finals_only=True))
ROSTER = {name.split("/")[-1]: (name, get_bot(name)) for name in FINALS + ["builtin/random"]}
# Strength ranks of the benchmark field, weakest first, as in bots/phase_4_sea_monsters/*/bench.py
STRENGTH_RANK = {name: rank for rank, name in enumerate([
    "random", "jack_joyce_gemini", "jesse_faden_glm", "max_payne_gpt", "alan_wake_claude", "bassander_lin_glm",
    "aggressor", "valka_onderra_claude", "hadrian_marlowe_gpt", "meepo_claude", "invoker_gpt", "windrunner_glm",
], start=1)}


def weighted_scores(results: Path) -> pl.DataFrame:
    """A win against rank r counts r / mean_rank, a loss (N + 1 - r) / mean_rank; mean over the field opponents played."""
    n, mean_rank = len(STRENGTH_RANK), (len(STRENGTH_RANK) + 1) / 2
    ranks = pl.DataFrame({"opponent": list(STRENGTH_RANK), "rank": list(STRENGTH_RANK.values())})
    pairs = pl.read_parquet(results / "pairs.parquet").join(ranks, on="opponent")
    weight = pl.when(pl.col("bb_per_100") > 0).then(pl.col("rank")).otherwise(n + 1 - pl.col("rank")) / mean_rank
    return (pairs.group_by("bot").agg((weight * pl.col("bb_per_100")).mean().alias("weighted_score"),
                                      pl.len().alias("field_opponents"))
            .sort("weighted_score", descending=True))


if __name__ == "__main__":
    main(HERE / "results", ROSTER, max_players=2)
    scores = weighted_scores(HERE / "results")
    scores.write_parquet(HERE / "results" / "weighted.parquet")
    with pl.Config(tbl_rows=-1):
        print(scores)
