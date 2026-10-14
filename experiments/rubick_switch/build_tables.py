"""Build bots/rubick/tables.json: per (seat, hole-card class), which bot realised the most EV.

    uv run python experiments/rubick_switch/build_tables.py

Source: experiments/phase_3_dota/results seat_hands, seeds 0 and 1, with RandomBot removed both as a
candidate and as an opponent. A cell is (seats_from_button, hand_class): 2 × 169 cells, about 950
hands per bot per cell. EV is mean net chips per hand in big blinds.

Also prints a split-half check: choose on one seed, score on the other. The in-sample value of
argmax is inflated by selection (winner's curse); the out-of-sample value is what to expect.
"""
import json
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "experiments/phase_3_dota/results"
OUT = REPO / "bots/rubick/tables.json"
BASE = "windrunner_glm"        # the strict table's default: best single bot in phase 3
STRICT_Z = 2.0
RANKS = "23456789TJQKA"
KEYS = ["pair", "seed", "table", "hand"]

seat_hands = pl.read_parquet(SOURCE / "seat_hands/**/*.parquet", hive_partitioning=False)
opponent = seat_hands.select(*KEYS, "seat", pl.col("bot").alias("opponent")).with_columns(
    (1 - pl.col("seat")).cast(seat_hands["seat"].dtype).alias("seat"))
seat_hands = (seat_hands.join(opponent, on=[*KEYS, "seat"]).filter(pl.col("bot") != "random", pl.col("opponent") != "random")
              .with_columns(bb=pl.col("net_chips") / 2, pos=pl.col("seats_from_button")))


def cells(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.group_by("bot", "pos", "hand_class").agg(n=pl.len(), ev=pl.col("bb").mean(),
                                                          se=pl.col("bb").std() / pl.len().sqrt())


def choose(c: pl.DataFrame, strict: bool) -> pl.DataFrame:
    best = c.sort("ev", descending=True).group_by("pos", "hand_class").first()
    if not strict:
        return best.select("pos", "hand_class", "bot")
    base = c.filter(pl.col("bot") == BASE).select("pos", "hand_class", ev_base="ev", se_base="se")
    better = (pl.col("ev") - pl.col("ev_base")) > STRICT_Z * (pl.col("se") ** 2 + pl.col("se_base") ** 2).sqrt()
    return best.join(base, on=["pos", "hand_class"]).select(
        "pos", "hand_class", bot=pl.when(better).then(pl.col("bot")).otherwise(pl.lit(BASE)))


def value(choice: pl.DataFrame, c: pl.DataFrame) -> float:
    """bb/100 if every hand were played by its cell's chosen bot, cells weighted by how often they are dealt."""
    weight = c.group_by("pos", "hand_class").agg(w=pl.col("n").sum())
    v = choice.join(c, on=["pos", "hand_class", "bot"]).join(weight, on=["pos", "hand_class"])
    return 100 * float((v["ev"] * v["w"]).sum() / v["w"].sum())


by_seed = {seed: cells(seat_hands.filter(pl.col("seed") == seed)) for seed in (0, 1)}
for train, test in ((0, 1), (1, 0)):
    single = value(by_seed[test].filter(pl.col("bot") == BASE).select("pos", "hand_class", "bot"), by_seed[test])
    for strict in (False, True):
        chosen = choose(by_seed[train], strict)
        print(f"choose on seed {train}, score on seed {test}: {'strict' if strict else 'argmax'} "
              f"in-sample {value(chosen, by_seed[train]):+.1f}, out-of-sample {value(chosen, by_seed[test]):+.1f} "
              f"bb/100 ({BASE} alone {single:+.1f})")

full = cells(seat_hands)
bots = sorted(full["bot"].unique().to_list())
tables = {"bots": bots, "source": "experiments/phase_3_dota/results seeds 0-1, random excluded", "base": BASE,
          "strict_z": STRICT_Z}
for label, strict in (("argmax", False), ("strict", True)):
    chosen = choose(full, strict)
    table = [[[[0, 0] for _ in range(13)] for _ in range(13)] for _ in range(2)]   # [pos][hi][lo][suited]
    for pos, hand_class, bot in chosen.iter_rows():
        hi, lo = RANKS.index(hand_class[0]), RANKS.index(hand_class[1])
        for suited in ((0, 1) if len(hand_class) == 2 else ((1,) if hand_class[2] == "s" else (0,))):
            table[pos][hi][lo][suited] = bots.index(bot)
    tables[label] = table
    tables[f"{label}_in_sample_bb100"] = round(value(chosen, full), 1)
    print(label, "cells per bot:", dict(chosen.group_by("bot").len().sort("len", descending=True).rows()))
OUT.write_text(json.dumps(tables) + "\n")
print(f"wrote {OUT}")
