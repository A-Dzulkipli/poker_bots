"""Actual ranges per line for every bot, measured from the phase-6 hand histories (blocks involving a phase-6 bot).

    uv run python experiments/phase_6_trails/actual_ranges.py experiments/phase_6_trails/results experiments/phase_6_trails/results/actual_ranges

A line's share of starting hands is the product of the actor's conditional frequencies at each of its decisions
along the line; the opponent's replies depend only on public history, so the product is the share of dealt hands
that take the line. Composition columns describe the hands that completed the line (postflop: made-hand category
on that street). Writes OUT.parquet and OUT_classes.parquet (preflop take-rate per hand class, phase-6 bots).
"""
import sys
import numpy as np
import polars as pl

R, OUT = sys.argv[1], sys.argv[2]
PH6 = {"estelle_claude", "joshua_gpt", "scherezard_glm"}
sh = pl.scan_parquet(f"{R}/seat_hands/**/*.parquet", hive_partitioning=False)
ac = pl.scan_parquet(f"{R}/actions/**/*.parquet", hive_partitioning=False)
hd = pl.scan_parquet(f"{R}/hands/**/*.parquet", hive_partitioning=False)
# only blocks involving a phase-6 bot
is6 = pl.any_horizontal([pl.col("pair").str.contains(b) for b in PH6])
sh, ac, hd = sh.filter(is6), ac.filter(is6), hd.filter(is6)
K = ["pair", "seed", "table", "hand"]

seats = sh.select(K + ["seat", "bot", "seats_from_button", "hole_cards", "hand_class"]).collect()
role = seats.select(K + ["seat", pl.when(pl.col("seats_from_button") == 0).then(pl.lit("B")).otherwise(pl.lit("E")).alias("role")])
a = (ac.filter(~pl.col("type").cast(pl.Utf8).is_in(["small_blind", "big_blind"])).collect()
     .join(role, on=K + ["seat"])
     .with_columns(pl.when(pl.col("type").cast(pl.Utf8) == "fold").then(pl.lit("f"))
                   .when(pl.col("type").cast(pl.Utf8) == "raise").then(pl.lit("r"))
                   .when(pl.col("to_call") == 0).then(pl.lit("k")).otherwise(pl.lit("c")).alias("k"))
     .sort(K + ["idx"]))
lines = (a.with_columns((pl.col("role") + pl.col("k")).alias("tok"))
         .group_by(K + ["street"]).agg(pl.col("tok").str.join(""))
         .pivot(on="street", index=K, values="tok"))
for s in ["0", "1", "2", "3"]:
    if s not in lines.columns:
        lines = lines.with_columns(pl.lit(None, pl.Utf8).alias(s))
lines = lines.rename({"0": "pre", "1": "flop", "2": "turn", "3": "river"}).with_columns(pl.col(["pre", "flop", "turn", "river"]).fill_null(""))
boards = hd.select(K + ["board"]).collect()
df = (seats.join(lines, on=K, how="left").join(boards, on=K, how="left")
      .with_columns(pl.col(["pre", "flop", "turn", "river"]).fill_null(""),
                    pl.when(pl.col("seats_from_button") == 0).then(pl.lit("B")).otherwise(pl.lit("E")).alias("role")))

def sw(col, p): return pl.col(col).str.starts_with(p)
def eq(col, p): return pl.col(col) == p
# each line: (actor role, list of (spot condition, take condition)) ; share = product of P(take | spot)
LINES = [
    ("button opens", "B", [(pl.lit(True), sw("pre", "Br"))]),
    ("button limps", "B", [(pl.lit(True), sw("pre", "Bc"))]),
    ("big blind calls an open", "E", [(sw("pre", "Br"), sw("pre", "BrEc"))]),
    ("big blind 3-bets", "E", [(sw("pre", "Br"), sw("pre", "BrEr"))]),
    ("button c-bets flop after opening", "B", [(pl.lit(True), sw("pre", "Br")), (eq("pre", "BrEc") & sw("flop", "Ek"), sw("flop", "EkBr"))]),
    ("big blind check-raises flop", "E", [(sw("pre", "Br"), sw("pre", "BrEc")), (eq("pre", "BrEc") & (pl.col("flop") != ""), sw("flop", "Ek")),
                                          (eq("pre", "BrEc") & sw("flop", "EkBr"), sw("flop", "EkBrEr"))]),
    ("button bets turn after flop bet-call", "B", [(pl.lit(True), sw("pre", "Br")), (eq("pre", "BrEc") & sw("flop", "Ek"), sw("flop", "EkBr")),
                                                   (eq("pre", "BrEc") & eq("flop", "EkBrEc") & sw("turn", "Ek"), sw("turn", "EkBr"))]),
    ("button bets all three streets", "B", [(pl.lit(True), sw("pre", "Br")), (eq("pre", "BrEc") & sw("flop", "Ek"), sw("flop", "EkBr")),
                                            (eq("pre", "BrEc") & eq("flop", "EkBrEc") & sw("turn", "Ek"), sw("turn", "EkBr")),
                                            (eq("pre", "BrEc") & eq("flop", "EkBrEc") & eq("turn", "EkBrEc") & sw("river", "Ek"), sw("river", "EkBr"))]),
]
RANKS = "23456789TJQKA"
def made_category(hole, board, n):
    from poker_bots.hand_eval import evaluate_hands
    def ids(s): return [RANKS.index(s[i]) + 13 * "dchs".index(s[i + 1]) for i in range(0, len(s), 2)]
    cards = np.array([ids(h) + ids(b)[:n] for h, b in zip(hole, board)], dtype=np.int32)
    return np.asarray(evaluate_hands(cards)) // 13 ** 5
def describe(sub, street_cards):
    hc = sub["hand_class"]
    pairs = (hc.str.len_chars() == 2).mean()
    suited = hc.str.ends_with("s").mean()
    broad = hc.map_elements(lambda c: all(RANKS.index(x) >= 8 for x in c[:2]), return_dtype=pl.Boolean).mean()
    ace = hc.str.contains("A").mean()
    junk = hc.map_elements(lambda c: len(c) == 3 and c[2] == "o" and RANKS.index(c[0]) < 10 and RANKS.index(c[0]) - RANKS.index(c[1]) > 2, return_dtype=pl.Boolean).mean()
    out = {"pairs": pairs, "suited": suited, "two_broadway": broad, "has_ace": ace, "offsuit_junk": junk}
    if street_cards:
        cat = made_category(sub["hole_cards"].to_list(), sub["board"].to_list(), street_cards)
        out |= {"air": float((cat == 0).mean()), "one_pair": float((cat == 1).mean()), "two_pair_plus": float((cat >= 2).mean())}
    return out
rows = []
for bot in sorted(df["bot"].unique()):
    d = df.filter(pl.col("bot") == bot)
    for name, r, steps in LINES:
        dr = d.filter(pl.col("role") == r)
        share, n_last = 1.0, 0
        for spot, take in steps:
            sp = dr.filter(spot)
            n_last = len(sp)
            share = share * sp.filter(take).height / len(sp) if len(sp) else None
            if share is None:
                break
        took = dr.filter(steps[-1][0] & steps[-1][1])
        street_cards = {"button c-bets flop after opening": 3, "big blind check-raises flop": 3,
                        "button bets turn after flop bet-call": 4, "button bets all three streets": 5}.get(name, 0)
        desc = describe(took, street_cards) if took.height >= 20 else {}
        rows.append({"bot": bot, "line": name, "share": share, "spot_hands": n_last, "line_hands": took.height, **desc})
res = pl.DataFrame(rows)
res.write_parquet(f"{OUT}.parquet")
# class take-rates for preflop lines (phase-6 bots)
cls = []
for bot in sorted(PH6):
    d = df.filter(pl.col("bot") == bot)
    for name, r, spot, take in [("open", "B", pl.lit(True), sw("pre", "Br")), ("bb_3bet", "E", sw("pre", "Br"), sw("pre", "BrEr")),
                                ("bb_continue", "E", sw("pre", "Br"), sw("pre", "BrEc") | sw("pre", "BrEr"))]:
        g = d.filter((pl.col("role") == r) & spot).group_by("hand_class").agg(take.mean().alias("rate"), pl.len().alias("n"))
        cls.append(g.with_columns(pl.lit(bot).alias("bot"), pl.lit(name).alias("line")))
pl.concat(cls).write_parquet(f"{OUT}_classes.parquet")
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=3):
    print(res)
classes = pl.read_parquet(f"{OUT}_classes.parquet")
for (bot, line), g in classes.sort("rate", descending=True).group_by(["bot", "line"], maintain_order=True):
    kept = g.filter(pl.col("rate") >= 0.9)["hand_class"].to_list()
    edge = g.filter((pl.col("rate") >= 0.3) & (pl.col("rate") < 0.7))["hand_class"].to_list()
    print(f"{bot} {line}: always (>=90%) {len(kept)} classes; boundary (30-70%): {' '.join(edge[:40])}")
