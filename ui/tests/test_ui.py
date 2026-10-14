import json

import numpy as np
import polars as pl
import pytest

from poker_bots.poker_game import RandomBot
from ui.game import PlaySession
from ui.replay import list_sessions, list_tournaments, load_session, tournament_table
from ui.storage import TABLES

from hand_records import records_to_frames  # noqa: E402  (on sys.path via ui.storage)
from tournament import regenerate_block  # noqa: E402


def play(session: PlaySession, hands: int, raise_every: int = 3) -> list[dict]:
    """Human calls, and raises to the minimum every few decisions."""
    finished, decisions = [], 0
    while len(finished) < hands:
        what = session.advance()
        if what == "human":
            controls = session.snapshot()["controls"]
            assert controls["raise_min"] <= controls["raise_max"]
            decisions += 1
            if controls["can_raise"] and decisions % raise_every == 0:
                session.act("raise", controls["raise_min"])
            else:
                session.act("check_call")
        elif what == "hand_over":
            finished.append(session.last_hand)
        assert what != "finished"
    return finished


@pytest.mark.parametrize("seat_bots", [["builtin/random"], ["builtin/random", "builtin/random", "builtin/random"]])
def test_session_round_trip(tmp_path, seat_bots):
    session = PlaySession(seat_bots, seed=7, flush_every=4, sessions_dir=tmp_path)
    hands = play(session, 9)
    folder = tmp_path / session.session_id
    assert (folder / "seat_hands.parquet").exists()   # flushed before close
    session.close()

    # same columns as a tournament's format-2 tables, with `session` as the key
    _, records, num_recorded = regenerate_block(RandomBot(), RandomBot(), 0, 2, 3, 2)
    tournament = records_to_frames(records, num_recorded, ["a", "b"], {"pair": "a__b", "seed": 0})
    frames = {name: pl.read_parquet(folder / f"{name}.parquet") for name in TABLES}
    for name in TABLES:
        assert frames[name].columns == ["session"] + tournament[name].columns[2:]
    seat_hands, actions = frames["seat_hands"], frames["actions"]
    assert seat_hands.height == 9 * len(session.seat_names)
    assert seat_hands.group_by("hand").agg(pl.col("net_chips").sum())["net_chips"].to_list() == [0] * 9
    assert set(seat_hands.filter(pl.col("seat") == 0)["bot"]) == {"human"}
    assert set(actions.filter(pl.col("seat") == 0)["by"]) == {"human"}
    assert set(actions.filter(pl.col("seat") != 0)["by"]) == {"bot"}
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["hands"] == 9 and "closed_at" in manifest and manifest["format_version"] == 2

    loaded = load_session(session.session_id, tmp_path)["hands"]
    for original, replayed in zip(hands, loaded):
        for key in ("hole_cards", "board", "board_dealt", "payouts", "ending_stacks", "button", "sb_seat", "bb_seat"):
            assert replayed[key] == original[key], key
        assert [{k: a[k] for k in ("seat", "street", "type", "value_applied")} for a in replayed["actions"]] == \
            [{k: a[k] for k in ("seat", "street", "type", "value_applied")} for a in original["actions"]]
    assert list_sessions(tmp_path)[0]["hands"] == 9


def test_session_closed_before_a_hand_writes_nothing(tmp_path):
    session = PlaySession(["builtin/random"], seed=1, sessions_dir=tmp_path)
    snap = session.snapshot()
    assert snap["seats"][0]["cards"] and snap["seats"][1]["cards"] is None
    assert session.snapshot(show_bot_cards=True)["seats"][1]["cards"]
    session.close()
    assert not any(tmp_path.iterdir())


def test_format_1_tournament_table_regenerates_to_stored_results():
    blocks = [b for b in list_tournaments() if b["replayable"] and b["format_version"] == 1]
    if not blocks:
        pytest.skip("no format-1 block was run on the current engine")
    block = blocks[0]
    result = tournament_table(block["experiment"], block["pair"], block["seed"], 1)
    assert result["warning"] is None
    assert len(result["hands"]) == block["hands"]
    assert all(np.sum(h["ending_stacks"]) == np.sum(h["starting_stacks"]) for h in result["hands"])
