"""Scratch: measure the range the bot assigns on standard lines, from the code itself.

Builds synthetic PlayerTableState/history situations and prints, for each line, the weighted
share of all 1326 starting hands the bot's range keeps (a fully-kept hand has weight 1.0) and
the hands carrying the most weight.  Usage: uv run python scratch_ranges.py v1
"""
import sys
import importlib
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from poker_bots.poker_game import Action, ActionRecord

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
mod = importlib.import_module(f"scherezard_glm.{sys.argv[1] if len(sys.argv) > 1 else 'v1'}")
A = 100  # max_actions
FLOP = [11, 18, 28]       # Ks 7h 4c
TURN = FLOP + [35]        # Jc
RIVER = TURN + [47]       # 9d


def hist(records):
    """records: (player_id, action_type, amount, round) tuples, blinds prepended by caller."""
    h = {f: jnp.full(A, -1 if f == "player_id" else 0, jnp.int32) for f in
         ("player_id", "action_type", "amount", "requested_type", "requested", "round", "is_all_in")}
    valid = jnp.zeros(A, bool)
    for i, (p, t, amt, rnd) in enumerate(records):
        h["player_id"], h["action_type"], h["amount"], h["round"] = (
            h["player_id"].at[i].set(p), h["action_type"].at[i].set(t), h["amount"].at[i].set(amt), h["round"].at[i].set(rnd))
        h["is_valid"] = valid = valid.at[i].set(True)
    return {k: (v.astype(bool) if k == "is_valid" else v) for k, v in h.items()}


def obs_for(button, records, board, round_):
    class O:  # minimal stand-in carrying exactly the fields the range code reads
        pass
    o = O()
    o.button, o.big_blind, o.current_round, o.board = button, 2, round_, jnp.array(board + [-1] * (5 - len(board)), jnp.int32)
    o.hole_cards = jnp.array([24, 43], jnp.int32)  # 9h 4d: a middling hero hand that blocks little
    o.history = ActionRecord(**hist(records))
    return o


def range_share(button, records, board, round_):
    obs = obs_for(button, records, board, round_)
    weights = mod.narrow_postflop(obs, mod.preflop_range(obs)).block_until_ready()
    w = np.triu(np.array(weights).reshape(52, 52), 1)
    names = lambda c: f"{'23456789TJQKA'[c % 13]}{'shcd'[c // 13]}"
    top = sorted(((w[a, b], names(a) + names(b)) for a in range(52) for b in range(a + 1, 52)), reverse=True)[:8]
    return w.sum() / 1326, ", ".join(f"{n} {v:.2f}" for v, n in top)


SB, BB = Action.SMALL_BLIND, Action.BIG_BLIND
R, C, F = Action.RAISE, Action.CHECK_CALL, Action.FOLD
LINES = [
    ("button opens to 3bb", 1, [(1, SB, 1, 0), (0, BB, 2, 0), (1, R, 6, 0)], [], 0),
    ("button limps", 1, [(1, SB, 1, 0), (0, BB, 2, 0), (1, C, 2, 0)], [], 0),
    ("bb calls a 3bb open", 0, [(0, SB, 1, 0), (1, BB, 2, 0), (0, R, 6, 0), (1, C, 6, 0)], [], 0),
    ("bb 3-bets to 9bb", 0, [(0, SB, 1, 0), (1, BB, 2, 0), (0, R, 6, 0), (1, R, 18, 0)], [], 0),
    ("button c-bets flop after opening (Ks7h4c)", 1,
     [(1, SB, 1, 0), (0, BB, 2, 0), (1, R, 6, 0), (0, C, 6, 0), (0, C, 0, 1), (1, R, 6, 1)], FLOP, 1),
    ("bb check-raises flop (Ks7h4c)", 0,
     [(0, SB, 1, 0), (1, BB, 2, 0), (0, R, 6, 0), (1, C, 6, 0), (1, C, 0, 1), (0, R, 6, 1), (1, R, 18, 1)], FLOP, 1),
    ("bb bets turn after flop bet-call (Ks7h4c Jc)", 0,
     [(0, SB, 1, 0), (1, BB, 2, 0), (0, R, 6, 0), (1, C, 6, 0), (1, R, 6, 1), (0, C, 6, 1), (1, R, 8, 2)], TURN, 2),
    ("bb bets all three streets (Ks7h4c Jc 9d)", 0,
     [(0, SB, 1, 0), (1, BB, 2, 0), (0, R, 6, 0), (1, C, 6, 0), (1, R, 6, 1), (0, C, 6, 1),
      (1, R, 8, 2), (0, C, 8, 2), (1, R, 12, 3)], RIVER, 3),
]

if __name__ == "__main__":
    for name, button, records, board, rnd in LINES:
        share, top = range_share(button, records, board, rnd)
        print(f"{name:>46}: weighted share {share:5.2f} | top: {top}")
