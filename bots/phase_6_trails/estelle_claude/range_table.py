"""Scratch: measure the share of starting hands each version's range keeps on constructed lines."""
import sys
import importlib
import jax
import jax.numpy as jnp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poker_bots.poker_game import ActionRecord, PlayerTableState, Action

R, C, SB, BB = Action.RAISE, Action.CHECK_CALL, Action.SMALL_BLIND, Action.BIG_BLIND
def card(s):
    return "23456789TJQKA".index(s[0]) + 13 * "cdhs".index(s[1])
BOARD = [card("Kh"), card("8d"), card("3c"), card("2s"), card("9h")]  # example board K83 rainbow, 2, 9

def obs_for(actions, hero_button, street, hero=("7c", "6d")):
    # actions: list of (player, type, amount, round); blinds added automatically.
    opp = 0 if not hero_button else 1
    sb_player = 0 if hero_button else 1
    recs = [(sb_player, SB, 1, 0), (1 - sb_player, BB, 2, 0)] + actions
    n = 100
    pad = lambda vals, fill: jnp.array(vals + [fill] * (n - len(vals)), dtype=jnp.int32)
    hist = ActionRecord(player_id=pad([r[0] for r in recs], -1), action_type=pad([r[1] for r in recs], 0),
                        amount=pad([r[2] for r in recs], 0), requested_type=pad([r[1] for r in recs], 0),
                        requested=pad([r[2] for r in recs], 0), is_all_in=jnp.zeros(n, bool),
                        round=pad([r[3] for r in recs], 0), is_valid=jnp.array([True] * len(recs) + [False] * (n - len(recs))))
    board = jnp.array([BOARD[i] if i < [0, 3, 4, 5][street] else -1 for i in range(5)], dtype=jnp.int32)
    z = lambda v: jnp.asarray(v, dtype=jnp.int32)
    return PlayerTableState(hole_cards=jnp.array([card(c) for c in hero], dtype=jnp.int32), board=board, history=hist,
                            stacks=z([180, 180]), seated=jnp.array([True, True]), in_hand=jnp.array([True, True]),
                            button=z(0 if hero_button else 1), sb_seat=z(0 if hero_button else 1), bb_seat=z(1 if hero_button else 0),
                            contributed=z([0, 0]), current_round=z(street), current_bet=z(0), min_raise=z(2), pot=z(20),
                            can_raise=jnp.array(True), raise_lower_bound=z(2), raise_upper_bound=z(180), big_blind=z(2), hand_count=z(0))

LINES = {  # opponent is seat 1
    "button opens": (False, 0, [(1, R, 6, 0)]),
    "button limps": (False, 0, [(1, C, 2, 0)]),
    "big blind calls an open": (True, 1, [(0, R, 6, 0), (1, C, 6, 0)]),
    "big blind 3-bets": (True, 0, [(0, R, 6, 0), (1, R, 18, 0)]),
    "button c-bets flop after opening": (False, 1, [(1, R, 6, 0), (0, C, 6, 0), (0, C, 0, 1), (1, R, 8, 1)]),
    "big blind check-raises flop": (True, 1, [(0, R, 6, 0), (1, C, 6, 0), (1, C, 0, 1), (0, R, 8, 1), (1, R, 24, 1)]),
    "bets turn after bet flop, called": (True, 2, [(0, R, 6, 0), (1, C, 6, 0), (1, R, 8, 1), (0, C, 8, 1), (1, R, 20, 2)]),
    "bets all three streets": (True, 3, [(0, R, 6, 0), (1, C, 6, 0), (1, R, 8, 1), (0, C, 8, 1), (1, R, 20, 2), (0, C, 20, 2), (1, R, 50, 3)]),
}

def main(version, seeds=8):
    bot = importlib.import_module(f"estelle_claude.{version}").Bot()
    for name, (hero_button, street, actions) in LINES.items():
        obs = obs_for(actions, hero_button, street)
        shares, examples = [], {}
        for s in range(seeds):
            cards, _ = bot.sample_opponent_holdings(obs, jax.random.PRNGKey(s))
            logw = bot.opponent_range_log_weights(obs, bot.strength_percentiles(obs, cards))
            w = jnp.exp(logw)
            shares.append(float(jnp.mean(w)))
            for c, wi in zip(cards.tolist(), w.tolist()):
                r = sorted([c[0] % 13, c[1] % 13], reverse=True)
                key = "23456789TJQKA"[r[0]] + "23456789TJQKA"[r[1]] + ("" if r[0] == r[1] else ("s" if c[0] // 13 == c[1] // 13 else "o"))
                examples.setdefault(key, []).append(wi)
        top = sorted(examples.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
        kept = [k for k, v in top if sum(v) / len(v) > 0.5]
        print(f"{name:36s} share {sum(shares)/len(shares):.3f}  kept(w>0.5, {len(kept)} classes seen): {' '.join(kept[:25])}")

if __name__ == "__main__":
    main(sys.argv[1])
