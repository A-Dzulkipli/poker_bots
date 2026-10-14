"""Scratch diagnostic: play a candidate vs one opponent and bucket per-hand outcomes.
Usage: uv run python debug.py v2:Bot windrunner_glm --tables 16 --hands 250 --seed 0
"""
import sys
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

import bench
from poker_bots.poker_game import Action, make_poker_env


def main():
    candidate, opponent = sys.argv[1], sys.argv[2]
    tables, hands, seed = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    env = make_poker_env((bench.load(candidate), bench.load(opponent)), max_players=2, small_blind=1, big_blind=2)
    keys = jax.random.split(jax.random.PRNGKey(seed), tables)
    seated = jnp.ones((tables, 2), dtype=jnp.bool_)
    seat_bot = jnp.where((jnp.arange(tables) % 2 == 0)[:, None], jnp.array([0, 1]), jnp.array([1, 0])).astype(jnp.int32)
    states = jax.vmap(env.init, in_axes=(0, 0, None, None, 0))(keys, seated, None, 0, seat_bot)
    _, records, n = jax.device_get(jax.jit(jax.vmap(lambda s: env.run_hands(s, hands)))(states))
    h = records.history
    bb = env.big_blind
    totals = defaultdict(float)
    counts = defaultdict(int)
    for t in range(tables):
        for i in range(n[t]):
            s = int(np.argmax(records.seat_bot[t, i] == 0))  # my seat's view
            net = float(records.ending_stacks[t, i, s] - records.starting_stacks[t, i, s]) / bb
            hh = h.is_valid[t, i]
            my_actions = hh & (h.player_id[t, i] == s)
            i_folded = bool(records.seated[t, i, s]) and not bool(records.in_hand[t, i, s])
            opp_folded = bool(records.seated[t, i, 1 - s]) and not bool(records.in_hand[t, i, 1 - s])
            if i_folded:
                fold_round = int(np.max(np.where(my_actions & (h.action_type[t, i] == Action.FOLD), h.round[t, i], -1)))
                key = f"i_fold_street{fold_round}"
            elif opp_folded:
                key = "opp_folded"
            elif bool(records.went_to_showdown[t, i]):
                key = "showdown_win" if net > 0 else ("showdown_loss" if net < 0 else "showdown_tie")
            else:
                key = "other"
            totals[key] += net
            counts[key] += 1
            counts["total"] += 1
            totals["total"] += net
            my_raises = int(np.sum(my_actions & (h.action_type[t, i] == Action.RAISE) & (h.round[t, i] > 0)))
            counts["hands_with_postflop_raise"] += my_raises > 0
            totals["my_postflop_raise_count"] += my_raises
            my_calls = int(np.sum(my_actions & (h.action_type[t, i] == Action.CHECK_CALL) & (h.round[t, i] > 0)))
            totals["my_postflop_calls"] += my_calls
    print(f"hands {counts['total']}, net {totals['total'] / max(counts['total'],1):+.2f} bb/hand "
          f"({100 * totals['total'] / max(counts['total'],1):+.0f} bb/100)")
    for k in sorted(counts):
        if k in ("total",):
            continue
        c = counts[k]
        t = totals[k] if k in totals else 0
        extra = f"  avg {t / max(c,1):+.2f} bb" if c and not k.startswith("count") else ""
        print(f"  {k:28s} {c:6d} ({100 * c / max(counts['total'],1):5.1f}%){extra}")


if __name__ == "__main__":
    main()
