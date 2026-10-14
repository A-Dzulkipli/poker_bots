"""Measure the bot's actual likelihood weights on constructed public lines."""
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp

# Only the package inside this bot directory is loaded explicitly.
import bench
version = sys.argv[1]
bot = importlib.import_module(f"joshua_gpt.{version}").Bot()
button_blinds = [(1, 4, 1, 0), (0, 5, 2, 0)]
big_blind_blinds = [(0, 4, 1, 0), (1, 5, 2, 0)]
button_open = button_blinds + [(1, 2, 5, 0)]
big_blind_call = big_blind_blinds + [(0, 2, 5, 0), (1, 1, 5, 0)]
flop_bet = button_open + [(0, 1, 5, 0), (0, 1, 0, 1), (1, 2, 7, 1)]
turn_bet = flop_bet + [(0, 1, 7, 1), (0, 1, 0, 2), (1, 2, 16, 2)]
river_bet = turn_bet + [(0, 1, 16, 2), (0, 1, 0, 3), (1, 2, 35, 3)]
scenarios = [
    ("Button opens", 1, 0, button_open),
    ("Button limps", 1, 0, button_blinds + [(1, 1, 2, 0)]),
    ("Big blind calls an open", 0, 0, big_blind_call),
    ("Big blind 3-bets", 0, 0, big_blind_blinds + [(0, 2, 5, 0), (1, 2, 18, 0)]),
    ("Button c-bets flop after opening", 1, 3, flop_bet),
    ("Big blind check-raises flop", 0, 3, big_blind_call + [(1, 1, 0, 1), (0, 2, 7, 1), (1, 2, 22, 1)]),
    ("Opponent bets turn after flop bet/call", 1, 4, turn_bet),
    ("Opponent bets all three streets", 1, 5, river_bet),
    ("Small turn re-raise", 1, 4, turn_bet + [(0, 2, 32, 2), (1, 2, 48, 2)]),
]
example_board = jnp.array([11, 20, 2, 44, 26], dtype=jnp.int32)
output = []
for label, button, board_count, records in scenarios:
    padded = records + [(-1, 0, 0, 0)] * (100 - len(records))
    history = SimpleNamespace(
        player_id=jnp.array([r[0] for r in padded]),
        action_type=jnp.array([r[1] for r in padded]),
        amount=jnp.array([r[2] for r in padded]),
        round=jnp.array([r[3] for r in padded]),
        is_valid=jnp.arange(100) < len(records),
    )
    obs = SimpleNamespace(history=history, button=jnp.array(button), bb_seat=jnp.array(1-button), big_blind=jnp.array(2))
    masses = []
    effective_samples = []
    pair_masses = []
    suited_masses = []
    broadway_masses = []
    for seed in range(10):
        known = jnp.zeros(52, dtype=bool).at[example_board[:board_count]].set(True)
        random_order = jax.random.uniform(jax.random.PRNGKey(seed), (bot.samples, 52)) + 2.0 * known
        draws = jnp.argsort(random_order, axis=1)[:, :7]
        boards = jnp.where(jnp.arange(5) < board_count, example_board, draws[:, :5])
        holdings = draws[:, 5:7]
        weights = bot.range_weights(obs, holdings, boards)
        assert bool(jnp.all(jnp.isfinite(weights) & (weights > 0) & (weights <= 1)))
        effective_samples.append(float(jnp.sum(weights) ** 2 / jnp.sum(weights ** 2)))
        ranks = holdings % 13
        masses.append(float(jnp.mean(weights)))
        pair_masses.append(float(jnp.mean(weights * (ranks[:, 0] == ranks[:, 1]))))
        suited_masses.append(float(jnp.mean(weights * (holdings[:, 0] // 13 == holdings[:, 1] // 13))))
        broadway_masses.append(float(jnp.mean(weights * jnp.all(ranks >= 8, axis=1))))
    mass = sum(masses) / len(masses)
    legal_share = (52-board_count) * (51-board_count) / (52*51)
    output.append({"line": label, "weighted_starting_share": mass * legal_share,
                   "legal_holding_mean_weight": mass, "effective_samples": sum(effective_samples) / len(effective_samples), "pair_posterior_share": sum(pair_masses) / sum(masses),
                   "suited_posterior_share": sum(suited_masses) / sum(masses),
                   "two_broadway_posterior_share": sum(broadway_masses) / sum(masses)})
Path(f"{version}_ranges.json").write_text(json.dumps(output, indent=2))
print(json.dumps(output, indent=2))
