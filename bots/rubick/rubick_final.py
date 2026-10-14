"""Rubick: plays each hand as the existing bot with the best realised EV for its hole cards and seat.

The lookup table (`tables.json`) maps (seat, hole-card class) to one of the bots below. It was built
from the phase-3 round robin (seeds 0 and 1, all opponents except RandomBot), in bb/hand per
(seats_from_button, hand_class) cell. Two tables:
    argmax  the bot with the highest mean EV in the cell
    strict  the cell's best bot only if it beats windrunner_glm by more than 2 standard errors,
            otherwise windrunner_glm
Every bot in the corpus is stateless and none models its opponent, so a hand played by one bot's
policy has the same expected value whatever policies were used on other hands. Mixing per hand is
therefore sound, and the table's expected value is the frequency-weighted sum of its cells.

All sub-bots are evaluated each decision (lax.switch is a select under vmap), so this is about as
slow as all eleven bots together.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from poker_bots.poker_game import PlayerTableState

from bots.registry import get_bot

TABLES = json.loads((Path(__file__).parent / "tables.json").read_text())
REGISTRY_NAMES = {
    "aggressor": "probes/aggressor",
    "alan_wake_claude": "phase_1_remedy/alan_wake_claude", "jack_joyce_gemini": "phase_1_remedy/jack_joyce_gemini",
    "jesse_faden_glm": "phase_1_remedy/jesse_faden_glm", "max_payne_gpt": "phase_1_remedy/max_payne_gpt",
    "bassander_lin_glm": "phase_2_suneater/bassander_lin_glm", "hadrian_marlowe_gpt": "phase_2_suneater/hadrian_marlowe_gpt",
    "valka_onderra_claude": "phase_2_suneater/valka_onderra_claude",
    "invoker_gpt": "phase_3_dota/invoker_gpt", "meepo_claude": "phase_3_dota/meepo_claude",
    "windrunner_glm": "phase_3_dota/windrunner_glm",
}


class Bot:
    def __init__(self, table: str = "argmax"):
        self.table_name = table
        full = jnp.asarray(TABLES[table], dtype=jnp.int32)          # [seat 0 button / 1 big blind, hi, lo, suited]
        used = sorted(set(int(i) for i in full.ravel()))            # only evaluate bots the table actually uses
        self.names = [TABLES["bots"][i] for i in used]
        self.bots = [get_bot(REGISTRY_NAMES[name])() for name in self.names]
        self.table = jnp.searchsorted(jnp.asarray(used), full)      # remap to positions in self.bots

    def init_state(self, key: jax.Array) -> None:
        return None

    def choose(self, obs: PlayerTableState) -> jax.Array:
        rank, suit = obs.hole_cards % 13, obs.hole_cards // 13
        hi, lo = jnp.max(rank), jnp.min(rank)
        return self.table[jnp.where(obs.button == 0, 0, 1), hi, lo, (suit[0] == suit[1]).astype(jnp.int32)]

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        branches = [lambda o, k, b=b: b.act(o, None, k)[0] for b in self.bots]
        return jax.lax.switch(self.choose(obs), branches, obs, key), state

    def metadata(self) -> dict:
        return {"bot_name": f"rubick_{self.table_name}", "kind": "bot", "role": "meta",
                "description": "plays each hand as the bot with the best realised EV for its hole-card class and seat",
                "params": {"table": self.table_name, "bots": self.names,
                           "source": "experiments/phase_3_dota/results seeds 0-1, excluding random"}}
