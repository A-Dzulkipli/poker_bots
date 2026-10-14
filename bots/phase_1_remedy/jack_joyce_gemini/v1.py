import jax
import jax.numpy as jnp
from poker_bots.poker_game import PlayerTableState
from .equity import check_call, choose, equity, fold_or_check, pot_odds, raise_to

class Bot:
    """Baseline bot. Calls when equity > pot odds. Raises a fixed fraction when equity is high."""
    
    def __init__(self, raise_equity: float = 0.65, call_margin: float = 0.03, bet_fraction: float = 0.75):
        self.raise_equity = raise_equity
        self.call_margin = call_margin
        self.bet_fraction = bet_fraction

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        eq = equity(obs, key, 128)
        bet = obs.current_bet + self.bet_fraction * (obs.pot + obs.current_bet - obs.contributed[0])
        action = choose(eq >= pot_odds(obs) + self.call_margin, check_call(), fold_or_check(obs))
        action = choose(obs.can_raise & (eq >= self.raise_equity), raise_to(obs, bet), action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "jack_joyce_gemini",
            "version": 1,
            "parent": None,
            "lineage": "phase_1_remedy/jack_joyce_gemini",
            "model": "Gemini-2.5-Pro",
            "prompt_version": "botgen-v1",
            "engine_commit": "02abc49",
            "params": {"raise_equity": self.raise_equity, "call_margin": self.call_margin, "bet_fraction": self.bet_fraction},
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "Gemini"}
            ],
        }