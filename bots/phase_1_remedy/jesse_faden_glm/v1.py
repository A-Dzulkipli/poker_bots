import jax
import jax.numpy as jnp

from poker_bots.poker_game import PlayerTableState

from .equity import check_call, choose, equity, fold_or_check, pot_odds, raise_to, to_call


class Bot:
    """v1 baseline: call when Monte Carlo equity beats pot odds, raise a pot fraction with strong equity."""

    def __init__(self, call_margin: float = 0.03, raise_equity: float = 0.62, bet_fraction: float = 0.75, num_samples: int = 128):
        self.call_margin = call_margin      # equity edge over pot odds required to call (probability)
        self.raise_equity = raise_equity    # equity required to bet or raise (probability)
        self.bet_fraction = bet_fraction    # raise-to size as a fraction of the pot after calling
        self.num_samples = num_samples      # Monte Carlo equity samples per decision

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        eq = equity(obs, key, self.num_samples)
        bet_total = obs.current_bet + self.bet_fraction * (obs.pot + to_call(obs))
        call_or_fold = choose(eq >= pot_odds(obs) + self.call_margin, check_call(), fold_or_check(obs))
        action = choose(obs.can_raise & (eq >= self.raise_equity), raise_to(obs, bet_total), call_or_fold)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "jesse_faden_glm",
            "version": 1,
            "parent": None,
            "lineage": "phase_1_remedy/jesse_faden_glm",
            "model": "GLM-5.3",
            "prompt_version": "botgen-v1",
            "engine_commit": "02abc49",
            "params": {
                "call_margin": self.call_margin,
                "raise_equity": self.raise_equity,
                "bet_fraction": self.bet_fraction,
                "num_samples": self.num_samples,
            },
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GLM"},
            ],
        }
