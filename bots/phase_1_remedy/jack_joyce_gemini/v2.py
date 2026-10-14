import jax
import jax.numpy as jnp
from poker_bots.poker_game import PlayerTableState, Round
from .equity import check_call, choose, equity, fold_or_check, pot_odds, raise_to

class Bot:
    """Street-aware bot. Requires higher equity to raise preflop since realization is harder, lower threshold postflop."""
    
    def __init__(self, raise_eq_pre: float = 0.70, raise_eq_post: float = 0.60, bet_fraction: float = 0.75):
        self.raise_eq_pre = raise_eq_pre
        self.raise_eq_post = raise_eq_post
        self.bet_fraction = bet_fraction

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        eq = equity(obs, key, 128)
        
        # Adjust equity requirements depending on if it's preflop or postflop
        is_preflop = obs.current_round == Round.PRE_FLOP
        raise_thresh = jnp.where(is_preflop, self.raise_eq_pre, self.raise_eq_post)
        
        bet = obs.current_bet + self.bet_fraction * (obs.pot + obs.current_bet - obs.contributed[0])
        action = choose(eq >= pot_odds(obs), check_call(), fold_or_check(obs))
        action = choose(obs.can_raise & (eq >= raise_thresh), raise_to(obs, bet), action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "jack_joyce_gemini",
            "version": 2,
            "parent": 1,
            "lineage": "phase_1_remedy/jack_joyce_gemini",
            "model": "Gemini-2.5-Pro",
            "prompt_version": "botgen-v1",
            "engine_commit": "02abc49",
            "params": {"raise_eq_pre": self.raise_eq_pre, "raise_eq_post": self.raise_eq_post, "bet_fraction": self.bet_fraction},
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "Gemini"}
            ],
        }