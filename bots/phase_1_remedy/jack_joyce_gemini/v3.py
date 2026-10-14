import jax
import jax.numpy as jnp
from poker_bots.poker_game import PlayerTableState, Round
from .equity import check_call, choose, equity, fold_or_check, pot_odds, raise_to

class Bot:
    """Dynamic sizing and bluffing bot. Bets larger with massive equity and incorporates a small frequency of pure bluffs when unchecked."""
    
    def __init__(self, raise_eq: float = 0.60, massive_eq: float = 0.80, bluff_prob: float = 0.10):
        self.raise_eq = raise_eq
        self.massive_eq = massive_eq
        self.bluff_prob = bluff_prob

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        eq_key, bluff_key = jax.random.split(key)
        eq = equity(obs, eq_key, 128)
        
        # Sizing: Pot if massive equity, else half pot
        fraction = jnp.where(eq >= self.massive_eq, 1.0, 0.5)
        bet = obs.current_bet + fraction * (obs.pot + obs.current_bet - obs.contributed[0])
        
        # Bluffing: randomly bluff if no one has bet/raised yet
        is_bluff = (jax.random.uniform(bluff_key) < self.bluff_prob) & (obs.current_bet == 0)
        
        action = choose(eq >= pot_odds(obs), check_call(), fold_or_check(obs))
        should_raise = obs.can_raise & ((eq >= self.raise_eq) | is_bluff)
        action = choose(should_raise, raise_to(obs, bet), action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "jack_joyce_gemini",
            "version": 3,
            "parent": 2,
            "lineage": "phase_1_remedy/jack_joyce_gemini",
            "model": "Gemini-2.5-Pro",
            "prompt_version": "botgen-v1",
            "engine_commit": "02abc49",
            "params": {"raise_eq": self.raise_eq, "massive_eq": self.massive_eq, "bluff_prob": self.bluff_prob},
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "Gemini"}
            ],
        }