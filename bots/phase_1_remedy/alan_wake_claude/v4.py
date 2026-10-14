import jax
import jax.numpy as jnp

from poker_bots.poker_game import Action, PlayerTableState, Round
from .equity import check_call, choose, equity, fold_or_check, num_live_opponents, pot_odds, raise_to


class Bot:
    """Equity-driven bot: call when equity beats pot odds, raise when equity is a multiple
    of our fair share against the players left. Equity assumes random opponent hands, so
    it is shrunk for every raise an opponent has made on this street. On the button and in
    the cutoff the raise bar is lower, to take more pots from the players acting after us."""

    def __init__(
        self,
        num_samples: int = 128,               # Monte Carlo samples per decision (cap 256)
        raise_strength: float = 1.5,          # raise when equity >= this times our fair share
        raise_discount: float = 0.15,         # fraction of equity removed per opponent raise this street
        late_position_bonus: float = 0.3,     # raise-bar cut (x fair share) on button/cutoff
        call_margin: float = 0.03,            # equity probability required above pot odds to call
        preflop_raise_multiple: float = 3.0,  # preflop raise-to = this times the current bet
        bet_pot_fraction: float = 0.66,       # postflop raise = call + this fraction of the pot
        shove_commit_fraction: float = 0.40,  # shove if the raise uses this share of our total chips
    ):
        self.num_samples = num_samples
        self.raise_strength = raise_strength
        self.raise_discount = raise_discount
        self.late_position_bonus = late_position_bonus
        self.call_margin = call_margin
        self.preflop_raise_multiple = preflop_raise_multiple
        self.bet_pot_fraction = bet_pot_fraction
        self.shove_commit_fraction = shove_commit_fraction

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        history = obs.history
        this_street = history.is_valid & (history.round == obs.current_round)
        opponent_raises = jnp.sum(this_street & (history.action_type == Action.RAISE) & (history.player_id != 0))
        # random-hand equity overstates our chances against someone who raised
        win_prob = equity(obs, key, self.num_samples) * jnp.power(1 - self.raise_discount, opponent_raises)
        is_preflop = obs.current_round == Round.PRE_FLOP
        pot_after_call = obs.pot + obs.current_bet - obs.contributed[0]
        # preflop: open to 3bb, re-raise to 3x the raise
        preflop_size = self.preflop_raise_multiple * obs.current_bet
        # postflop: match the bet, then add a fraction of the pot
        postflop_size = obs.current_bet + self.bet_pot_fraction * pot_after_call
        bet_size = jnp.where(is_preflop, preflop_size, postflop_size)
        # once a raise would commit a big share of our chips, just go all-in
        all_in_total = obs.contributed[0] + obs.stacks[0]
        commits_most = bet_size >= self.shove_commit_fraction * all_in_total
        bet_size = jnp.where(commits_most, all_in_total, bet_size)
        wants_call = win_prob >= pot_odds(obs) + self.call_margin
        # 1.0 = an average hand against this many opponents, 2.0 = twice our fair share
        strength = win_prob * (num_live_opponents(obs) + 1)
        # button is seat 0 (us) or seat 1 (we are the cutoff, one seat to its right)
        in_late_position = (obs.button == 0) | (obs.button == 1)
        raise_bar = self.raise_strength - self.late_position_bonus * in_late_position
        wants_raise = obs.can_raise & (strength >= raise_bar)
        action = choose(wants_call, check_call(), fold_or_check(obs))
        action = choose(wants_raise, raise_to(obs, bet_size), action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "alan_wake_claude",
            "version": 4,
            "parent": 3,
            "lineage": "phase_1_remedy/alan_wake_claude",
            "model": "Claude Opus 5.5 (claude-opus-5-5)",
            "prompt_version": "botgen-v1",
            "engine_commit": "02abc49",
            "params": {
                "num_samples": self.num_samples,
                "raise_strength": self.raise_strength,
                "raise_discount": self.raise_discount,
                "late_position_bonus": self.late_position_bonus,
                "call_margin": self.call_margin,
                "preflop_raise_multiple": self.preflop_raise_multiple,
                "bet_pot_fraction": self.bet_pot_fraction,
                "shove_commit_fraction": self.shove_commit_fraction,
            },
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "Claude desktop (Code)"},
            ],
        }
