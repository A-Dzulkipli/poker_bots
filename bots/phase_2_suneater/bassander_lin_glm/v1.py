"""v1: simple heads-up baseline. Monte Carlo equity vs a random hand decides everything:
open every button hand, defend wide from the big blind, call when equity beats pot odds,
bet or raise for value with strong equity. Equity is shrunk toward 0.5 for every raise
the opponent has made this street, since a raiser's range is stronger than random."""
import jax
import jax.numpy as jnp

from poker_bots.poker_game import Action, ActionData, PlayerTableState, Round

from .equity import equity, pot_odds, to_call, raise_to, check_call, fold_or_check, choose


class Bot:
    def __init__(self,
                 num_samples: int = 96,               # Monte Carlo equity samples per decision (cap 256)
                 button_open_bb: float = 2.5,         # button preflop open raise-to size (big blinds)
                 limp_raise_equity: float = 0.55,     # preflop equity to raise a button limp (probability)
                 three_bet_equity: float = 0.60,      # preflop equity to re-raise (probability)
                 value_bet_equity: float = 0.64,      # postflop equity to bet or raise (probability)
                 bet_pot_fraction: float = 0.65,      # bet size as a fraction of the pot after calling
                 call_margin: float = 0.02,           # equity edge over pot odds required to call (probability)
                 raise_shrink: float = 0.85,          # equity shrunk this much per opponent raise this street
                 preflop_reraise_multiple: float = 3.0):  # preflop re-raise-to as a multiple of current bet
        self.num_samples = num_samples
        self.button_open_bb = button_open_bb
        self.limp_raise_equity = limp_raise_equity
        self.three_bet_equity = three_bet_equity
        self.value_bet_equity = value_bet_equity
        self.bet_pot_fraction = bet_pot_fraction
        self.call_margin = call_margin
        self.raise_shrink = raise_shrink
        self.preflop_reraise_multiple = preflop_reraise_multiple

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        eq = equity(obs, key, self.num_samples)
        street_actions = obs.history.is_valid & (obs.history.round == obs.current_round)
        opp_raises = jnp.sum(street_actions & (obs.history.action_type == Action.RAISE) & (obs.history.player_id != 0))
        # equity vs a random hand overstates our chances against someone who has raised
        eq_faced = 0.5 + (eq - 0.5) * jnp.power(self.raise_shrink, opp_raises)

        # --- shared postflop / facing-bet logic: call on pot odds, raise for value ---
        pot_after_call = obs.pot + to_call(obs)
        bet_total = obs.current_bet + self.bet_pot_fraction * pot_after_call
        preflop = obs.current_round == Round.PRE_FLOP
        raise_total = jnp.where(preflop, self.preflop_reraise_multiple * obs.current_bet, bet_total)
        raise_equity = jnp.where(preflop, self.three_bet_equity, self.value_bet_equity)
        wants_raise = obs.can_raise & (eq >= raise_equity)
        wants_call = eq_faced >= pot_odds(obs) + self.call_margin
        action = choose(wants_call, check_call(), fold_or_check(obs))
        action = choose(wants_raise, raise_to(obs, raise_total), action)

        # --- preflop as button (small blind): open every hand for a min-raise-plus ---
        blinds_only = preflop & (obs.current_bet <= obs.big_blind)
        on_button = obs.button == 0
        action = choose(blinds_only & on_button, raise_to(obs, self.button_open_bb * obs.big_blind), action)

        # --- preflop as big blind vs a limp: raise the top of the range, check the rest ---
        limped_pot = blinds_only & ~on_button
        punish_limp = choose(eq >= self.limp_raise_equity, raise_to(obs, 3.5 * obs.big_blind), check_call())
        action = choose(limped_pot, punish_limp, action)

        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "bassander_lin_glm",
            "version": 1,
            "parent": None,
            "lineage": "phase_2_suneater/bassander_lin_glm",
            "model": "GLM-5.3",
            "prompt_version": "botgen-v2",
            "phase": "phase_2_suneater",
            "engine": "02abc49+wt sha256:295477eff7637056",
            "params": {
                "num_samples": self.num_samples,
                "button_open_bb": self.button_open_bb,
                "limp_raise_equity": self.limp_raise_equity,
                "three_bet_equity": self.three_bet_equity,
                "value_bet_equity": self.value_bet_equity,
                "bet_pot_fraction": self.bet_pot_fraction,
                "call_margin": self.call_margin,
                "raise_shrink": self.raise_shrink,
                "preflop_reraise_multiple": self.preflop_reraise_multiple,
            },
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GLM"},
            ],
        }
