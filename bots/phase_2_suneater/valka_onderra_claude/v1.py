"""v1: sound heads-up baseline built on equity against a random hand."""
import jax
import jax.numpy as jnp

from poker_bots.poker_game import Action, PlayerTableState, Round

from .equity import check_call, choose, equity, fold_or_check, pot_odds, raise_to, to_call


class Bot:
    """Open most buttons, defend the big blind by pot odds, bet postflop for value.
    Equity is measured against a random hand and shrunk for every raise the opponent
    has made this hand, because a raiser holds a stronger-than-random hand."""

    def __init__(
        self,
        num_samples: int = 128,             # Monte Carlo samples per decision (cap 256)
        open_equity: float = 0.40,          # button opens with equity >= this (folds the worst ~15%)
        preflop_raise_equity: float = 0.60, # 3-bet / isolate / 4-bet with adjusted equity >= this
        postflop_bet_equity: float = 0.66,  # bet or raise postflop with adjusted equity >= this
        call_margin: float = 0.02,          # equity probability required above pot odds to call
        raise_discount: float = 0.12,       # fraction of equity removed per opponent raise this hand
        open_size_bb: float = 2.5,          # button open size, in big blinds
        reraise_multiple: float = 3.0,      # preflop raise-to = this times the current bet
        bet_pot_fraction: float = 0.66,     # postflop bet = call + this fraction of the pot
    ):
        self.num_samples = num_samples
        self.open_equity = open_equity
        self.preflop_raise_equity = preflop_raise_equity
        self.postflop_bet_equity = postflop_bet_equity
        self.call_margin = call_margin
        self.raise_discount = raise_discount
        self.open_size_bb = open_size_bb
        self.reraise_multiple = reraise_multiple
        self.bet_pot_fraction = bet_pot_fraction

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        history = obs.history
        # every opponent raise this hand means a stronger-than-random hand: shrink our equity
        opponent_raises = jnp.sum(history.is_valid & (history.player_id == 1) & (history.action_type == Action.RAISE))
        win_prob = equity(obs, key, self.num_samples) * (1 - self.raise_discount) ** opponent_raises
        # default: call when equity beats the price, otherwise check or fold
        call_or_fold = choose(win_prob >= pot_odds(obs) + self.call_margin, check_call(), fold_or_check(obs))

        # --- preflop ---
        is_preflop = obs.current_round == Round.PRE_FLOP
        unopened = obs.current_bet <= obs.big_blind
        # button first in: folding only costs the small blind, so open wide
        button_first_in = unopened & (to_call(obs) > 0)
        button_open = raise_to(obs, self.open_size_bb * obs.big_blind)
        open_or_fold = choose(win_prob >= self.open_equity, button_open, fold_or_check(obs))
        # big blind after a limp, or anyone facing a raise: re-raise strong hands, else price-based call
        preflop_reraise = raise_to(obs, self.reraise_multiple * obs.current_bet)
        preflop_action = choose(win_prob >= self.preflop_raise_equity, preflop_reraise, call_or_fold)
        preflop_action = choose(button_first_in, open_or_fold, preflop_action)

        # --- postflop ---
        pot_after_call = obs.pot + to_call(obs)
        value_bet = raise_to(obs, obs.current_bet + self.bet_pot_fraction * pot_after_call)
        postflop_action = choose(win_prob >= self.postflop_bet_equity, value_bet, call_or_fold)

        action = choose(is_preflop, preflop_action, postflop_action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "valka_onderra_claude",
            "version": 1,
            "parent": None,
            "lineage": "phase_2_suneater/valka_onderra_claude",
            "model": "Claude Opus 5.5 (claude-opus-5-5)",
            "prompt_version": "botgen-v2",
            "phase": "phase_2_suneater",
            "engine": "02abc49+wt sha256:295477eff7637056",
            "params": {
                "num_samples": self.num_samples,
                "open_equity": self.open_equity,
                "preflop_raise_equity": self.preflop_raise_equity,
                "postflop_bet_equity": self.postflop_bet_equity,
                "call_margin": self.call_margin,
                "raise_discount": self.raise_discount,
                "open_size_bb": self.open_size_bb,
                "reraise_multiple": self.reraise_multiple,
                "bet_pot_fraction": self.bet_pot_fraction,
            },
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "Claude desktop (Code)"},
            ],
        }
