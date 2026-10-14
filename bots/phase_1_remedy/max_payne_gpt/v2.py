"""Discount uniform-range equity when opponents raise on the current street."""
import jax
import jax.numpy as jnp
from poker_bots.poker_game import Action, PlayerTableState, Round
from .equity import check_call, choose, equity, fold_or_check, num_live_opponents, pot_odds, raise_to, to_call


class Bot:
    def __init__(self, num_samples=64, realization_fraction=0.85, call_margin_equity=0.04,
                 preflop_entry_multiplier=1.15, preflop_raise_multiplier=1.8,
                 value_raise_equity=0.62, bet_pot_fraction=0.75, opening_size_bb=2.5, equity_retained_per_raise=0.82):
        self.equity_retained_per_raise = equity_retained_per_raise
        self.num_samples = num_samples
        self.realization_fraction = realization_fraction
        self.call_margin_equity = call_margin_equity
        self.preflop_entry_multiplier = preflop_entry_multiplier
        self.preflop_raise_multiplier = preflop_raise_multiplier
        self.value_raise_equity = value_raise_equity
        self.bet_pot_fraction = bet_pot_fraction
        self.opening_size_bb = opening_size_bb
        if not 1 <= num_samples <= 256:
            raise ValueError("num_samples must be between 1 and 256")

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state, key: jax.Array):
        raw_equity = equity(obs, key, self.num_samples)
        fair_share = 1.0 / (num_live_opponents(obs) + 1)
        history = obs.history
        opponent_raises = jnp.sum(history.is_valid & (history.round == obs.current_round) & (history.player_id != 0) & (history.action_type == Action.RAISE))
        # Each opposing raise signals a stronger range than uniformly random cards.
        raw_equity = raw_equity * self.equity_retained_per_raise ** opponent_raises
        preflop = obs.current_round == Round.PRE_FLOP
        realization = jnp.where(obs.current_round == Round.RIVER, 1.0, self.realization_fraction)
        calling_equity = raw_equity * realization
        entry_equity = jnp.where(preflop, self.preflop_entry_multiplier * fair_share, 0.0)
        required_equity = jnp.maximum(pot_odds(obs) + self.call_margin_equity, entry_equity)
        action = choose(calling_equity >= required_equity, check_call(), fold_or_check(obs))
        preflop_raise_equity = jnp.minimum(self.value_raise_equity, self.preflop_raise_multiplier * fair_share)
        raise_equity = jnp.where(preflop, preflop_raise_equity, self.value_raise_equity)
        value_raise = raw_equity >= raise_equity
        pot_after_call = obs.pot + to_call(obs)
        raise_total = obs.current_bet + self.bet_pot_fraction * pot_after_call
        opening_total = self.opening_size_bb * obs.big_blind
        unopened = preflop & (obs.current_bet <= obs.big_blind)
        raise_total = jnp.where(unopened, jnp.maximum(raise_total, opening_total), raise_total)
        action = choose(obs.can_raise & value_raise, raise_to(obs, raise_total), action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "max_payne_gpt",
            "version": 2,
            "parent": 1,
            "lineage": "phase_1_remedy/max_payne_gpt",
            "model": "GPT-6 (Codex; exact serving variant unavailable)",
            "prompt_version": "botgen-v1",
            "engine_commit": "02abc49",
            "params": {
                "num_samples": self.num_samples,
                "equity_retained_per_raise": self.equity_retained_per_raise,
                "realization_fraction": self.realization_fraction,
                "call_margin_equity": self.call_margin_equity,
                "preflop_entry_multiplier": self.preflop_entry_multiplier,
                "preflop_raise_multiplier": self.preflop_raise_multiplier,
                "value_raise_equity": self.value_raise_equity,
                "bet_pot_fraction": self.bet_pot_fraction,
                "opening_size_bb": self.opening_size_bb,
            },
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GPT (Codex)"}
            ],
        }
