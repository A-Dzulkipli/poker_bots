"""meepo_claude v9: v5 with more frequent bluffs when checked to.

Change from v5: bluff_frequency 0.35 -> 0.55. A two-thirds-pot bluff needs 40% folds, and
value-only opponents fold far more often than that.
v5 change from v2: heads-up button opens are wide, so a preflop raise removes only
preflop_raise_discount of the range (not the postflop 0.25); the big blind defends more.
v2 change from v1: with weak hands, bet as a bluff at a fixed frequency when betting is free,
so opponents who fold to bets lose pots they would have checked down.
v1 summary:

Equity is measured against a random hand, then discounted for every bet or raise the
opponent has made this hand (their range is stronger than random). Preflop the button
opens wide and the big blind defends by pot odds; postflop the bot bets for value,
checks medium hands, and calls or folds on pot odds against the discounted equity.
"""
import jax
import jax.numpy as jnp
from poker_bots.poker_game import Action, PlayerTableState, Round

from .equity import equity, to_call, pot_odds, fold_or_check, check_call, raise_to, choose


class Bot:
    def __init__(self, num_samples=192, open_equity=0.42, preflop_raise_equity=0.62,
                 limp_raise_equity=0.55, value_bet_equity=0.64, value_raise_equity=0.78,
                 street_raise_discount=0.25, earlier_raise_discount=0.12, preflop_raise_discount=0.10, call_margin=0.02,
                 open_size_bb=2.5, reraise_multiple=3.0, bet_pot_fraction=0.66,
                 bluff_max_equity=0.35, bluff_frequency=0.55):
        self.num_samples = num_samples                    # Monte Carlo samples per decision (cap 256)
        self.open_equity = open_equity                    # button opens with equity vs random >= this
        self.preflop_raise_equity = preflop_raise_equity  # 3-bet / 4-bet with discounted equity >= this
        self.limp_raise_equity = limp_raise_equity        # big blind raises a limp with equity >= this
        self.value_bet_equity = value_bet_equity          # postflop bet when checked to, discounted equity
        self.value_raise_equity = value_raise_equity      # postflop raise facing a bet, discounted equity
        self.street_raise_discount = street_raise_discount    # weakest range fraction removed per raise this street
        self.earlier_raise_discount = earlier_raise_discount  # same, per raise on an earlier street
        self.preflop_raise_discount = preflop_raise_discount  # same, per preflop raise (wide heads-up opens)
        self.call_margin = call_margin                    # equity probability required above pot odds
        self.open_size_bb = open_size_bb                  # button open, raise-to in big blinds
        self.reraise_multiple = reraise_multiple          # preflop re-raise = this times the current bet
        self.bet_pot_fraction = bet_pot_fraction          # postflop bet/raise size as a pot fraction
        self.bluff_max_equity = bluff_max_equity          # hands below this equity may bluff (they cannot win by checking)
        self.bluff_frequency = bluff_frequency            # probability of bluffing such a hand when checked to

    def init_state(self, key):
        return None

    # ---- shared: equity against the opponent's likely range ----
    def range_equity(self, obs, key):
        h = obs.history
        their_raises = h.is_valid & (h.player_id == 1) & (h.action_type == Action.RAISE)
        preflop_raises = jnp.sum(their_raises & (h.round == Round.PRE_FLOP))
        postflop = their_raises & (h.round > Round.PRE_FLOP)
        this_street = jnp.sum(postflop & (h.round == obs.current_round))
        earlier = jnp.sum(postflop & (h.round < obs.current_round))
        # each raise removes the weakest part of their range; what remains is what we must beat
        remaining = (1 - self.street_raise_discount) ** this_street * (1 - self.earlier_raise_discount) ** earlier
        remaining = remaining * (1 - self.preflop_raise_discount) ** preflop_raises
        raw = equity(obs, key, self.num_samples)
        return jnp.clip((raw - (1 - remaining)) / remaining, 0.0, 1.0), raw

    def bet(self, obs, pot_fraction):
        pot_after_call = obs.pot + to_call(obs)
        return raise_to(obs, obs.current_bet + pot_fraction * pot_after_call)

    def call_or_fold(self, obs, win_prob):
        # call when equity beats the price; never fold when checking is free
        return choose(win_prob >= pot_odds(obs) + self.call_margin, check_call(), fold_or_check(obs))

    # ---- preflop as button, first to act ----
    def button_open(self, obs, raw):
        return choose(raw >= self.open_equity, raise_to(obs, self.open_size_bb * obs.big_blind), fold_or_check(obs))

    # ---- preflop facing a limp or a raise ----
    def preflop_response(self, obs, win_prob):
        facing_limp = obs.current_bet <= obs.big_blind
        threshold = jnp.where(facing_limp, self.limp_raise_equity, self.preflop_raise_equity)
        reraise = raise_to(obs, self.reraise_multiple * jnp.maximum(obs.current_bet, obs.big_blind))
        return choose(win_prob >= threshold, reraise, self.call_or_fold(obs, win_prob))

    # ---- postflop, checked to us or first to act ----
    def postflop_unopened(self, obs, win_prob, key):
        value = win_prob >= self.value_bet_equity
        # bluff with hands too weak to win at showdown; medium hands check and keep their equity
        bluff = (win_prob < self.bluff_max_equity) & (jax.random.uniform(key) < self.bluff_frequency)
        return choose(value | bluff, self.bet(obs, self.bet_pot_fraction), check_call())

    # ---- postflop facing a bet ----
    def postflop_facing_bet(self, obs, win_prob):
        raise_ok = win_prob >= self.value_raise_equity
        return choose(raise_ok, self.bet(obs, self.bet_pot_fraction), self.call_or_fold(obs, win_prob))

    def act(self, obs: PlayerTableState, state, key):
        equity_key, mix_key = jax.random.split(key)
        win_prob, raw = self.range_equity(obs, equity_key)
        preflop = obs.current_round == Round.PRE_FLOP
        button_first_in = preflop & (obs.button == 0) & (obs.current_bet <= obs.big_blind) & (to_call(obs) > 0)
        preflop_action = choose(button_first_in, self.button_open(obs, raw), self.preflop_response(obs, win_prob))
        facing_bet = to_call(obs) > 0
        postflop_action = choose(facing_bet, self.postflop_facing_bet(obs, win_prob),
                                 self.postflop_unopened(obs, win_prob, mix_key))
        return choose(preflop, preflop_action, postflop_action), state

    def metadata(self) -> dict:
        return {
            "bot_name": "meepo_claude", "version": 9, "parent": 5,
            "lineage": "phase_3_dota/meepo_claude", "model": "Claude Opus 5.5 (claude-opus-5-5)",
            "prompt_version": "botgen-v3", "phase": "phase_3_dota",
            "engine": "02abc49+wt sha256:295477eff7637056",
            "params": dict(vars(self)),
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "Claude desktop (Code)"}
            ],
        }
