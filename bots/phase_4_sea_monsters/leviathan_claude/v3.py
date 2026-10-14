"""v3: v1 plus tracking of how often the opponent folds once we bet postflop.

Same strategy as v1 (v2's aggression tracking is dropped). Bluff rates scale with the observed
fold rate (more bluffs against a folder, fewer against a caller), and against an opponent that
folds less than usual the value-bet threshold drops so we bet thinner for value.
"""
import jax
import jax.numpy as jnp

from poker_bots.poker_game import Action, PlayerTableState, Round

from .equity import check_call, choose, equity, fold_or_check, pot_odds, raise_to, to_call

MODEL = "claude-opus-5-5 (Claude Opus 5.5)"


class Bot:
    def __init__(self,
                 num_samples=128,              # Monte Carlo equity samples per decision (cap 256)
                 open_size_bb=2.5,             # button open raise-to, big blinds
                 reraise_multiple=3.0,         # preflop re-raise-to as a multiple of the current bet
                 three_bet_equity=0.63,        # equity vs random hand to 3-bet an open (probability)
                 light_three_bet_rate=0.15,    # share of calling hands 3-bet as a bluff (probability)
                 limp_raise_equity=0.55,       # equity to raise a button limp (probability)
                 value_bet_equity=0.62,        # equity to bet when checked to (probability)
                 air_bluff_rate=0.33,          # flop bet rate with air when checked to (probability)
                 semi_bluff_rate=0.50,         # flop bet rate with medium hands and draws (probability)
                 air_equity_limit=0.45,        # below this equity a hand counts as air (probability)
                 bluff_street_decay=0.55,      # bluff rates multiply by this per street after the flop
                 bet_pot_fraction=0.65,        # postflop bet size, fraction of pot after calling
                 value_raise_equity=0.80,      # equity to raise a postflop bet (probability)
                 call_margin=0.02,             # equity edge over pot odds required to call (probability)
                 pressure_discount=0.58,       # equity multiplier per opponent bet this hand
                 earlier_street_weight=0.5,    # an opponent bet on an earlier street counts this much
                 realization_factor=0.90,      # share of equity realized when more betting can follow
                 typical_fold_rate=0.5,        # share of hands a typical opponent folds after we bet postflop
                 fold_prior_hands=10.0,        # pseudo-count hands behind the typical fold rate
                 min_bluff_scale=0.3,          # bluff-rate multiplier floor against a non-folder
                 max_bluff_scale=1.6,          # bluff-rate multiplier cap against a heavy folder
                 thin_value_shift=0.08):       # value-bet equity drop against an opponent that never folds
        self.num_samples = num_samples
        self.open_size_bb = open_size_bb
        self.reraise_multiple = reraise_multiple
        self.three_bet_equity = three_bet_equity
        self.light_three_bet_rate = light_three_bet_rate
        self.limp_raise_equity = limp_raise_equity
        self.value_bet_equity = value_bet_equity
        self.air_bluff_rate = air_bluff_rate
        self.semi_bluff_rate = semi_bluff_rate
        self.air_equity_limit = air_equity_limit
        self.bluff_street_decay = bluff_street_decay
        self.bet_pot_fraction = bet_pot_fraction
        self.value_raise_equity = value_raise_equity
        self.call_margin = call_margin
        self.pressure_discount = pressure_discount
        self.earlier_street_weight = earlier_street_weight
        self.realization_factor = realization_factor
        self.typical_fold_rate = typical_fold_rate
        self.fold_prior_hands = fold_prior_hands
        self.min_bluff_scale = min_bluff_scale
        self.max_bluff_scale = max_bluff_scale
        self.thin_value_shift = thin_value_shift

    def init_state(self, key):
        return {"hands_i_bet_postflop": jnp.zeros((), jnp.float32),
                "opponent_folds_after_my_bet": jnp.zeros((), jnp.float32)}

    # ---- tracking: in how many hands we bet postflop, and in how many of those the opponent folded ----
    def observe(self, record, state, key):
        history = record.history
        my_postflop_bets = history.is_valid & (history.player_id == 0) & (history.round >= Round.FLOP) & (history.action_type == Action.RAISE)
        i_bet_postflop = jnp.any(my_postflop_bets)
        opponent_folded = record.seated[1] & ~record.in_hand[1]
        return {"hands_i_bet_postflop": state["hands_i_bet_postflop"] + i_bet_postflop,
                "opponent_folds_after_my_bet": state["opponent_folds_after_my_bet"] + (i_bet_postflop & opponent_folded)}

    # ---- tracking: opponent fold rate after our postflop bets, relative to a typical opponent ----
    def fold_rate_ratio(self, state):
        prior_folds = self.typical_fold_rate * self.fold_prior_hands
        opponent_fold_rate = (state["opponent_folds_after_my_bet"] + prior_folds) / (
            state["hands_i_bet_postflop"] + self.fold_prior_hands)
        return opponent_fold_rate / self.typical_fold_rate

    # ---- shared: how much betting pressure the opponent has applied this hand ----
    def pressure_units(self, obs):
        history = obs.history
        opponent_raised = history.is_valid & (history.player_id == 1) & (history.action_type == Action.RAISE)
        i_raised = history.is_valid & (history.player_id == 0) & (history.action_type == Action.RAISE)
        this_street = jnp.sum(opponent_raised & (history.round == obs.current_round))
        earlier_streets = jnp.sum(opponent_raised & (history.round < obs.current_round))
        my_raises_this_street = jnp.sum(i_raised & (history.round == obs.current_round))
        # a button open is wide, so preflop only a raise over my own raise counts as pressure
        preflop_units = jnp.where(my_raises_this_street > 0, this_street, jnp.maximum(this_street - 1, 0))
        postflop_units = this_street + self.earlier_street_weight * earlier_streets
        return jnp.where(obs.current_round == Round.PRE_FLOP, preflop_units, postflop_units)

    # ---- shared: call if pressure-discounted, realizable equity beats the pot odds ----
    def wants_to_call(self, obs, win_prob, discount):
        pressured_equity = win_prob * jnp.power(discount, self.pressure_units(obs))
        more_betting_possible = (obs.current_round < Round.RIVER) & (to_call(obs) < obs.stacks[0]) & (obs.stacks[1] > 0)
        realized_equity = pressured_equity * jnp.where(more_betting_possible, self.realization_factor, 1.0)
        return realized_equity >= pot_odds(obs) + self.call_margin

    # ---- preflop as big blind: 3-bet strong hands and some light ones, raise limps ----
    def big_blind_raises(self, obs, win_prob, roll, calls):
        facing_open = to_call(obs) > 0
        three_bet = facing_open & ((win_prob >= self.three_bet_equity) | (calls & (roll < self.light_three_bet_rate)))
        raise_limp = ~facing_open & (win_prob >= self.limp_raise_equity)
        return three_bet | raise_limp

    # ---- postflop when checked to: value bet, semi-bluff and bluff less on later streets ----
    def bets_when_checked_to(self, obs, win_prob, roll, fold_ratio):
        street_decay = jnp.power(self.bluff_street_decay, jnp.clip(obs.current_round - 1, 0, 2))
        bluff_scale = jnp.clip(fold_ratio, self.min_bluff_scale, self.max_bluff_scale)
        # against an opponent that folds less than usual, bet thinner for value
        value_threshold = self.value_bet_equity - self.thin_value_shift * jnp.clip(1.0 - fold_ratio, 0.0, 1.0)
        value_bet = win_prob >= value_threshold
        is_air = win_prob < self.air_equity_limit
        bluff_rate = jnp.where(is_air, self.air_bluff_rate, self.semi_bluff_rate) * street_decay * bluff_scale
        return value_bet | (roll < bluff_rate)

    # ---- decision ----
    def act(self, obs: PlayerTableState, state, key):
        equity_key, roll_key = jax.random.split(key)
        win_prob = equity(obs, equity_key, self.num_samples)
        roll = jax.random.uniform(roll_key)
        preflop = obs.current_round == Round.PRE_FLOP
        on_button = obs.button == 0
        facing_bet = to_call(obs) > 0
        unopened = preflop & (obs.current_bet <= obs.big_blind)
        calls = self.wants_to_call(obs, win_prob, self.pressure_discount)
        bet_total = obs.current_bet + self.bet_pot_fraction * (obs.pot + to_call(obs))

        # default: call or check when the price is right, otherwise fold (or check for free)
        action = choose(calls, check_call(), fold_or_check(obs))
        # preflop as button: open every hand; facing a 3-bet the default call test applies
        action = choose(unopened & on_button, raise_to(obs, self.open_size_bb * obs.big_blind), action)
        # preflop as big blind
        bb_raise = preflop & ~on_button & obs.can_raise & self.big_blind_raises(obs, win_prob, roll, calls)
        action = choose(bb_raise, raise_to(obs, self.reraise_multiple * obs.current_bet), action)
        # postflop when checked to
        bet = ~preflop & ~facing_bet & obs.can_raise & self.bets_when_checked_to(obs, win_prob, roll, self.fold_rate_ratio(state))
        action = choose(bet, raise_to(obs, bet_total), action)
        # postflop facing a bet: raise only the top of the range, else the default call test
        value_raise = ~preflop & facing_bet & obs.can_raise & (win_prob >= self.value_raise_equity)
        action = choose(value_raise, raise_to(obs, bet_total), action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "leviathan_claude",
            "version": 3,
            "parent": 1,
            "lineage": "phase_4_sea_monsters/leviathan_claude",
            "model": MODEL,
            "prompt_version": "botgen-v4",
            "phase": "phase_4_sea_monsters",
            "engine": "decafb9 sha256:295477eff7637056",
            "params": {name: value for name, value in vars(self).items()},
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "Claude desktop"},
            ],
        }
