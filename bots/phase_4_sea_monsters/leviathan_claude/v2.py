"""v2: v1 plus opponent tracking of postflop aggression.

Same strategy as v1, but the equity discount per opponent bet now depends on how often this
opponent bets or raises postflop: against a frequent bettor its bets mean less and we call down
lighter; against a rare bettor its bets mean more and we fold more.
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
                 typical_aggression=0.35,      # share of postflop actions that are bets/raises for a typical opponent
                 aggression_prior_actions=20.0,  # pseudo-count postflop actions behind the typical rate
                 max_pressure_scale=1.5):      # cap on how much deeper than default a rare bettor's bets discount
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
        self.typical_aggression = typical_aggression
        self.aggression_prior_actions = aggression_prior_actions
        self.max_pressure_scale = max_pressure_scale

    def init_state(self, key):
        return {"opponent_postflop_actions": jnp.zeros((), jnp.float32),
                "opponent_postflop_bets": jnp.zeros((), jnp.float32)}

    # ---- tracking: count the opponent's postflop actions and how many were bets or raises ----
    def observe(self, record, state, key):
        history = record.history
        opponent_postflop = history.is_valid & (history.player_id == 1) & (history.round >= Round.FLOP)
        bets = jnp.sum(opponent_postflop & (history.action_type == Action.RAISE))
        return {"opponent_postflop_actions": state["opponent_postflop_actions"] + jnp.sum(opponent_postflop),
                "opponent_postflop_bets": state["opponent_postflop_bets"] + bets}

    # ---- tracking: equity discount per opponent bet, scaled by how rarely this opponent bets ----
    def adapted_pressure_discount(self, state):
        prior_bets = self.typical_aggression * self.aggression_prior_actions
        opponent_aggression = (state["opponent_postflop_bets"] + prior_bets) / (
            state["opponent_postflop_actions"] + self.aggression_prior_actions)
        # 1 for a typical opponent, below 1 for a frequent bettor, up to the cap for a rare one
        pressure_scale = jnp.minimum(self.typical_aggression / opponent_aggression, self.max_pressure_scale)
        return 1.0 - (1.0 - self.pressure_discount) * pressure_scale

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
    def bets_when_checked_to(self, obs, win_prob, roll, bluff_scale):
        street_decay = jnp.power(self.bluff_street_decay, jnp.clip(obs.current_round - 1, 0, 2))
        value_bet = win_prob >= self.value_bet_equity
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
        calls = self.wants_to_call(obs, win_prob, self.adapted_pressure_discount(state))
        bet_total = obs.current_bet + self.bet_pot_fraction * (obs.pot + to_call(obs))

        # default: call or check when the price is right, otherwise fold (or check for free)
        action = choose(calls, check_call(), fold_or_check(obs))
        # preflop as button: open every hand; facing a 3-bet the default call test applies
        action = choose(unopened & on_button, raise_to(obs, self.open_size_bb * obs.big_blind), action)
        # preflop as big blind
        bb_raise = preflop & ~on_button & obs.can_raise & self.big_blind_raises(obs, win_prob, roll, calls)
        action = choose(bb_raise, raise_to(obs, self.reraise_multiple * obs.current_bet), action)
        # postflop when checked to
        bet = ~preflop & ~facing_bet & obs.can_raise & self.bets_when_checked_to(obs, win_prob, roll, 1.0)
        action = choose(bet, raise_to(obs, bet_total), action)
        # postflop facing a bet: raise only the top of the range, else the default call test
        value_raise = ~preflop & facing_bet & obs.can_raise & (win_prob >= self.value_raise_equity)
        action = choose(value_raise, raise_to(obs, bet_total), action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "leviathan_claude",
            "version": 2,
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
