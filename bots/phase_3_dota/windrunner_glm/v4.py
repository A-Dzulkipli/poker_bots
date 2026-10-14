"""v4: v3 with mixed semi-bluff raises against postflop bets.

Hypothesis: the benchmark field folds far too often to aggression (the probe bot continues against
a raise only above 65% equity, the phase-2 finals fold over half of postflop bets), so raising
bets with strong draws and medium hands should take pots immediately more often than calling.
Only hands with real equity when called are raised this way; pure air keeps folding or calling.
"""
import jax
import jax.numpy as jnp

from poker_bots.poker_game import Action, ActionData, PlayerTableState, Round

from .equity import check_call, choose, equity, fold_or_check, pot_odds, raise_to, to_call


class Bot:
    def __init__(self,
                 num_samples=128,             # Monte Carlo equity samples per decision (cap 256)
                 open_size_bb=2.5,            # button open raise-to, big blinds
                 preflop_reraise_multiple=3.0,  # raise-to as a multiple of the current bet preflop
                 three_bet_equity=0.63,       # equity vs a random hand to 3-bet an open (probability)
                 light_three_bet_freq=0.15,   # 3-bet bluffs with hands that would call (probability)
                 limp_raise_equity=0.55,      # equity to raise a button limp (probability)
                 value_bet_equity=0.62,       # equity to bet postflop when checked to (probability)
                 bluff_frequency=0.33,        # flop bluff rate with air when checked to (probability)
                 semi_bluff_frequency=0.50,   # flop bluff rate with draws or medium hands (probability)
                 bluff_street_decay=0.55,     # multiply the bluff rate by this per street past the flop
                 bluff_equity_split=0.45,     # boundary between air and semi-bluff (probability)
                 bet_pot_fraction=0.65,       # postflop bet size, fraction of pot after calling
                 reraise_equity=0.80,         # equity to raise a postflop bet for value (probability)
                 raise_bluff_frequency=0.25,  # raise instead of call with medium equity (probability)
                 raise_bluff_floor=0.45,      # minimum equity for a semi-bluff raise (probability)
                 call_margin=0.02,            # equity edge over pot odds required to call (probability)
                 aggression_discount=0.75,    # equity multiplier per unit of betting pressure
                 prior_street_weight=0.5,     # weight of earlier-street bets in the pressure count
                 realization_factor=0.90):    # equity realization before the river
        self.num_samples = num_samples
        self.open_size_bb = open_size_bb
        self.preflop_reraise_multiple = preflop_reraise_multiple
        self.three_bet_equity = three_bet_equity
        self.light_three_bet_freq = light_three_bet_freq
        self.limp_raise_equity = limp_raise_equity
        self.value_bet_equity = value_bet_equity
        self.bluff_frequency = bluff_frequency
        self.semi_bluff_frequency = semi_bluff_frequency
        self.bluff_street_decay = bluff_street_decay
        self.bluff_equity_split = bluff_equity_split
        self.bet_pot_fraction = bet_pot_fraction
        self.reraise_equity = reraise_equity
        self.raise_bluff_frequency = raise_bluff_frequency
        self.raise_bluff_floor = raise_bluff_floor
        self.call_margin = call_margin
        self.aggression_discount = aggression_discount
        self.prior_street_weight = prior_street_weight
        self.realization_factor = realization_factor

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        eq_key, roll_key = jax.random.split(key)
        win_prob = equity(obs, eq_key, self.num_samples)
        roll = jax.random.uniform(roll_key)
        preflop = obs.current_round == Round.PRE_FLOP
        on_button = obs.button == 0
        facing_bet = to_call(obs) > 0
        unopened = preflop & (obs.current_bet <= obs.big_blind)

        # betting pressure: each opponent bet this street is a full unit, earlier streets half
        opp = obs.history.is_valid & (obs.history.player_id != 0)
        is_bet = opp & (obs.history.action_type == Action.RAISE)
        street_bets = jnp.sum(is_bet & (obs.history.round == obs.current_round))
        prior_bets = jnp.sum(is_bet & (obs.history.round < obs.current_round))
        my_raises = jnp.sum(obs.history.is_valid & (obs.history.player_id == 0)
                            & (obs.history.action_type == Action.RAISE) & (obs.history.round == obs.current_round))
        # preflop an open is wide, so only a re-raise counts; once I have raised, any opponent raise is one
        preflop_units = jnp.where(my_raises > 0, street_bets, jnp.maximum(street_bets - 1, 0))
        units = jnp.where(preflop, preflop_units, street_bets + self.prior_street_weight * prior_bets)
        discounted = win_prob * jnp.power(self.aggression_discount, units)

        # passive baseline: call when discounted equity beats the price, check when free
        opponent_can_bet = jnp.any(obs.seated[1:] & obs.in_hand[1:] & (obs.stacks[1:] > 0))
        future_betting = (obs.current_round < Round.RIVER) & (to_call(obs) < obs.stacks[0]) & opponent_can_bet
        realized = jnp.where(future_betting, self.realization_factor, 1.0)
        wants_call = discounted * realized >= pot_odds(obs) + self.call_margin
        passive = choose(wants_call, check_call(), fold_or_check(obs))

        # preflop as button: open every hand; facing a 3-bet, fall through to the call test
        action = choose(unopened & on_button, raise_to(obs, self.open_size_bb * obs.big_blind), passive)

        # preflop as big blind: 3-bet the top plus light bluffs, punish limps with the strong range
        three_bet = preflop & ~on_button & facing_bet & ((win_prob >= self.three_bet_equity)
                                                         | (wants_call & (roll < self.light_three_bet_freq)))
        raise_limp = unopened & ~on_button & (win_prob >= self.limp_raise_equity)
        reraise_total = self.preflop_reraise_multiple * obs.current_bet
        action = choose((three_bet | raise_limp) & obs.can_raise, raise_to(obs, reraise_total), action)

        # postflop when checked to: bet for value; bluff air and draws at a rate that decays by street
        bet_total = obs.current_bet + self.bet_pot_fraction * (obs.pot + to_call(obs))
        street_step = jnp.clip(obs.current_round - 1, 0, 2)  # 0 flop, 1 turn, 2 river
        bluff_scale = jnp.power(self.bluff_street_decay, street_step)
        value_bet = win_prob >= self.value_bet_equity
        air_bluff = (win_prob < self.bluff_equity_split) & (roll < self.bluff_frequency * bluff_scale)
        semi_bluff = (win_prob >= self.bluff_equity_split) & (win_prob < self.value_bet_equity) \
            & (roll < self.semi_bluff_frequency * bluff_scale)
        stab = (~preflop) & (~facing_bet) & obs.can_raise & (value_bet | air_bluff | semi_bluff)
        action = choose(stab, raise_to(obs, bet_total), action)

        # postflop facing a bet: raise the top of the range for value, and medium hands as semi-bluffs
        value_raise = win_prob >= self.reraise_equity
        semi_raise = (win_prob >= self.raise_bluff_floor) & (win_prob < self.reraise_equity) & (roll < self.raise_bluff_frequency)
        wants_raise = (~preflop) & facing_bet & obs.can_raise & (value_raise | semi_raise)
        action = choose(wants_raise, raise_to(obs, bet_total), action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "windrunner_glm",
            "version": 4,
            "parent": 3,
            "lineage": "phase_3_dota/windrunner_glm",
            "model": "GLM-5.3 (account:zai-individual-coding-plan/GLM-5.3)",
            "prompt_version": "botgen-v3",
            "phase": "phase_3_dota",
            "engine": "02abc49+wt sha256:295477eff7637056",
            "params": {
                "num_samples": self.num_samples,
                "open_size_bb": self.open_size_bb,
                "preflop_reraise_multiple": self.preflop_reraise_multiple,
                "three_bet_equity": self.three_bet_equity,
                "light_three_bet_freq": self.light_three_bet_freq,
                "limp_raise_equity": self.limp_raise_equity,
                "value_bet_equity": self.value_bet_equity,
                "bluff_frequency": self.bluff_frequency,
                "semi_bluff_frequency": self.semi_bluff_frequency,
                "bluff_street_decay": self.bluff_street_decay,
                "bluff_equity_split": self.bluff_equity_split,
                "bet_pot_fraction": self.bet_pot_fraction,
                "reraise_equity": self.reraise_equity,
                "raise_bluff_frequency": self.raise_bluff_frequency,
                "raise_bluff_floor": self.raise_bluff_floor,
                "call_margin": self.call_margin,
                "aggression_discount": self.aggression_discount,
                "prior_street_weight": self.prior_street_weight,
                "realization_factor": self.realization_factor,
            },
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GLM"},
            ],
        }
