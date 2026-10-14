"""v3: v2 with the fold statistic scoped to postflop bets.

Same two statistics as v2, but the fold rate now counts only hands where we made a postflop
bet or raise: v2's denominator included our 100%-frequency button opens, so an opponent that
simply defends its big blind wide (like a pot-odds caller) read as a non-folder. The bluff
scaling and thin-value threshold respond to the postflop fold rate, the call discount to the
flop betting rate, exactly as in v2.
"""
import jax
import jax.numpy as jnp

from poker_bots.poker_game import Action, ActionData, Round

from .equity import equity, to_call, pot_odds, fold_or_check, check_call, raise_to, choose


class Bot:
    def __init__(self,
                 num_samples=128,               # Monte Carlo equity samples per decision (cap 256)
                 open_size_bb=2.5,              # button open raise-to, big blinds
                 preflop_reraise_multiple=3.0,  # raise-to as a multiple of the bet faced preflop
                 three_bet_equity=0.63,         # equity vs a random hand to 3-bet an open (probability)
                 light_three_bet_freq=0.15,     # 3-bet bluffs with hands that would call (probability)
                 limp_raise_equity=0.55,        # equity to raise a button limp (probability)
                 value_bet_equity=0.62,         # equity to bet postflop when checked to (probability)
                 bluff_frequency=0.33,          # flop bluff rate with air when checked to (probability)
                 semi_bluff_frequency=0.50,     # flop bluff rate with draws and medium hands (probability)
                 bluff_street_decay=0.55,       # multiply the bluff rate by this per street past the flop
                 bluff_equity_split=0.45,       # boundary between air and semi-bluff hands (probability)
                 bet_pot_fraction=0.65,         # postflop bet size, fraction of pot after calling
                 reraise_equity=0.80,           # equity to raise a postflop bet (probability)
                 call_margin=0.02,              # equity edge over pot odds required to call (probability)
                 aggression_discount=0.58,      # equity multiplier per unit of opponent betting pressure
                 prior_street_weight=0.5,       # weight of earlier-street bets in the pressure count
                 realization_factor=0.90,       # equity realization on streets before the river
                 tracking_prior_hands=10.0,     # pseudo-count hands behind each tracking prior
                 prior_fold_rate=0.55,          # assumed fold-to-my-postflop-bet rate before evidence (probability)
                 prior_flop_bet_rate=0.5,       # assumed flop betting rate before evidence (probability)
                 thin_value_slope=0.2,          # value threshold drop per unit fold-rate shortfall (probability)
                 discount_relief_slope=0.4):    # call-discount relief per unit flop-bet-rate excess
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
        self.call_margin = call_margin
        self.aggression_discount = aggression_discount
        self.prior_street_weight = prior_street_weight
        self.realization_factor = realization_factor
        self.tracking_prior_hands = tracking_prior_hands
        self.prior_fold_rate = prior_fold_rate
        self.prior_flop_bet_rate = prior_flop_bet_rate
        self.thin_value_slope = thin_value_slope
        self.discount_relief_slope = discount_relief_slope

    def init_state(self, key: jax.Array):
        return {
            "hands_i_bet_postflop": jnp.zeros((), jnp.float32),
            "opponent_folded_those": jnp.zeros((), jnp.float32),
            "flops_seen": jnp.zeros((), jnp.float32),
            "opponent_bet_flops": jnp.zeros((), jnp.float32),
        }

    def observe(self, record, state, key):
        h = record.history
        valid = h.is_valid
        my_bets = valid & (h.player_id == 0) & (h.action_type == Action.RAISE) & (h.round >= 1)
        i_bet_postflop = jnp.any(my_bets)
        opponent_folded = record.seated[1] & ~record.in_hand[1]
        flop_seen = record.board_dealt >= 3
        opponent_bet_postflop = jnp.any(valid & (h.player_id == 1)
                                        & (h.action_type == Action.RAISE) & (h.round >= 1))
        return {
            "hands_i_bet_postflop": state["hands_i_bet_postflop"] + i_bet_postflop.astype(jnp.float32),
            "opponent_folded_those": state["opponent_folded_those"]
                                     + (i_bet_postflop & opponent_folded).astype(jnp.float32),
            "flops_seen": state["flops_seen"] + flop_seen.astype(jnp.float32),
            "opponent_bet_flops": state["opponent_bet_flops"]
                                  + (flop_seen & opponent_bet_postflop).astype(jnp.float32),
        }

    def act(self, obs, state, key):
        eq_key, roll_key = jax.random.split(key)
        win_prob = equity(obs, eq_key, self.num_samples)
        roll = jax.random.uniform(roll_key)
        preflop = obs.current_round == Round.PRE_FLOP
        on_button = obs.button == 0
        facing_bet = to_call(obs) > 0
        unopened = preflop & (obs.current_bet <= obs.big_blind)

        # --- tracking: smoothed session statistics from pseudo-count priors ---
        fold_to_my_bet = (state["opponent_folded_those"] + self.prior_fold_rate * self.tracking_prior_hands) \
            / (state["hands_i_bet_postflop"] + self.tracking_prior_hands)
        flop_bet_rate = (state["opponent_bet_flops"] + self.prior_flop_bet_rate * self.tracking_prior_hands) \
            / (state["flops_seen"] + self.tracking_prior_hands)

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
        # an opponent that bets nearly every flop bets a near-random range: relax the discount
        discount = jnp.clip(self.aggression_discount
                            + (flop_bet_rate - self.prior_flop_bet_rate) * self.discount_relief_slope,
                            self.aggression_discount, 1.0)
        discounted = win_prob * jnp.power(discount, units)

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

        # postflop when checked to: value bet (thinner vs a station); bluff air and draws more vs a folder
        bluff_appetite = jnp.clip(fold_to_my_bet / self.prior_fold_rate, 0.6, 1.6)
        value_threshold = self.value_bet_equity \
            - jnp.clip(self.prior_fold_rate - fold_to_my_bet, 0.0, 0.3) * self.thin_value_slope
        bet_total = obs.current_bet + self.bet_pot_fraction * (obs.pot + to_call(obs))
        street_step = jnp.clip(obs.current_round - 1, 0, 2)  # 0 flop, 1 turn, 2 river
        bluff_scale = jnp.power(self.bluff_street_decay, street_step) * bluff_appetite
        value_bet = win_prob >= value_threshold
        air_bluff = (win_prob < self.bluff_equity_split) & (roll < self.bluff_frequency * bluff_scale)
        semi_bluff = (win_prob >= self.bluff_equity_split) & (win_prob < value_threshold) \
            & (roll < self.semi_bluff_frequency * bluff_scale)
        stab = (~preflop) & (~facing_bet) & obs.can_raise & (value_bet | air_bluff | semi_bluff)
        action = choose(stab, raise_to(obs, bet_total), action)

        # postflop facing a bet: raise only near the top of the range
        value_raise = (~preflop) & facing_bet & obs.can_raise & (win_prob >= self.reraise_equity)
        action = choose(value_raise, raise_to(obs, bet_total), action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "megalodon_glm",
            "version": 3,
            "parent": 2,
            "lineage": "phase_4_sea_monsters/megalodon_glm",
            "model": "GLM-5.3 (account:zai-individual-coding-plan/GLM-5.3)",
            "prompt_version": "botgen-v4",
            "phase": "phase_4_sea_monsters",
            "engine": "decafb9 sha256:295477eff7637056",
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
                "call_margin": self.call_margin,
                "aggression_discount": self.aggression_discount,
                "prior_street_weight": self.prior_street_weight,
                "realization_factor": self.realization_factor,
                "tracking_prior_hands": self.tracking_prior_hands,
                "prior_fold_rate": self.prior_fold_rate,
                "prior_flop_bet_rate": self.prior_flop_bet_rate,
                "thin_value_slope": self.thin_value_slope,
                "discount_relief_slope": self.discount_relief_slope,
            },
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GLM"},
            ],
        }
