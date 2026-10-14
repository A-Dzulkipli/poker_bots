"""v8: use smaller ordinary postflop bets with the same betting range."""
import jax
import jax.numpy as jnp
from poker_bots.poker_game import Action, PlayerTableState, Round
from .equity import equity, to_call, choose, raise_to, check_call, fold_or_check


class Bot:
    def __init__(self, num_samples=128, open_equity=0.38, isolation_equity=0.56,
                 preflop_raise_equity=0.60, value_equity=0.70, raise_equity=0.86,
                 open_size_bb=2.5, reraise_multiple=3.0, bet_pot_fraction=0.40,
                 premium_equity=0.92, premium_pot_fraction=1.15,
                 raise_exclusion_fraction=0.28, prior_raise_weight=0.4,
                 opening_raise_weight=0.4, realization_fraction=0.92,
                 call_margin_equity=0.025, bluff_frequency=0.35,
                 bluff_ceiling_equity=0.50, threebet_bluff_frequency=0.20,
                 threebet_bluff_ceiling_equity=0.58):
        self.num_samples = num_samples  # Monte Carlo deals, at most 256
        self.open_equity = open_equity  # equity probabilities throughout
        self.isolation_equity = isolation_equity
        self.preflop_raise_equity = preflop_raise_equity
        self.value_equity = value_equity
        self.raise_equity = raise_equity
        self.open_size_bb = open_size_bb  # raise-to, big blinds
        self.reraise_multiple = reraise_multiple  # multiple of current bet
        self.bet_pot_fraction = bet_pot_fraction
        self.premium_equity = premium_equity
        self.premium_pot_fraction = premium_pot_fraction
        self.raise_exclusion_fraction = raise_exclusion_fraction  # weakest range fraction removed
        self.prior_raise_weight = prior_raise_weight  # current-street raise equivalents
        self.opening_raise_weight = opening_raise_weight  # opening raise equivalents
        self.realization_fraction = realization_fraction  # fraction of equity realized
        self.call_margin_equity = call_margin_equity
        self.bluff_frequency = bluff_frequency  # probability with eligible weak hands
        self.bluff_ceiling_equity = bluff_ceiling_equity
        self.threebet_bluff_frequency = threebet_bluff_frequency  # probability with eligible hands
        self.threebet_bluff_ceiling_equity = threebet_bluff_ceiling_equity  # probability against random
        if not 1 <= num_samples <= 256:
            raise ValueError("num_samples must be in 1..256")

    def init_state(self, key):
        return None

    def act(self, obs: PlayerTableState, state, key):
        equity_key, mix_key = jax.random.split(key)
        raw_equity = equity(obs, equity_key, self.num_samples)
        mix = jax.random.uniform(mix_key)
        preflop = obs.current_round == Round.PRE_FLOP
        on_button = obs.button == 0
        call_cost = to_call(obs)
        facing_bet = call_cost > 0
        hero_total = obs.contributed[0] + obs.stacks[0]
        live_opponents = obs.seated & obs.in_hand & (jnp.arange(obs.stacks.shape[0]) != 0)
        opponent_total = jnp.max(jnp.where(live_opponents, obs.stacks + obs.contributed, 0))
        effective_total = jnp.minimum(hero_total, opponent_total)
        uncallable_excess = jnp.maximum(obs.current_bet - hero_total, 0)
        pot_after_call = obs.pot - uncallable_excess + call_cost
        call_odds = call_cost / jnp.maximum(pot_after_call, 1)

        # History: an opening raise conveys less strength than a re-raise.
        history = obs.history
        valid_raise = history.is_valid & (history.action_type == Action.RAISE)
        opposing_raise = valid_raise & (history.player_id != 0)
        current_street = history.round == obs.current_round
        street_raises = jnp.sum(opposing_raise & current_street)
        previous_raises = jnp.sum(opposing_raise & (history.round < obs.current_round))
        preflop_raises = jnp.sum(opposing_raise & (history.round == Round.PRE_FLOP))
        opening_discount = (1 - self.opening_raise_weight) * jnp.minimum(preflop_raises, 1)
        current_pressure = street_raises - jnp.where(preflop, opening_discount, 0)
        previous_pressure = previous_raises - jnp.where(preflop, 0, opening_discount)
        raise_pressure = current_pressure + self.prior_raise_weight * previous_pressure
        remaining_range = jnp.power(1 - self.raise_exclusion_fraction, raise_pressure)
        adjusted_equity = jnp.maximum(raw_equity + remaining_range - 1, 0) / remaining_range

        # Calling: protect short stacks from unmatched-pot odds and future-bet penalties.
        opponent_can_bet = jnp.any(live_opponents & (obs.stacks > 0))
        future_betting = (obs.current_round < Round.RIVER) & (call_cost < obs.stacks[0]) & opponent_can_bet
        realization = jnp.where(future_betting, self.realization_fraction, 1.0)
        wants_call = adjusted_equity * realization >= call_odds + self.call_margin_equity
        passive = choose(wants_call, check_call(), fold_or_check(obs))

        # Preflop as button: open wide, fold the weakest hands rather than limp them.
        unopened = obs.current_bet <= obs.big_blind
        opening_total = self.open_size_bb * obs.big_blind
        button_action = choose(raw_equity >= self.open_equity, raise_to(obs, opening_total), fold_or_check(obs))

        # Preflop as big blind: isolate strong hands; otherwise take the free flop.
        isolate = raw_equity >= self.isolation_equity
        blind_action = choose(isolate, raise_to(obs, opening_total), check_call())
        opening_action = choose(on_button, button_action, blind_action)
        preflop_value = adjusted_equity >= self.preflop_raise_equity
        single_open = jnp.sum(valid_raise & (history.round == Round.PRE_FLOP)) == 1
        suited = obs.hole_cards[0] // 13 == obs.hole_cards[1] // 13
        ace_blocker = jnp.any(obs.hole_cards % 13 == 12)
        bluff_candidate = (suited | ace_blocker) & (raw_equity < self.threebet_bluff_ceiling_equity)
        bluff_threebet = single_open & bluff_candidate & (mix < self.threebet_bluff_frequency)
        preflop_raise = preflop_value | bluff_threebet
        reraise_total = jnp.minimum(self.reraise_multiple * obs.current_bet, effective_total)
        preflop_response = choose(preflop_raise & obs.can_raise, raise_to(obs, reraise_total), passive)
        preflop_action = choose(unopened & obs.can_raise, opening_action, preflop_response)

        # Postflop when checked to: mix weak hands into a value-heavy betting range.
        earlier_street = history.is_valid & (history.round < obs.current_round)
        latest_raise_index = jnp.max(jnp.where(earlier_street & valid_raise, jnp.arange(history.is_valid.shape[0]), -1))
        previous_aggressor = history.player_id[jnp.maximum(latest_raise_index, 0)]
        initiative = (latest_raise_index >= 0) & (previous_aggressor == 0)
        bluff_opportunity = on_button | initiative
        weak_hand = raw_equity < self.bluff_ceiling_equity
        bluff = bluff_opportunity & weak_hand & (mix < self.bluff_frequency)
        value_bet = adjusted_equity >= self.value_equity
        unopened_bet = value_bet | bluff

        # Postflop facing a bet: call bluff-catchers and raise only strong value.
        value_raise = adjusted_equity >= self.raise_equity
        wants_bet = jnp.where(facing_bet, value_raise, unopened_bet)
        premium = adjusted_equity >= self.premium_equity
        bet_fraction = jnp.where(premium, self.premium_pot_fraction, self.bet_pot_fraction)
        postflop_total = obs.current_bet + bet_fraction * pot_after_call
        postflop_total = jnp.minimum(postflop_total, effective_total)
        postflop_action = choose(wants_bet & obs.can_raise, raise_to(obs, postflop_total), passive)
        action = choose(preflop, preflop_action, postflop_action)
        return action, state

    def metadata(self):
        return {
            "bot_name": "invoker_gpt",
            "version": 8,
            "parent": 6,
            "lineage": "phase_3_dota/invoker_gpt",
            "model": "GPT-6 (Codex; exact serving variant unavailable)",
            "prompt_version": "botgen-v3",
            "phase": "phase_3_dota",
            "engine": "02abc49+wt sha256:295477eff7637056",
            "params": dict(vars(self)),
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GPT (Codex)"}
            ],
        }
