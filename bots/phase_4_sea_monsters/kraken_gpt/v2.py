"""Track postflop betting and folding; adapt range strength and bluff frequency."""
import jax
import jax.numpy as jnp
from poker_bots.poker_game import Action, Round
from .equity import equity, to_call, check_call, fold_or_check, raise_to, choose


class Bot:
    def __init__(self, num_samples=128, open_size_bb=2.5, reraise_multiple=3.0,
                 threebet_equity=0.63, light_threebet_frequency=0.15,
                 isolate_equity=0.55, value_equity=0.62, raise_equity=0.85,
                 bluff_frequency=0.35, semibluff_frequency=0.50,
                 bluff_split_equity=0.45, bluff_street_decay=0.65,
                 bet_pot_fraction=0.65, range_retention=0.58,
                 prior_street_weight=0.5, realization_fraction=0.90,
                 call_margin_equity=0.02, prior_events=20.0, prior_bet_rate=0.40,
                 prior_fold_rate=0.50, betting_adjustment=0.65, bluff_adjustment=1.25):
        self.num_samples = num_samples  # Monte Carlo deals per decision
        self.open_size_bb = open_size_bb  # raise-to in big blinds
        self.reraise_multiple = reraise_multiple  # multiple of current bet
        self.threebet_equity = threebet_equity  # equity probabilities throughout
        self.light_threebet_frequency = light_threebet_frequency  # mixing probability
        self.isolate_equity = isolate_equity
        self.value_equity = value_equity
        self.raise_equity = raise_equity
        self.bluff_frequency = bluff_frequency  # mixing probability with air
        self.semibluff_frequency = semibluff_frequency  # mixing probability with medium equity
        self.bluff_split_equity = bluff_split_equity
        self.bluff_street_decay = bluff_street_decay  # frequency multiplier per later street
        self.bet_pot_fraction = bet_pot_fraction  # fraction of pot after calling
        self.range_retention = range_retention  # fraction of range retained per pressure unit
        self.prior_street_weight = prior_street_weight  # current-street bet equivalents
        self.realization_fraction = realization_fraction  # fraction of equity realized
        self.call_margin_equity = call_margin_equity
        self.prior_events = prior_events  # pseudo-count opportunities for each rate
        self.prior_bet_rate = prior_bet_rate  # probability of betting when checking is free
        self.prior_fold_rate = prior_fold_rate  # probability of folding when facing our aggression
        self.betting_adjustment = betting_adjustment  # retention change per unit of bet-rate deviation
        self.bluff_adjustment = bluff_adjustment  # bluff probability change per unit of fold-rate deviation
        if not 1 <= num_samples <= 256:
            raise ValueError("num_samples must be in 1..256")

    def init_state(self, key):
        return {
            "opponent_free_actions": jnp.asarray(self.prior_events, jnp.float32),
            "opponent_bets_when_free": jnp.asarray(self.prior_events * self.prior_bet_rate, jnp.float32),
            "opponent_facing_bets": jnp.asarray(self.prior_events, jnp.float32),
            "opponent_folds_to_bets": jnp.asarray(self.prior_events * self.prior_fold_rate, jnp.float32),
        }

    # Tracking: count actual postflop opportunities, including hands where we folded.
    def observe(self, record, state, key):
        history = record.history
        opponent_action = history.is_valid[1:] & (history.player_id[1:] != 0)
        postflop_action = opponent_action & (history.round[1:] > Round.PRE_FLOP) & (history.action_type[1:] <= Action.RAISE)
        same_street = history.is_valid[:-1] & (history.round[:-1] == history.round[1:])
        our_previous_bet = (history.player_id[:-1] == 0) & (history.action_type[:-1] == Action.RAISE)
        facing_our_bet = postflop_action & same_street & our_previous_bet
        checking_free = postflop_action & ~facing_our_bet
        opponent_bet = history.action_type[1:] == Action.RAISE
        opponent_fold = history.action_type[1:] == Action.FOLD
        return {
            "opponent_free_actions": state["opponent_free_actions"] + jnp.sum(checking_free, dtype=jnp.float32),
            "opponent_bets_when_free": state["opponent_bets_when_free"] + jnp.sum(checking_free & opponent_bet, dtype=jnp.float32),
            "opponent_facing_bets": state["opponent_facing_bets"] + jnp.sum(facing_our_bet, dtype=jnp.float32),
            "opponent_folds_to_bets": state["opponent_folds_to_bets"] + jnp.sum(facing_our_bet & opponent_fold, dtype=jnp.float32),
        }

    def act(self, obs, state, key):
        equity_key, mix_key = jax.random.split(key)
        raw_equity = equity(obs, equity_key, self.num_samples)
        mix = jax.random.uniform(mix_key)
        preflop = obs.current_round == Round.PRE_FLOP
        on_button = obs.button == 0
        call_cost = to_call(obs)
        facing_bet = call_cost > 0
        hero_total = obs.contributed[0] + obs.stacks[0]
        opponent_total = jnp.max(jnp.where(obs.seated[1:] & obs.in_hand[1:], obs.contributed[1:] + obs.stacks[1:], 0))
        effective_total = jnp.minimum(hero_total, opponent_total)
        unmatched_bet = jnp.maximum(obs.current_bet - hero_total, 0)
        pot_after_call = jnp.maximum(obs.pot - unmatched_bet + call_cost, 1)
        call_odds = call_cost / pot_after_call

        # Betting pressure: opening preflop raises are wide; reraises narrow ranges.
        history = obs.history
        raises = history.is_valid & (history.action_type == Action.RAISE)
        opponent_raises = raises & (history.player_id != 0)
        this_street = history.round == obs.current_round
        street_bets = jnp.sum(opponent_raises & this_street)
        prior_bets = jnp.sum(opponent_raises & (history.round < obs.current_round))
        my_raises = jnp.sum(raises & this_street & (history.player_id == 0))
        preflop_pressure = jnp.where(my_raises > 0, street_bets, jnp.maximum(street_bets - 1, 0))
        pressure = jnp.where(preflop, preflop_pressure, street_bets + self.prior_street_weight * prior_bets)
        # Broad bettors get less credit; frequent folders face more mixed bluffs.
        opponent_bet_rate = state["opponent_bets_when_free"] / state["opponent_free_actions"]
        opponent_fold_to_bet_rate = state["opponent_folds_to_bets"] / state["opponent_facing_bets"]
        learned_retention = self.range_retention + self.betting_adjustment * (opponent_bet_rate - self.prior_bet_rate)
        retention = jnp.where(preflop, self.range_retention, jnp.clip(learned_retention, jnp.finfo(jnp.float32).eps, 1.0))
        remaining_range = jnp.power(retention, pressure)
        # Remove the weakest part of a random range, preserving equity near the nuts.
        adjusted_equity = jnp.maximum(raw_equity + remaining_range - 1, 0) / remaining_range
        opponent_can_bet = jnp.any(obs.seated[1:] & obs.in_hand[1:] & (obs.stacks[1:] > 0))
        future_betting = (obs.current_round < Round.RIVER) & (call_cost < obs.stacks[0]) & opponent_can_bet
        realization = jnp.where(future_betting, self.realization_fraction, 1.0)
        wants_call = adjusted_equity * realization >= call_odds + self.call_margin_equity
        passive = choose(wants_call, check_call(), fold_or_check(obs))

        # Preflop as button: open every hand, then respect reraises.
        unopened = preflop & (obs.current_bet <= obs.big_blind)
        opening_total = jnp.minimum(self.open_size_bb * obs.big_blind, effective_total)
        action = choose(unopened & on_button & obs.can_raise, raise_to(obs, opening_total), passive)

        # Preflop as big blind: isolate limps and mix light three-bets with value.
        single_open = jnp.sum(raises & (history.round == Round.PRE_FLOP)) == 1
        light_threebet = single_open & wants_call & (mix < self.light_threebet_frequency)
        threebet = preflop & ~on_button & facing_bet & ((adjusted_equity >= self.threebet_equity) | light_threebet)
        isolate = unopened & ~on_button & (raw_equity >= self.isolate_equity)
        reraise_total = jnp.minimum(self.reraise_multiple * obs.current_bet, effective_total)
        action = choose((threebet | isolate) & obs.can_raise, raise_to(obs, reraise_total), action)

        # Postflop when checked to: bet value and mix air and medium-equity bluffs.
        street_step = jnp.clip(obs.current_round - Round.FLOP, 0, 2)
        bluff_scale = jnp.power(self.bluff_street_decay, street_step)
        bluff_rate = jnp.where(raw_equity < self.bluff_split_equity, self.bluff_frequency, self.semibluff_frequency)
        bluff_shift = self.bluff_adjustment * (opponent_fold_to_bet_rate - self.prior_fold_rate)
        learned_bluff_rate = jnp.clip(bluff_rate + bluff_shift, 0.0, 1.0)
        bluff = mix < learned_bluff_rate * bluff_scale
        value_bet = raw_equity >= self.value_equity
        wants_bet = ~facing_bet & (value_bet | bluff)

        # Postflop facing a bet: raise value only after accounting for pressure.
        value_raise = facing_bet & (adjusted_equity >= self.raise_equity)
        bet_total = jnp.minimum(obs.current_bet + self.bet_pot_fraction * pot_after_call, effective_total)
        postflop_raise = ~preflop & obs.can_raise & (wants_bet | value_raise)
        action = choose(postflop_raise, raise_to(obs, bet_total), action)
        return action, state

    def metadata(self):
        return {
            "bot_name": "kraken_gpt",
            "version": 2,
            "parent": 1,
            "lineage": "phase_4_sea_monsters/kraken_gpt",
            "model": "GPT-6 (Codex; exact serving variant unavailable)",
            "prompt_version": "botgen-v4",
            "phase": "phase_4_sea_monsters",
            "engine": "decafb9 sha256:295477eff7637056",
            "params": dict(vars(self)),
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GPT (Codex)"}
            ],
        }
