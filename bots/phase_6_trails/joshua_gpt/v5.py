"""Stateless action-likelihood ranges and sampled showdown equity."""
import jax
import jax.numpy as jnp
from poker_bots.hand_eval import evaluate_hands
from .equity import to_call, choose, check_call, fold_or_check, raise_to


class Bot:
    def __init__(self, samples=128, range_power=1.0, likelihood_floor=0.12,
                 likelihood_width=0.12, open_percentile=0.38, reraise_percentile=0.82,
                 call_percentile=0.45, bet_percentile=0.64, check_discount=0.65,
                 pair_bonus_ranks=13.0, suited_bonus_ranks=3.0,
                 call_margin_equity=0.04, open_equity=0.44, enter_equity=0.44,
                 value_equity=0.64, raise_equity=0.80, bet_pot_fraction=0.65,
                 open_size_bb=2.5, bluff_frequency=0.50, draw_bonus_percentile=0.18):
        self.samples = samples
        self.range_power = range_power
        self.likelihood_floor = likelihood_floor
        self.likelihood_width = likelihood_width
        self.open_percentile = open_percentile
        self.reraise_percentile = reraise_percentile
        self.call_percentile = call_percentile
        self.bet_percentile = bet_percentile
        self.check_discount = check_discount
        self.pair_bonus_ranks = pair_bonus_ranks
        self.suited_bonus_ranks = suited_bonus_ranks
        self.call_margin_equity = call_margin_equity
        self.open_equity = open_equity
        self.enter_equity = enter_equity
        self.value_equity = value_equity
        self.raise_equity = raise_equity
        self.bet_pot_fraction = bet_pot_fraction
        self.open_size_bb = open_size_bb
        self.bluff_frequency = bluff_frequency
        self.draw_bonus_percentile = draw_bonus_percentile

    def init_state(self, key):
        return None

    def percentiles(self, strengths):
        # Ties receive their shared midrank, including board-only made hands.
        ordered = jnp.sort(strengths)
        below = jnp.searchsorted(ordered, strengths, side="left")
        through = jnp.searchsorted(ordered, strengths, side="right")
        return (below + through) / (2.0 * self.samples)

    def range_weights(self, obs, opponent_cards, completed_boards):
        # Rank candidate holdings independently on each historical board prefix.
        ranks = opponent_cards % 13
        paired = ranks[:, 0] == ranks[:, 1]
        suited = opponent_cards[:, 0] // 13 == opponent_cards[:, 1] // 13
        preflop_merit = jnp.sum(ranks, axis=1) + self.pair_bonus_ranks * paired
        preflop_merit = preflop_merit + self.suited_bonus_ranks * suited
        all_cards = jnp.concatenate([opponent_cards, completed_boards], axis=1)
        flop_strength = evaluate_hands(all_cards[:, :5])
        turn_strength = evaluate_hands(all_cards[:, :6])
        river_strength = evaluate_hands(all_cards)
        street_strengths = jnp.stack([preflop_merit, flop_strength, turn_strength, river_strength])
        street_percentiles = jax.vmap(self.percentiles)(street_strengths)
        # Four-card draws can support calls and semibluffs on flop and turn.
        prefix_mask = jnp.arange(7)[None, :] < jnp.array([2, 5, 6, 7])[:, None]
        suit_matches = all_cards[None, :, :, None] // 13 == jnp.arange(4)[None, None, None, :]
        suit_counts = jnp.sum(suit_matches & prefix_mask[:, None, :, None], axis=2)
        flush_draw = jnp.max(suit_counts, axis=2) == 4
        rank_matches = all_cards[None, :, :, None] % 13 == jnp.arange(13)[None, None, None, :]
        rank_present = jnp.any(rank_matches & prefix_mask[:, None, :, None], axis=2)
        straight_windows = (jnp.arange(10)[:, None] + jnp.arange(5)[None, :] - 1) % 13
        straight_draw = jnp.max(jnp.sum(rank_present[:, :, straight_windows], axis=3), axis=2) == 4
        drawing_street = (jnp.arange(4) == 1) | (jnp.arange(4) == 2)
        draw_bonus = self.draw_bonus_percentile * (flush_draw | straight_draw) * drawing_street[:, None]
        street_percentiles = jnp.minimum(street_percentiles + draw_bonus, 1.0)

        # Every opponent action contributes a likelihood to the whole line.
        history = obs.history
        raises = history.is_valid & (history.action_type == 2)
        earlier_street = history.round[None, :] < history.round[:, None]
        previous_street_raises = jnp.sum(earlier_street & raises[None, :], axis=1)
        raise_number = jnp.cumsum(raises) - previous_street_raises
        is_preflop = history.round == 0
        first_raise_threshold = jnp.where(is_preflop, self.open_percentile, self.bet_percentile)
        raise_threshold = jnp.where(raise_number > 1, self.reraise_percentile, first_raise_threshold)
        action_quality = street_percentiles[jnp.clip(history.round, 0, 3), :]
        raise_likelihood = jax.nn.sigmoid((action_quality - raise_threshold[:, None]) / self.likelihood_width)
        call_likelihood = jax.nn.sigmoid((action_quality - self.call_percentile) / self.likelihood_width)
        check_likelihood = 1.0 - self.check_discount * raise_likelihood
        # The big blind can check at one blind; the button completing is a call.
        blind_check = is_preflop & (history.player_id == obs.bb_seat) & (history.amount == obs.big_blind)
        is_check = (history.amount == 0) | blind_check
        # Completing the button blind describes a weaker range than calling an open.
        is_limp = is_preflop & (history.player_id == obs.button) & (history.amount == obs.big_blind)
        passive_likelihood = jnp.where((is_check | is_limp)[:, None], check_likelihood, call_likelihood)
        action_likelihood = jnp.where(raises[:, None], raise_likelihood, passive_likelihood)
        action_likelihood = self.likelihood_floor + (1.0 - self.likelihood_floor) * action_likelihood
        opponent_action = history.is_valid & (history.player_id != 0) & (history.action_type >= 1) & (history.action_type <= 2)
        log_likelihood = jnp.sum(jnp.where(opponent_action[:, None], jnp.log(action_likelihood), 0.0), axis=0)
        return jnp.exp(self.range_power * log_likelihood)

    def act(self, obs, state, key):
        # Deal only unseen cards; reuse the same samples for range and equity.
        sample_key, bluff_key = jax.random.split(key)
        known = jnp.zeros(52, dtype=jnp.bool_).at[obs.hole_cards].set(True)
        known = known.at[jnp.where(obs.board >= 0, obs.board, 0)].max(obs.board >= 0)
        random_order = jax.random.uniform(sample_key, (self.samples, 52)) + known[None, :] * 2.0
        draws = jnp.argsort(random_order, axis=1)[:, :7]
        completed_boards = jnp.where(obs.board[None, :] >= 0, obs.board[None, :], draws[:, :5])
        opponent_cards = draws[:, 5:7]
        opponent_range_weight = self.range_weights(obs, opponent_cards, completed_boards)
        hero_cards = jnp.broadcast_to(obs.hole_cards, (self.samples, 2))
        hero_strength = evaluate_hands(jnp.concatenate([hero_cards, completed_boards], axis=1))
        opponent_strength = evaluate_hands(jnp.concatenate([opponent_cards, completed_boards], axis=1))
        showdown_share = (hero_strength > opponent_strength) + 0.5 * (hero_strength == opponent_strength)
        range_equity = jnp.sum(opponent_range_weight * showdown_share) / jnp.maximum(jnp.sum(opponent_range_weight), 1e-20)

        # Price calls against the posterior, with a margin for future betting.
        call_cost = to_call(obs)
        pot_after_call = obs.pot + call_cost
        call_equity = call_cost / jnp.maximum(pot_after_call, 1)
        future_margin = self.call_margin_equity * (3 - obs.current_round) / 3.0
        can_continue = range_equity >= call_equity + future_margin
        action = choose(can_continue, check_call(), fold_or_check(obs))

        # Preflop: broad button entry, value raises, and priced blind defense.
        preflop = obs.current_round == 0
        unopened = obs.current_bet <= obs.big_blind
        weak_button = preflop & unopened & (obs.button == 0) & (range_equity < self.enter_equity)
        action = choose(weak_button, fold_or_check(obs), action)
        opening = preflop & unopened & (range_equity >= self.open_equity)
        value_threshold = jnp.where(call_cost > 0, self.raise_equity, self.value_equity)
        value_raise = range_equity >= value_threshold

        # Postflop: value bets and occasional low-equity stabs when checked to.
        bluff_draw = jax.random.uniform(bluff_key) < self.bluff_frequency
        bluff = (~preflop) & (call_cost == 0) & (range_equity < call_equity + self.enter_equity) & bluff_draw
        want_raise = opening | value_raise | bluff
        raise_total = obs.current_bet + self.bet_pot_fraction * pot_after_call
        raise_total = jnp.where(preflop & unopened, self.open_size_bb * obs.big_blind, raise_total)
        action = choose(want_raise & obs.can_raise, raise_to(obs, raise_total), action)
        return action, state

    def metadata(self):
        return {
            "bot_name": "joshua_gpt", "version": 5, "parent": 3,
            "lineage": "phase_6_trails/joshua_gpt", "model": "GPT-6 (Codex)",
            "prompt_version": "botgen-v6", "phase": "phase_6_trails",
            "engine": "decafb9 sha256:295477eff7637056", "params": dict(vars(self)),
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GPT (Codex)"}
            ],
        }
