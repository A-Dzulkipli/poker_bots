"""Faye: equity-led heads-up play with explicit texture and drawing outs."""
import jax
import jax.numpy as jnp
from .equity import equity, to_call, pot_odds, fold_or_check, check_call, raise_to, choose


class Bot:
    def __init__(self, samples=64, open_equity=0.48, value_equity=0.70,
                 reraise_equity=0.84, call_margin=0.08, pressure_penalty=0.16,
                 open_size_bb=2.5, bet_pot_fraction=0.25, texture_weight=0.06,
                 draw_weight=0.12, bluff_frequency=0.12, features_enabled=True, bet_range_fraction=0.50,
                 quiet_range_fraction=0.80, reraise_range_fraction=0.25, prior_actions=12.0,
                 prior_fold_rate=0.45, prior_raise_rate=0.30,
                 exploit_bluff_frequency=0.80, fold_profit_margin=0.04, semibluff_draw_probability=0.15,
                 semibluff_equity=0.40, pressure_odds_cap=0.50):
        self.semibluff_draw_probability = semibluff_draw_probability  # minimum next-card improvement probability
        self.semibluff_equity = semibluff_equity  # minimum random-hand equity for a semibluff
        self.pressure_odds_cap = pressure_odds_cap  # maximum pot-odds component of pressure
        self.prior_actions = prior_actions  # pseudo-count decisions for rate smoothing
        self.prior_fold_rate = prior_fold_rate  # probability of folding when facing our bet
        self.prior_raise_rate = prior_raise_rate  # postflop raise/action probability
        self.exploit_bluff_frequency = exploit_bluff_frequency  # frequency against profitable folders
        self.fold_profit_margin = fold_profit_margin  # probability margin over immediate break-even
        self.samples = samples  # Monte Carlo deals per decision
        self.open_equity = open_equity  # showdown share versus random
        self.value_equity = value_equity  # minimum share for an unraised value bet
        self.reraise_equity = reraise_equity  # minimum share facing a raise
        self.call_margin = call_margin  # share added to pot odds
        self.pressure_penalty = pressure_penalty  # share lost per unit of pressure
        self.open_size_bb = open_size_bb
        self.bet_pot_fraction = bet_pot_fraction
        self.texture_weight = texture_weight  # maximum texture equity discount
        self.draw_weight = draw_weight  # equity bonus per next-card improvement probability
        self.bluff_frequency = bluff_frequency  # probability on eligible free actions
        self.features_enabled = features_enabled
        self.bet_range_fraction = bet_range_fraction  # fraction of strength distribution when betting
        self.quiet_range_fraction = quiet_range_fraction  # fraction after passive selection
        self.reraise_range_fraction = reraise_range_fraction  # fraction facing repeated raises

    def init_state(self, key):
        return {name: jnp.zeros((), jnp.float32) for name in (
            "bet_responses", "folds_to_bet", "postflop_actions", "postflop_raises",
            "preflop_actions", "preflop_raises")}

    def observe(self, record, state, key):
        # Count actual responses to our bets, including calls and reraises.
        history = record.history
        opponent_action = history.is_valid & (history.player_id != 0)
        previous_hero_raise = jnp.roll(history.is_valid & (history.player_id == 0) & (history.action_type == 2), 1)
        same_street = history.round == jnp.roll(history.round, 1)
        bet_response = opponent_action & previous_hero_raise & same_street & (history.round > 0) & (jnp.arange(history.is_valid.size) > 0)
        postflop_action = opponent_action & (history.round > 0)
        preflop_action = opponent_action & (history.round == 0) & (history.action_type < 3)
        return {
            "preflop_actions": state["preflop_actions"] + jnp.sum(preflop_action).astype(jnp.float32),
            "preflop_raises": state["preflop_raises"] + jnp.sum(preflop_action & (history.action_type == 2)).astype(jnp.float32),
            "bet_responses": state["bet_responses"] + jnp.sum(bet_response).astype(jnp.float32),
            "folds_to_bet": state["folds_to_bet"] + jnp.sum(bet_response & (history.action_type == 0)).astype(jnp.float32),
            "postflop_actions": state["postflop_actions"] + jnp.sum(postflop_action).astype(jnp.float32),
            "postflop_raises": state["postflop_raises"] + jnp.sum(postflop_action & (history.action_type == 2)).astype(jnp.float32),
        }

    def act(self, obs, state, key):
        # Cards and pot accounting.
        equity_key, bluff_key = jax.random.split(key)
        showdown_equity = equity(obs, equity_key, self.samples)
        cost = to_call(obs)
        preflop = obs.current_round == 0
        free_action = cost == 0
        pot_after_call = obs.pot + cost
        hole_ranks = obs.hole_cards % 13
        board_valid = obs.board >= 0
        board_ranks = obs.board % 13
        board_suits = obs.board // 13
        board_count = jnp.sum(board_valid)
        rank_axis = jnp.arange(13)
        suit_axis = jnp.arange(4)
        # Texture: paired ranks, suit concentration, straight windows, high cards.
        board_rank_counts = jnp.sum(board_valid[:, None] & (board_ranks[:, None] == rank_axis), axis=0)
        board_suit_counts = jnp.sum(board_valid[:, None] & (board_suits[:, None] == suit_axis), axis=0)
        paired_board = jnp.max(board_rank_counts) >= 2
        suit_concentration = jnp.max(board_suit_counts)
        straight_windows = (jnp.arange(10)[:, None] + jnp.arange(5)[None, :] - 1) % 13
        board_connections = jnp.max(jnp.sum(board_rank_counts[straight_windows] > 0, axis=1))
        high_board = jnp.max(jnp.where(board_valid, board_ranks, -1)) >= 10
        wetness = ((suit_concentration >= 2).astype(jnp.float32) + (board_connections >= 3)) / 2
        danger = ((suit_concentration >= 3).astype(jnp.float32) + (board_connections >= 4) + paired_board) / 3
        # Count distinct unseen next cards completing flushes, straights, or overcards.
        hole_rank_counts = jnp.sum(hole_ranks[:, None] == rank_axis, axis=0)
        all_rank_counts = board_rank_counts + hole_rank_counts
        hole_suit_counts = jnp.sum((obs.hole_cards // 13)[:, None] == suit_axis, axis=0)
        flush_draw_suits = (board_suit_counts + hole_suit_counts == 4) & (hole_suit_counts > 0)
        window_counts = jnp.sum(all_rank_counts[straight_windows] > 0, axis=1)
        missing_window_rank = (straight_windows[:, :, None] == rank_axis) & (all_rank_counts[straight_windows, None] == 0)
        straight_out_ranks = jnp.any((window_counts == 4)[:, None, None] & missing_window_rank, axis=(0, 1))
        unseen_cards = jnp.arange(52)
        known_cards = jnp.any(unseen_cards[:, None] == obs.hole_cards, axis=1) | jnp.any(unseen_cards[:, None] == obs.board, axis=1)
        flush_outs = flush_draw_suits[unseen_cards // 13]
        straight_outs = straight_out_ranks[unseen_cards % 13] & ~jnp.any(window_counts == 5)
        overcard_ranks = (hole_rank_counts > 0) & (rank_axis > jnp.max(jnp.where(board_valid, board_ranks, -1)))
        pair_outs = overcard_ranks[unseen_cards % 13] & (jnp.max(all_rank_counts) == 1)
        draw_outs = jnp.sum(~known_cards & (flush_outs | straight_outs | pair_outs))
        draw_probability = jnp.where(board_count < 5, draw_outs / jnp.maximum(50 - board_count, 1), 0.0)
        texture_adjustment = self.features_enabled * self.texture_weight * danger * (1 - showdown_equity)
        draw_adjustment = self.features_enabled * self.draw_weight * draw_probability
        # A bet is evidence of a stronger range; repeated raises compound that evidence.
        history = obs.history
        street_raises = jnp.sum(history.is_valid & (history.round == obs.current_round) & (history.action_type == 2))
        pressure = jnp.minimum(cost / jnp.maximum(pot_after_call, 1), self.pressure_odds_cap) + jnp.maximum(street_raises - 1, 0)
        fold_rate = (state["folds_to_bet"] + self.prior_actions * self.prior_fold_rate) / (state["bet_responses"] + self.prior_actions)
        raise_rate = (state["postflop_raises"] + self.prior_actions * self.prior_raise_rate) / (state["postflop_actions"] + self.prior_actions)
        preflop_raise_rate = (state["preflop_raises"] + self.prior_actions * self.prior_raise_rate) / (state["preflop_actions"] + self.prior_actions)
        betting_range = jnp.clip(self.bet_range_fraction + raise_rate - self.prior_raise_rate, self.reraise_range_fraction, 1.0)
        range_fraction = jnp.where(free_action, self.quiet_range_fraction, betting_range)
        range_fraction = jnp.where(street_raises > 1, self.reraise_range_fraction, range_fraction)
        range_equity = jnp.maximum(1 - (1 - showdown_equity) / range_fraction, 0.0)
        adjusted_equity = jnp.where(preflop, showdown_equity - self.pressure_penalty * pressure * (1 - preflop_raise_rate), range_equity)
        adjusted_equity = adjusted_equity - jnp.where(preflop, 0.0, texture_adjustment) + jnp.where(preflop, 0.0, draw_adjustment)
        # Open pots cheaply, defend by price, and demand real strength for reraises.
        unopened = preflop & (obs.current_bet <= obs.big_blind)
        call_threshold = pot_odds(obs) + self.call_margin
        continue_hand = free_action | (adjusted_equity >= call_threshold)
        action = choose(continue_hand, check_call(), fold_or_check(obs))
        open_raise = unopened & (showdown_equity >= self.open_equity)
        repeated_raise = street_raises > jnp.where(preflop, 1, 0)
        value_threshold = jnp.where(repeated_raise, self.reraise_equity, self.value_equity)
        value_threshold = value_threshold - jnp.where(preflop, self.pressure_penalty * jnp.maximum(preflop_raise_rate - self.prior_raise_rate, 0), 0.0)
        value_raise = adjusted_equity >= value_threshold
        # Draws support semibluffs; dry high-card boards support occasional stabs.
        useful_draw = (draw_probability >= self.semibluff_draw_probability) & (showdown_equity >= self.semibluff_equity)
        dry_stab = (wetness == 0) & high_board & (obs.button == 0)
        eligible_bluff = jnp.where(self.features_enabled, useful_draw | dry_stab, obs.button == 0)
        break_even_fold_rate = self.bet_pot_fraction / (1 + self.bet_pot_fraction)
        profitable_steal = fold_rate > break_even_fold_rate + self.fold_profit_margin
        bluff_probability = jnp.where(profitable_steal, self.exploit_bluff_frequency, self.bluff_frequency)
        bluff = free_action & ~preflop & (eligible_bluff | profitable_steal) & (jax.random.uniform(bluff_key) < bluff_probability)
        raise_hand = obs.can_raise & (open_raise | value_raise | bluff)
        bet_fraction = self.bet_pot_fraction + self.features_enabled * self.texture_weight * wetness
        raise_total = obs.current_bet + bet_fraction * pot_after_call
        raise_total = jnp.where(open_raise, self.open_size_bb * obs.big_blind, raise_total)
        action = choose(raise_hand, raise_to(obs, raise_total), action)
        return action, state

    def metadata(self):
        return {
            "bot_name": "faye_gpt", "version": 8, "parent": 7,
            "lineage": "phase_5_cowboy_bebop/faye_gpt", "model": "GPT-6 (Codex)",
            "prompt_version": "botgen-v5", "phase": "phase_5_cowboy_bebop",
            "engine": "decafb9 sha256:295477eff7637056", "params": dict(vars(self)),
            "contributions": [{"role": "requester", "who": "a"},
                              {"role": "generator", "interface": "GPT (Codex)"}],
        }
