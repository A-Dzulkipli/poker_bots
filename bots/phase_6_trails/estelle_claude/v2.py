"""estelle_claude v2 (wider opponent ranges): decisions from equity against an explicit, action-narrowed opponent range.

The range is a weighted sample of opponent holdings. Each holding's weight starts at 1 and is
multiplied, for every opponent action in this hand, by how plausible that action is for a holding of
that strength on that street (a soft threshold on the holding's strength percentile).
"""
import jax
import jax.numpy as jnp

from poker_bots.hand_eval import evaluate_hands
from poker_bots.poker_game import Action, ActionData
from .equity import check_call, choose, fold_or_check, raise_to


class Bot:
    def __init__(self, num_samples=128, preflop_raise_keep_share=0.85, postflop_bet_keep_share=0.7,
                 call_drop_share=0.15, slowplay_discount=0.5, slowplay_percentile=0.8, range_softness=0.08,
                 open_equity=0.45, value_bet_equity=0.6, value_raise_equity=0.72, call_margin=0.0,
                 bluff_frequency=0.15, bluff_max_equity=0.35, bet_pot_fraction=0.66,
                 preflop_raise_multiple=3.0, use_range=True):
        self.num_samples = num_samples                    # opponent holdings (with runouts) per decision; cap 256
        self.preflop_raise_keep_share = preflop_raise_keep_share  # share of holdings kept by each preflop raise
        self.postflop_bet_keep_share = postflop_bet_keep_share    # share of holdings kept by each postflop bet/raise
        self.call_drop_share = call_drop_share            # weakest share of holdings removed by a call (or limp)
        self.slowplay_discount = slowplay_discount        # weight cut (0..1) for strong holdings that only check/call
        self.slowplay_percentile = slowplay_percentile    # strength percentile above which a hand "should have raised"
        self.range_softness = range_softness              # width of the soft thresholds, in percentile units
        self.open_equity = open_equity                    # equity vs range needed to open / raise a limp preflop
        self.value_bet_equity = value_bet_equity          # equity vs range needed to bet when checked to
        self.value_raise_equity = value_raise_equity      # equity vs range needed to raise a bet
        self.call_margin = call_margin                    # equity above pot odds needed to call
        self.bluff_frequency = bluff_frequency            # probability of betting a weak hand when checked to postflop
        self.bluff_max_equity = bluff_max_equity          # only hands below this equity are bluff candidates
        self.bet_pot_fraction = bet_pot_fraction          # postflop bet/raise size, fraction of the pot after calling
        self.preflop_raise_multiple = preflop_raise_multiple  # preflop raise-to, as a multiple of the current bet
        self.use_range = use_range                        # False = decide against a uniformly random hand (ablation)

    def init_state(self, key):
        return None

    def metadata(self):
        return {
            "bot_name": "estelle_claude", "version": 2, "parent": 1,
            "lineage": "phase_6_trails/estelle_claude", "model": "claude-opus-5-5",
            "prompt_version": "botgen-v6", "phase": "phase_6_trails",
            "engine": "decafb9 sha256:295477eff7637056",
            "params": {k: v for k, v in vars(self).items()},
            "contributions": [{"role": "requester", "who": "a"}, {"role": "generator", "interface": "Claude desktop"}],
        }

    # ---- Sampling: opponent holdings and board runouts that avoid the known cards ----
    def sample_opponent_holdings(self, obs, key):
        known = jnp.zeros(52, dtype=jnp.bool_).at[obs.hole_cards].set(True)
        known = known.at[jnp.maximum(obs.board, 0)].max(obs.board >= 0)

        def deal(sample_key):
            permutation = jax.random.permutation(sample_key, 52)
            unseen_first = permutation[jnp.argsort(known[permutation], stable=True)]
            runout = jnp.where(obs.board >= 0, obs.board, unseen_first[2:7])
            return unseen_first[:2], runout

        return jax.vmap(deal)(jax.random.split(key, self.num_samples))

    # ---- Holding strength: percentile among the sampled holdings, on each street's real board ----
    def strength_percentiles(self, obs, opponent_cards):
        high_rank = jnp.max(opponent_cards % 13, axis=1)
        low_rank = jnp.min(opponent_cards % 13, axis=1)
        suited = (opponent_cards[:, 0] // 13) == (opponent_cards[:, 1] // 13)
        # Short preflop rule: high card, kicker, pairs, suitedness, connectedness.
        preflop_score = high_rank + 0.5 * low_rank + 10.0 * (high_rank == low_rank) + 2.0 * suited
        preflop_score = preflop_score - 0.5 * jnp.clip(high_rank - low_rank - 1, 0, 4)
        board = jnp.broadcast_to(jnp.maximum(obs.board, 0), (self.num_samples, 5))
        flop_score = evaluate_hands(jnp.concatenate([opponent_cards, board[:, :3]], axis=1))
        turn_score = evaluate_hands(jnp.concatenate([opponent_cards, board[:, :4]], axis=1))
        river_score = evaluate_hands(jnp.concatenate([opponent_cards, board], axis=1))
        scores = jnp.stack([preflop_score, flop_score, turn_score, river_score]).astype(jnp.float32)
        beats = (scores[:, :, None] > scores[:, None, :]) + 0.5 * (scores[:, :, None] == scores[:, None, :])
        return jnp.mean(beats, axis=2)

    # ---- Range narrowing: every opponent action re-weights every sampled holding ----
    def opponent_range_log_weights(self, obs, percentiles):
        h = obs.history
        index = jnp.arange(h.player_id.shape[0])
        earlier_same_player_street = ((h.player_id[:, None] == h.player_id[None, :]) & (h.round[:, None] == h.round[None, :])
                                      & (index[None, :] < index[:, None]) & h.is_valid[None, :])
        amount_before = jnp.max(jnp.where(earlier_same_player_street, h.amount[None, :], 0), axis=1)
        by_opponent = h.is_valid & (h.player_id == 1)
        raised = by_opponent & (h.action_type == Action.RAISE)
        called = by_opponent & (h.action_type == Action.CHECK_CALL) & (h.amount > amount_before)
        checked = by_opponent & (h.action_type == Action.CHECK_CALL) & (h.amount <= amount_before)
        percentile_then = percentiles[jnp.clip(h.round, 0, 3)]
        keep_share = jnp.where(h.round == 0, self.preflop_raise_keep_share, self.postflop_bet_keep_share)[:, None]
        soft = self.range_softness
        raise_fit = jax.nn.log_sigmoid((percentile_then - (1.0 - keep_share)) / soft)
        call_fit = jax.nn.log_sigmoid((percentile_then - self.call_drop_share) / soft)
        # Strong holdings usually bet or raise, so a check or call makes them less likely.
        slowplay_fit = jnp.log1p(-self.slowplay_discount * jax.nn.sigmoid((percentile_then - self.slowplay_percentile) / soft))
        per_action = (jnp.where(raised[:, None], raise_fit, 0.0) + jnp.where(called[:, None], call_fit + slowplay_fit, 0.0)
                      + jnp.where(checked[:, None], slowplay_fit, 0.0))
        return jnp.sum(per_action, axis=0)

    # ---- Decision ----
    def act(self, obs, state, key):
        sample_key, bluff_key = jax.random.split(key)
        opponent_cards, runouts = self.sample_opponent_holdings(obs, sample_key)
        log_weights = self.opponent_range_log_weights(obs, self.strength_percentiles(obs, opponent_cards))
        range_weight = jnp.exp(log_weights - jnp.max(log_weights)) if self.use_range else jnp.ones(self.num_samples)
        my_hands = jnp.concatenate([jnp.broadcast_to(obs.hole_cards, (self.num_samples, 2)), runouts], axis=1)
        my_strength = evaluate_hands(my_hands)
        opponent_strength = evaluate_hands(jnp.concatenate([opponent_cards, runouts], axis=1))
        win_share = (my_strength > opponent_strength) + 0.5 * (my_strength == opponent_strength)
        equity_vs_range = jnp.sum(range_weight * win_share) / jnp.sum(range_weight)

        amount_to_call = jnp.minimum(obs.current_bet - obs.contributed[0], obs.stacks[0])
        pot_odds = amount_to_call / jnp.maximum(obs.pot + amount_to_call, 1)
        is_preflop = obs.current_round == 0
        unopened_preflop = is_preflop & (obs.current_bet <= obs.big_blind)
        # Sizing: preflop multiples of the current bet; postflop a fraction of the pot after calling.
        postflop_total = obs.current_bet + self.bet_pot_fraction * (obs.pot + amount_to_call)
        raise_total = jnp.where(is_preflop, self.preflop_raise_multiple * obs.current_bet, postflop_total)

        # Raising: open/iso preflop with a wide range; otherwise only well ahead of the range.
        raise_needed = jnp.where(unopened_preflop, self.open_equity,
                                 jnp.where(amount_to_call > 0, self.value_raise_equity, self.value_bet_equity))
        value_raise = equity_vs_range > raise_needed
        # Bluffing: postflop, checked to, with a hand that rarely wins at showdown.
        bluff = (~is_preflop & (amount_to_call == 0) & (equity_vs_range < self.bluff_max_equity)
                 & (jax.random.uniform(bluff_key) < self.bluff_frequency))
        worth_calling = equity_vs_range >= pot_odds + self.call_margin
        passive = choose(worth_calling, check_call(), fold_or_check(obs))
        action = choose(value_raise | bluff, raise_to(obs, raise_total), passive)
        return action, state
