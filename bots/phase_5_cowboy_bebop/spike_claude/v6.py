"""spike_claude v6: draws are bet when checked to, never used to raise a bet (v5 base) range-weighted Monte Carlo equity (opponent bets narrow their range), plus board texture and draws."""
import jax
import jax.numpy as jnp
from poker_bots.poker_game import Action
from poker_bots.hand_eval import evaluate_hands

from .equity import made_hand, hand_category, to_call, pot_odds, fold_or_check, check_call, raise_to, choose


def has_straight(rank_present):
    """True where a (..., 13) rank-presence mask contains five ranks in a row (ace plays low too)."""
    with_low_ace = jnp.concatenate([rank_present[..., 12:], rank_present], axis=-1).astype(jnp.int32)
    window_counts = sum(with_low_ace[..., start:start + 10] for start in range(5))
    return jnp.any(window_counts == 5, axis=-1)


def max_straight_window(rank_present):
    """Most distinct ranks inside any five-rank straight window."""
    with_low_ace = jnp.concatenate([rank_present[12:], rank_present]).astype(jnp.int32)
    return jnp.max(sum(with_low_ace[start:start + 10] for start in range(5)))


def count_by(values, valid, size):
    """Histogram of `values` over `size` bins, counting only valid cards."""
    return jnp.zeros(size, jnp.int32).at[jnp.where(valid, values, 0)].add(valid.astype(jnp.int32))


def range_weighted_equity(obs, key, num_samples, range_tightness):
    """Share of the pot won vs one opponent whose sampled hands are weighted by how strong they are now:
    weight = (percentile of the villain's current hand among the samples) ** range_tightness."""
    known = jnp.zeros(52, jnp.bool_).at[obs.hole_cards].set(True)
    known = known.at[jnp.where(obs.board >= 0, obs.board, 0)].max(obs.board >= 0)

    def sample(sample_key):
        perm = jax.random.permutation(sample_key, 52)
        draws = perm[jnp.argsort(known[perm], stable=True)]
        board = jnp.where(obs.board >= 0, obs.board, draws[2:7])
        villain = draws[:2]
        hero_final = evaluate_hands(jnp.concatenate([obs.hole_cards, board])[None])[0]
        villain_cards = jnp.concatenate([villain, board])
        villain_final = evaluate_hands(villain_cards[None])[0]
        by_board_size = jnp.stack([evaluate_hands(villain_cards[:5]), evaluate_hands(villain_cards[:6]), villain_final])
        high, low = jnp.max(villain % 13), jnp.min(villain % 13)
        preflop_score = 200 * (high == low) + 13 * high + low + 6 * (villain[0] // 13 == villain[1] // 13)
        villain_now = jnp.where(obs.current_round == 0, preflop_score, by_board_size[jnp.clip(obs.current_round - 1, 0, 2)])
        win_share = jnp.where(hero_final > villain_final, 1.0, jnp.where(hero_final == villain_final, 0.5, 0.0))
        return win_share, villain_now

    win_share, villain_now = jax.vmap(sample)(jax.random.split(key, num_samples))
    percentile = (jnp.argsort(jnp.argsort(villain_now)) + 1.0) / num_samples
    weight = percentile ** range_tightness
    return jnp.sum(weight * win_share) / jnp.sum(weight)


class Bot:
    def __init__(self, open_equity=0.47, preflop_reraise_equity=0.63, open_size_bb=2.5, reraise_multiple=3.0,
                 bet_equity=0.62, raise_equity=0.80, call_margin=0.04, tightness_per_raise=0.8, tightness_per_preflop_raise=0.5, tightness_per_call=0.5,
                 scary_board_penalty=0.07, dry_bet_fraction=0.5, wet_bet_fraction=0.8, semi_bluff_outs=8,
                 stab_frequency=0.3, equity_samples=192, typical_aggression=0.35, prior_actions=20.0):
        self.open_equity = open_equity                        # preflop equity vs random needed to raise first in
        self.preflop_reraise_equity = preflop_reraise_equity  # preflop equity needed to re-raise a raise
        self.open_size_bb = open_size_bb                      # first-in raise size, big blinds
        self.reraise_multiple = reraise_multiple              # preflop re-raise size, multiple of the current bet
        self.bet_equity = bet_equity                          # postflop equity needed to bet for value
        self.raise_equity = raise_equity                      # postflop equity needed to raise a bet
        self.call_margin = call_margin                        # equity cushion above pot odds to call
        self.tightness_per_raise = tightness_per_raise        # range-weight exponent added per opponent bet/raise this hand
        self.tightness_per_preflop_raise = tightness_per_preflop_raise  # exponent per opponent preflop raise (HU opens are wide)
        self.tightness_per_call = tightness_per_call          # range-weight exponent added per opponent postflop call
        self.scary_board_penalty = scary_board_penalty        # equity discount for one pair or less on a scary board
        self.dry_bet_fraction = dry_bet_fraction              # bet size on boards without draws, fraction of pot
        self.wet_bet_fraction = wet_bet_fraction              # bet size on boards with draws, fraction of pot
        self.semi_bluff_outs = semi_bluff_outs                # outs needed to bet/raise a draw on flop or turn
        self.stab_frequency = stab_frequency                  # probability of betting air on a dry board when checked to
        self.equity_samples = equity_samples                  # Monte Carlo samples per decision
        self.typical_aggression = typical_aggression          # assumed share of opponent postflop actions that are bets/raises
        self.prior_actions = prior_actions                    # pseudo-count of postflop actions behind that assumption

    def init_state(self, key):
        return {"opponent_postflop_actions": jnp.zeros((), jnp.float32), "opponent_postflop_raises": jnp.zeros((), jnp.float32)}

    # --- Opponent tracking: how often they bet or raise when they act postflop ---
    def observe(self, record, state, key):
        h = record.history
        opponent_postflop = h.is_valid & (h.player_id == 1) & (h.round > 0) & (h.action_type <= Action.RAISE)
        raises = jnp.sum(opponent_postflop & (h.action_type == Action.RAISE)).astype(jnp.float32)
        return {"opponent_postflop_actions": state["opponent_postflop_actions"] + jnp.sum(opponent_postflop).astype(jnp.float32),
                "opponent_postflop_raises": state["opponent_postflop_raises"] + raises}

    def act(self, obs, state, key):
        equity_key, bluff_key = jax.random.split(key)
        street = obs.current_round
        postflop = street > 0
        before_river = postflop & (street < 3)
        board_valid = obs.board >= 0
        cards = jnp.concatenate([obs.hole_cards, obs.board])
        cards_valid = cards >= 0

        # --- Board texture: what the community cards make possible for either player ---
        board_suit_counts = count_by(obs.board // 13, board_valid, 4)
        board_rank_counts = count_by(obs.board % 13, board_valid, 13)
        board_ranks_present = board_rank_counts > 0
        flush_possible = jnp.max(board_suit_counts) >= 3
        straight_possible = max_straight_window(board_ranks_present) >= 3
        board_paired = jnp.max(board_rank_counts) >= 2
        scary_board = flush_possible | straight_possible | board_paired
        draws_available = before_river & ((jnp.max(board_suit_counts) == 2) | straight_possible)
        top_board_rank = jnp.max(jnp.where(board_valid, obs.board % 13, -1))

        # --- Draws: what my hand can still become ---
        strength_category = hand_category(made_hand(obs))
        suit_counts = count_by(cards // 13, cards_valid, 4)
        i_hold_suit = jnp.any(jnp.arange(4)[:, None] == (obs.hole_cards // 13)[None, :], axis=1)
        flush_draw = jnp.any((suit_counts == 4) & (board_suit_counts < 4) & i_hold_suit)
        flush_draw_outs = 9 * flush_draw
        ranks_present = count_by(cards % 13, cards_valid, 13) > 0
        add_one_rank = jnp.eye(13, dtype=jnp.bool_)
        new_straight = has_straight(ranks_present[None, :] | add_one_rank)
        board_alone_straight = has_straight(board_ranks_present[None, :] | add_one_rank)
        straight_ranks = jnp.sum(new_straight & ~board_alone_straight & ~ranks_present)
        straight_draw_outs = 4 * straight_ranks * ~has_straight(ranks_present)
        overcards = jnp.sum(obs.hole_cards % 13 > top_board_rank) * (strength_category == 0)
        draw_outs = jnp.where(before_river, flush_draw_outs + straight_draw_outs + 3 * overcards, 0)

        # --- Effective strength: equity vs the range the opponent's actions this hand suggest ---
        h = obs.history
        opponent_acts = h.is_valid & (h.player_id == 1)
        opponent_raises = jnp.sum(opponent_acts & (h.round == street) & (h.action_type == Action.RAISE))
        postflop_raises = jnp.sum(opponent_acts & (h.round > 0) & (h.action_type == Action.RAISE))
        preflop_raises = jnp.sum(opponent_acts & (h.round == 0) & (h.action_type == Action.RAISE))
        postflop_calls = jnp.sum(opponent_acts & (h.round > 0) & (h.action_type == Action.CHECK_CALL) & (h.amount > 0))
        aggression = (state["opponent_postflop_raises"] + self.typical_aggression * self.prior_actions) / (state["opponent_postflop_actions"] + self.prior_actions)
        bet_tightness = self.tightness_per_raise * jnp.clip(self.typical_aggression / aggression, 0.4, 2.0)  # frequent bettors bet wider ranges
        range_tightness = bet_tightness * postflop_raises + self.tightness_per_preflop_raise * preflop_raises
        range_tightness = range_tightness + self.tightness_per_call * postflop_calls
        raw_equity = range_weighted_equity(obs, equity_key, self.equity_samples, range_tightness)
        weak_on_scary_board = postflop & scary_board & (strength_category <= 1)
        strength = raw_equity - self.scary_board_penalty * weak_on_scary_board
        owed = to_call(obs)
        facing_bet = owed > 0

        # --- Preflop: open, re-raise, or defend by pot odds ---
        facing_preflop_raise = obs.current_bet > obs.big_blind
        preflop_raise = jnp.where(facing_preflop_raise, strength > self.preflop_reraise_equity, strength > self.open_equity)
        preflop_total = jnp.where(facing_preflop_raise, obs.current_bet * self.reraise_multiple, self.open_size_bb * obs.big_blind)

        # --- Postflop: value bets, semi-bluffs with draws, stabs on dry boards ---
        value_raise = jnp.where(facing_bet, strength > self.raise_equity, strength > self.bet_equity)
        semi_bluff = (draw_outs >= self.semi_bluff_outs) & ~facing_bet  # raising a draw into a betting range gets re-raised or called by better
        stab = ~facing_bet & ~draws_available & ~board_paired & (jax.random.uniform(bluff_key) < self.stab_frequency)
        bet_fraction = jnp.where(draws_available, self.wet_bet_fraction, self.dry_bet_fraction)
        postflop_total = obs.current_bet + bet_fraction * (obs.pot + owed)

        # --- Combine ---
        wants_raise = jnp.where(postflop, value_raise | semi_bluff | stab, preflop_raise)
        raise_total = jnp.where(postflop, postflop_total, preflop_total)
        calls = strength > pot_odds(obs) + self.call_margin
        passive = choose(calls, check_call(), fold_or_check(obs))
        action = choose(wants_raise & obs.can_raise, raise_to(obs, raise_total), passive)
        return action, state

    def metadata(self):
        return {
            "bot_name": "spike_claude",
            "version": 6,
            "parent": 5,
            "lineage": "phase_5_cowboy_bebop/spike_claude",
            "model": "Claude Opus 5.5 (claude-opus-5-5)",
            "prompt_version": "botgen-v5",
            "phase": "phase_5_cowboy_bebop",
            "engine": "decafb9 sha256:295477eff7637056",
            "params": {k: v for k, v in vars(self).items()},
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "Claude desktop"},
            ],
        }
