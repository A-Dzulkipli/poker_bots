# Frozen copy of invoker_gpt/v9.py, the version deployed in that lineage's README.
# The shared equity.py helpers are inlined so this file stands alone.
# Source sha256 (first 16): v9.py 28696bad4b452b28, equity.py b040cb274e3b4bdd
"""v9: protect out-of-position checks by occasionally checking value hands."""
import jax
import jax.numpy as jnp
from poker_bots.poker_game import Action, PlayerTableState, Round

import jax
import jax.numpy as jnp

from poker_bots.hand_eval import evaluate_hands
from poker_bots.poker_game import Action, ActionData, PlayerTableState


# ---- shared helpers, verbatim from equity.py ----

def equity(obs: PlayerTableState, key: jax.Array, num_samples: int = 128) -> jax.Array:
    """Monte Carlo share of the pot won at showdown against the opponents still in the hand,
    each holding uniformly random cards. Returns a float32 scalar in [0, 1]."""
    num_opponents = obs.stacks.shape[0] - 1
    known = jnp.zeros(52, dtype=jnp.bool_).at[obs.hole_cards].set(True)
    known = known.at[jnp.where(obs.board >= 0, obs.board, 0)].max(obs.board >= 0)
    live = (obs.seated & obs.in_hand)[1:]

    def sample(key: jax.Array) -> jax.Array:
        perm = jax.random.permutation(key, 52)
        draws = perm[jnp.argsort(known[perm], stable=True)]
        board = jnp.where(obs.board >= 0, obs.board, draws[:5])
        opponents = draws[5:5 + 2 * num_opponents].reshape(num_opponents, 2)
        hero = evaluate_hands(jnp.concatenate([obs.hole_cards, board])[None])[0]
        villains = evaluate_hands(jnp.concatenate([opponents, jnp.broadcast_to(board, (num_opponents, 5))], axis=1))
        beaten = jnp.any(live & (villains > hero))
        ties = jnp.sum(live & (villains == hero))
        return jnp.where(beaten, 0.0, 1.0 / (1 + ties))

    return jnp.mean(jax.vmap(sample)(jax.random.split(key, num_samples)))


def to_call(obs: PlayerTableState) -> jax.Array:
    """Chips needed to call (0 means checking is free), capped at the stack."""
    return jnp.minimum(obs.current_bet - obs.contributed[0], obs.stacks[0])


def pot_odds(obs: PlayerTableState) -> jax.Array:
    """Equity needed to break even on a call: to_call / (pot + to_call)."""
    cost = to_call(obs)
    return cost / jnp.maximum(obs.pot + cost, 1)


def num_live_opponents(obs: PlayerTableState) -> jax.Array:
    return jnp.sum(obs.seated & obs.in_hand) - 1


def fold_or_check(obs: PlayerTableState) -> ActionData:
    """Folds only when there is something to call; otherwise checks."""
    kind = jnp.where(to_call(obs) > 0, Action.FOLD, Action.CHECK_CALL)
    return ActionData(type=jnp.asarray(kind, dtype=jnp.int32), value=jnp.asarray(0, dtype=jnp.int32))


def check_call() -> ActionData:
    return ActionData(type=jnp.asarray(Action.CHECK_CALL, dtype=jnp.int32), value=jnp.asarray(0, dtype=jnp.int32))


def raise_to(obs: PlayerTableState, total: jax.Array) -> ActionData:
    """Raise so this street's contribution becomes `total` chips, clipped to the legal range.
    Becomes a call when raising is not legal."""
    value = jnp.clip(jnp.round(total), obs.raise_lower_bound, obs.raise_upper_bound)
    return ActionData(type=jnp.asarray(Action.RAISE, dtype=jnp.int32), value=jnp.asarray(value, dtype=jnp.int32))


def choose(condition: jax.Array, if_true: ActionData, if_false: ActionData) -> ActionData:
    """Elementwise select between two actions (use instead of Python if)."""
    return jax.tree.map(lambda a, b: jnp.where(condition, a, b), if_true, if_false)


def made_hand(obs: PlayerTableState) -> jax.Array:
    """Strength of your best hand from hole cards plus the board dealt so far (higher is better;
    hand_category gives its category). Preflop, when only two cards are known, returns -1."""
    cards = jnp.concatenate([obs.hole_cards, jnp.maximum(obs.board, 0)])
    by_count = jnp.stack([evaluate_hands(cards[:5]), evaluate_hands(cards[:6]), evaluate_hands(cards)])
    return jnp.where(obs.current_round == 0, -1, by_count[jnp.clip(obs.current_round - 1, 0, 2)])


def hand_category(strength: jax.Array) -> jax.Array:
    """0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 full house, 7 quads,
    8 straight flush; -1 for the preflop value of made_hand."""
    return jnp.where(strength < 0, -1, strength // 13 ** 5)


# ---- bot, verbatim from v9.py ----

class Bot:
    def __init__(self, num_samples=128, open_equity=0.38, isolation_equity=0.56,
                 preflop_raise_equity=0.60, value_equity=0.70, raise_equity=0.86,
                 open_size_bb=2.5, reraise_multiple=3.0, bet_pot_fraction=0.67,
                 premium_equity=0.92, premium_pot_fraction=1.15,
                 raise_exclusion_fraction=0.28, prior_raise_weight=0.4,
                 opening_raise_weight=0.4, realization_fraction=0.92,
                 call_margin_equity=0.025, bluff_frequency=0.35,
                 bluff_ceiling_equity=0.50, threebet_bluff_frequency=0.20,
                 threebet_bluff_ceiling_equity=0.58, trap_frequency=0.35):
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
        self.trap_frequency = trap_frequency  # probability of checking value before the river, out of position
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
        protect_checks = (~on_button) & (obs.current_round < Round.RIVER) & (mix < self.trap_frequency)
        value_bet = (adjusted_equity >= self.value_equity) & ~protect_checks
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
            "version": 9,
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
