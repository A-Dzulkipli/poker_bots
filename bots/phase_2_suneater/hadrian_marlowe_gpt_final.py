# Frozen copy of hadrian_marlowe_gpt/v6.py, the version deployed in that lineage's README.
# The shared equity.py helpers are inlined so this file stands alone.
# Source sha256 (first 16): v6.py ae90c1a78b7ae83d, equity.py db3b665bb5319b37
"""Realize full equity when calling ends all further betting."""
import jax
import jax.numpy as jnp
from poker_bots.poker_game import Action, PlayerTableState, Round

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


# ---- bot, verbatim from v6.py ----

class Bot:
    def __init__(self, num_samples=128, button_open_equity=0.43,
                 isolation_equity=0.60, preflop_value_equity=0.69,
                 postflop_value_equity=0.68, reraise_margin_equity=0.12,
                 range_exclusion_fraction=0.30, prior_raise_weight=0.50, call_margin_equity=0.03,
                 realization_fraction=0.90, opening_size_bb=2.5,
                 bet_pot_fraction=0.66, preflop_reraise_multiple=3.0,
                 premium_equity=0.90, premium_pot_fraction=1.25):
        self.num_samples = num_samples  # Monte Carlo samples per decision
        self.button_open_equity = button_open_equity  # probability versus random
        self.isolation_equity = isolation_equity  # probability versus random
        self.preflop_value_equity = preflop_value_equity  # adjusted probability
        self.postflop_value_equity = postflop_value_equity  # adjusted probability
        self.reraise_margin_equity = reraise_margin_equity  # probability increment
        self.range_exclusion_fraction = range_exclusion_fraction  # weak-range fraction removed per raise
        self.prior_raise_weight = prior_raise_weight  # prior-street raise equivalents
        self.call_margin_equity = call_margin_equity  # probability above pot odds
        self.realization_fraction = realization_fraction  # before the river
        self.opening_size_bb = opening_size_bb
        self.premium_equity = premium_equity  # adjusted probability
        self.premium_pot_fraction = premium_pot_fraction
        self.bet_pot_fraction = bet_pot_fraction
        self.preflop_reraise_multiple = preflop_reraise_multiple  # current-bet multiple
        if not 1 <= num_samples <= 256:
            raise ValueError("num_samples must be between 1 and 256")

    def init_state(self, key):
        return None

    def act(self, obs: PlayerTableState, state, key):
        raw_equity = equity(obs, key, self.num_samples)
        preflop = obs.current_round == Round.PRE_FLOP
        on_button = obs.button == 0
        call_cost = to_call(obs)
        hero_all_in_total = obs.contributed[0] + obs.stacks[0]
        uncallable_excess = jnp.maximum(obs.current_bet - hero_all_in_total, 0)
        pot_after_call = obs.pot - uncallable_excess + call_cost
        history = obs.history
        on_street = history.is_valid & (history.round == obs.current_round)
        opposing_action = history.player_id != 0
        is_raise = history.action_type == Action.RAISE
        opposing_raises = jnp.sum(on_street & opposing_action & is_raise)
        # Earlier aggression still narrows ranges; almost certain winners stay strong.
        earlier_street = history.is_valid & (history.round < obs.current_round)
        earlier_raises = jnp.sum(earlier_street & opposing_action & is_raise)
        raise_pressure = opposing_raises + self.prior_raise_weight * earlier_raises
        remaining_range = jnp.power(1 - self.range_exclusion_fraction, raise_pressure)
        excluded_range = 1 - remaining_range
        adjusted_equity = jnp.maximum(raw_equity - excluded_range, 0.0)
        adjusted_equity = adjusted_equity / jnp.maximum(remaining_range, 1e-6)
        # No future betting means no equity-realization penalty, even before the river.
        opponent_can_bet = jnp.any(obs.seated[1:] & obs.in_hand[1:] & (obs.stacks[1:] > 0))
        future_streets = (obs.current_round < Round.RIVER) & (call_cost < obs.stacks[0]) & opponent_can_bet
        realization = jnp.where(future_streets, self.realization_fraction, 1.0)
        calling_equity = adjusted_equity * realization
        call_odds = call_cost / jnp.maximum(pot_after_call, 1)
        call_threshold = call_odds + self.call_margin_equity
        wants_call = calling_equity >= call_threshold
        passive_action = choose(wants_call, check_call(), fold_or_check(obs))

        # Preflop: raise a wide button range; isolate limps more selectively.
        unopened = obs.current_bet <= obs.big_blind
        button_open = on_button & unopened & (raw_equity >= self.button_open_equity)
        isolate_limp = (~on_button) & unopened & (raw_equity >= self.isolation_equity)
        value_reraise = (~unopened) & (adjusted_equity >= self.preflop_value_equity)
        preflop_raise = button_open | isolate_limp | value_reraise
        opening_total = self.opening_size_bb * obs.big_blind
        reraise_total = self.preflop_reraise_multiple * obs.current_bet
        preflop_total = jnp.where(unopened, opening_total, reraise_total)

        # Postflop: bet for value, but require more strength to raise a bet.
        facing_bet = call_cost > 0
        raise_margin = jnp.where(facing_bet, self.reraise_margin_equity, 0.0)
        value_threshold = self.postflop_value_equity + raise_margin
        postflop_raise = adjusted_equity >= value_threshold
        # Strong hands can build bigger pots without automatically going all-in.
        premium_hand = adjusted_equity >= self.premium_equity
        bet_fraction = jnp.where(premium_hand, self.premium_pot_fraction, self.bet_pot_fraction)
        postflop_total = obs.current_bet + bet_fraction * pot_after_call
        wants_raise = jnp.where(preflop, preflop_raise, postflop_raise)
        raise_total = jnp.where(preflop, preflop_total, postflop_total)
        # Respect effective stacks; betting into an all-in opponent is pointless.
        other_seats = jnp.arange(obs.stacks.shape[0]) != 0
        live_opponents = other_seats & obs.seated & obs.in_hand
        opponent_total = jnp.max(jnp.where(live_opponents, obs.stacks + obs.contributed, 0))
        effective_total = jnp.minimum(obs.raise_upper_bound, opponent_total)
        raise_total = jnp.minimum(raise_total, effective_total)
        legal_raise = obs.can_raise & wants_raise
        action = choose(legal_raise, raise_to(obs, raise_total), passive_action)
        return action, state

    def metadata(self):
        return {
            "bot_name": "hadrian_marlowe_gpt",
            "version": 6,
            "parent": 4,
            "lineage": "phase_2_suneater/hadrian_marlowe_gpt",
            "model": "GPT-6 (Codex; exact serving variant unavailable)",
            "prompt_version": "botgen-v2",
            "phase": "phase_2_suneater",
            "engine": "02abc49+wt sha256:295477eff7637056",
            "params": dict(vars(self)),
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GPT (Codex)"}
            ],
        }
