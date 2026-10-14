# Frozen copy of max_payne_gpt/v1.py, the version deployed in that lineage's README.
# The shared equity.py helpers are inlined so this file stands alone.
# Source sha256 (first 16): v1.py 3962e9284154c0d5, equity.py 442de1555e6a98a4
"""Simple equity-based baseline with a selective preflop entry range."""
import jax
import jax.numpy as jnp
from poker_bots.poker_game import PlayerTableState, Round

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


# ---- bot, verbatim from v1.py ----

class Bot:
    def __init__(self, num_samples=64, realization_fraction=0.85, call_margin_equity=0.04,
                 preflop_entry_multiplier=1.15, preflop_raise_multiplier=1.8,
                 value_raise_equity=0.62, bet_pot_fraction=0.75, opening_size_bb=2.5):
        self.num_samples = num_samples
        self.realization_fraction = realization_fraction
        self.call_margin_equity = call_margin_equity
        self.preflop_entry_multiplier = preflop_entry_multiplier
        self.preflop_raise_multiplier = preflop_raise_multiplier
        self.value_raise_equity = value_raise_equity
        self.bet_pot_fraction = bet_pot_fraction
        self.opening_size_bb = opening_size_bb
        if not 1 <= num_samples <= 256:
            raise ValueError("num_samples must be between 1 and 256")

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state, key: jax.Array):
        raw_equity = equity(obs, key, self.num_samples)
        opponents = num_live_opponents(obs)
        fair_share = 1.0 / (opponents + 1)
        preflop = obs.current_round == Round.PRE_FLOP
        realization = jnp.where(obs.current_round == Round.RIVER, 1.0, self.realization_fraction)
        calling_equity = raw_equity * realization
        entry_equity = jnp.where(preflop, self.preflop_entry_multiplier * fair_share, 0.0)
        required_equity = jnp.maximum(pot_odds(obs) + self.call_margin_equity, entry_equity)
        can_call = calling_equity >= required_equity
        action = choose(can_call, check_call(), fold_or_check(obs))
        preflop_raise_equity = jnp.minimum(self.value_raise_equity, self.preflop_raise_multiplier * fair_share)
        raise_equity = jnp.where(preflop, preflop_raise_equity, self.value_raise_equity)
        value_raise = raw_equity >= raise_equity
        pot_after_call = obs.pot + to_call(obs)
        raise_total = obs.current_bet + self.bet_pot_fraction * pot_after_call
        opening_total = self.opening_size_bb * obs.big_blind
        unopened = preflop & (obs.current_bet <= obs.big_blind)
        raise_total = jnp.where(unopened, jnp.maximum(raise_total, opening_total), raise_total)
        action = choose(obs.can_raise & value_raise, raise_to(obs, raise_total), action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "max_payne_gpt",
            "version": 1,
            "parent": None,
            "lineage": "phase_1_remedy/max_payne_gpt",
            "model": "GPT-6 (Codex; exact serving variant unavailable)",
            "prompt_version": "botgen-v1",
            "engine_commit": "02abc49",
            "params": {
                "num_samples": self.num_samples,
                "realization_fraction": self.realization_fraction,
                "call_margin_equity": self.call_margin_equity,
                "preflop_entry_multiplier": self.preflop_entry_multiplier,
                "preflop_raise_multiplier": self.preflop_raise_multiplier,
                "value_raise_equity": self.value_raise_equity,
                "bet_pot_fraction": self.bet_pot_fraction,
                "opening_size_bb": self.opening_size_bb,
            },
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GPT (Codex)"}
            ],
        }
