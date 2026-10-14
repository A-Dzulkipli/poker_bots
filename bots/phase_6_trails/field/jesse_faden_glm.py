# Frozen copy of jesse_faden_glm/v1.py, the version deployed in that lineage's README.
# The shared equity.py helpers are inlined so this file stands alone.
# Source sha256 (first 16): v1.py 5829ec1a61b9f0fc, equity.py 442de1555e6a98a4
import jax
import jax.numpy as jnp

from poker_bots.poker_game import PlayerTableState

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
    """v1 baseline: call when Monte Carlo equity beats pot odds, raise a pot fraction with strong equity."""

    def __init__(self, call_margin: float = 0.03, raise_equity: float = 0.62, bet_fraction: float = 0.75, num_samples: int = 128):
        self.call_margin = call_margin      # equity edge over pot odds required to call (probability)
        self.raise_equity = raise_equity    # equity required to bet or raise (probability)
        self.bet_fraction = bet_fraction    # raise-to size as a fraction of the pot after calling
        self.num_samples = num_samples      # Monte Carlo equity samples per decision

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        eq = equity(obs, key, self.num_samples)
        bet_total = obs.current_bet + self.bet_fraction * (obs.pot + to_call(obs))
        call_or_fold = choose(eq >= pot_odds(obs) + self.call_margin, check_call(), fold_or_check(obs))
        action = choose(obs.can_raise & (eq >= self.raise_equity), raise_to(obs, bet_total), call_or_fold)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "jesse_faden_glm",
            "version": 1,
            "parent": None,
            "lineage": "phase_1_remedy/jesse_faden_glm",
            "model": "GLM-5.3",
            "prompt_version": "botgen-v1",
            "engine_commit": "02abc49",
            "params": {
                "call_margin": self.call_margin,
                "raise_equity": self.raise_equity,
                "bet_fraction": self.bet_fraction,
                "num_samples": self.num_samples,
            },
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GLM"},
            ],
        }
