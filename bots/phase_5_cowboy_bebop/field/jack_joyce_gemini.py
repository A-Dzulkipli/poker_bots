# Frozen copy of jack_joyce_gemini/v3.py, the version deployed in that lineage's README.
# The shared equity.py helpers are inlined so this file stands alone.
# Source sha256 (first 16): v3.py 3d8fe1e768063504, equity.py 4dd6eb3c9ec3d529
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


# ---- bot, verbatim from v3.py ----

class Bot:
    """Dynamic sizing and bluffing bot. Bets larger with massive equity and incorporates a small frequency of pure bluffs when unchecked."""
    
    def __init__(self, raise_eq: float = 0.60, massive_eq: float = 0.80, bluff_prob: float = 0.10):
        self.raise_eq = raise_eq
        self.massive_eq = massive_eq
        self.bluff_prob = bluff_prob

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        eq_key, bluff_key = jax.random.split(key)
        eq = equity(obs, eq_key, 128)
        
        # Sizing: Pot if massive equity, else half pot
        fraction = jnp.where(eq >= self.massive_eq, 1.0, 0.5)
        bet = obs.current_bet + fraction * (obs.pot + obs.current_bet - obs.contributed[0])
        
        # Bluffing: randomly bluff if no one has bet/raised yet
        is_bluff = (jax.random.uniform(bluff_key) < self.bluff_prob) & (obs.current_bet == 0)
        
        action = choose(eq >= pot_odds(obs), check_call(), fold_or_check(obs))
        should_raise = obs.can_raise & ((eq >= self.raise_eq) | is_bluff)
        action = choose(should_raise, raise_to(obs, bet), action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "jack_joyce_gemini",
            "version": 3,
            "parent": 2,
            "lineage": "phase_1_remedy/jack_joyce_gemini",
            "model": "Gemini-2.5-Pro",
            "prompt_version": "botgen-v1",
            "engine_commit": "02abc49",
            "params": {"raise_eq": self.raise_eq, "massive_eq": self.massive_eq, "bluff_prob": self.bluff_prob},
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "Gemini"}
            ],
        }
