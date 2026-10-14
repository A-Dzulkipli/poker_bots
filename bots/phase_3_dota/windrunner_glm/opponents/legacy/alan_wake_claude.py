# Frozen copy of alan_wake_claude/v3.py, the version deployed in that lineage's README.
# The shared equity.py helpers are inlined so this file stands alone.
# Source sha256 (first 16): v3.py de77b7056a284d1c, equity.py 442de1555e6a98a4
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


# ---- bot, verbatim from v3.py ----

class Bot:
    """Equity-driven bot: call when equity beats pot odds, raise when equity is a multiple
    of our fair share against the players left. Equity assumes random opponent hands, so
    it is shrunk for every raise an opponent has made on this street."""

    def __init__(
        self,
        num_samples: int = 128,               # Monte Carlo samples per decision (cap 256)
        raise_strength: float = 1.5,          # raise when equity >= this times our fair share
        raise_discount: float = 0.15,         # fraction of equity removed per opponent raise this street
        call_margin: float = 0.03,            # equity probability required above pot odds to call
        preflop_raise_multiple: float = 3.0,  # preflop raise-to = this times the current bet
        bet_pot_fraction: float = 0.66,       # postflop raise = call + this fraction of the pot
        shove_commit_fraction: float = 0.40,  # shove if the raise uses this share of our total chips
    ):
        self.num_samples = num_samples
        self.raise_strength = raise_strength
        self.raise_discount = raise_discount
        self.call_margin = call_margin
        self.preflop_raise_multiple = preflop_raise_multiple
        self.bet_pot_fraction = bet_pot_fraction
        self.shove_commit_fraction = shove_commit_fraction

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        history = obs.history
        this_street = history.is_valid & (history.round == obs.current_round)
        opponent_raises = jnp.sum(this_street & (history.action_type == Action.RAISE) & (history.player_id != 0))
        # random-hand equity overstates our chances against someone who raised
        win_prob = equity(obs, key, self.num_samples) * jnp.power(1 - self.raise_discount, opponent_raises)
        is_preflop = obs.current_round == Round.PRE_FLOP
        pot_after_call = obs.pot + obs.current_bet - obs.contributed[0]
        # preflop: open to 3bb, re-raise to 3x the raise
        preflop_size = self.preflop_raise_multiple * obs.current_bet
        # postflop: match the bet, then add a fraction of the pot
        postflop_size = obs.current_bet + self.bet_pot_fraction * pot_after_call
        bet_size = jnp.where(is_preflop, preflop_size, postflop_size)
        # once a raise would commit a big share of our chips, just go all-in
        all_in_total = obs.contributed[0] + obs.stacks[0]
        commits_most = bet_size >= self.shove_commit_fraction * all_in_total
        bet_size = jnp.where(commits_most, all_in_total, bet_size)
        wants_call = win_prob >= pot_odds(obs) + self.call_margin
        # 1.0 = an average hand against this many opponents, 2.0 = twice our fair share
        strength = win_prob * (num_live_opponents(obs) + 1)
        wants_raise = obs.can_raise & (strength >= self.raise_strength)
        action = choose(wants_call, check_call(), fold_or_check(obs))
        action = choose(wants_raise, raise_to(obs, bet_size), action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "alan_wake_claude",
            "version": 3,
            "parent": 2,
            "lineage": "phase_1_remedy/alan_wake_claude",
            "model": "Claude Opus 5.5 (claude-opus-5-5)",
            "prompt_version": "botgen-v1",
            "engine_commit": "02abc49",
            "params": {
                "num_samples": self.num_samples,
                "raise_strength": self.raise_strength,
                "raise_discount": self.raise_discount,
                "call_margin": self.call_margin,
                "preflop_raise_multiple": self.preflop_raise_multiple,
                "bet_pot_fraction": self.bet_pot_fraction,
                "shove_commit_fraction": self.shove_commit_fraction,
            },
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "Claude desktop (Code)"},
            ],
        }
