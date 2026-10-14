# Frozen copy of bassander_lin_glm/v8.py, the version deployed in that lineage's README.
# The shared equity.py helpers are inlined so this file stands alone.
# Source sha256 (first 16): v8.py a2c3349eb617ea36, equity.py db3b665bb5319b37
"""v8: v7 with the limp-raise size exposed as a constructor argument (3.5bb, the value v7 used,
so every decision is identical). Strategy: open all buttons at 2.5bb, punish limps with the top
of the range, call when discounted equity beats pot odds, bet strong hands big, re-raise only
near-nut equity; the first preflop raise is exempt from the aggression discount."""
import jax
import jax.numpy as jnp

from poker_bots.poker_game import Action, ActionData, PlayerTableState, Round

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


# ---- bot, verbatim from v8.py ----

class Bot:
    def __init__(self,
                 num_samples: int = 96,               # Monte Carlo equity samples per decision (cap 256)
                 button_open_bb: float = 2.5,         # button preflop open raise-to size (big blinds)
                 limp_raise_equity: float = 0.55,     # preflop equity to raise a button limp (probability)
                 limp_raise_bb: float = 3.5,          # raise-to size vs a button limp (big blinds)
                 three_bet_equity: float = 0.60,      # preflop equity to re-raise (probability)
                 value_bet_equity: float = 0.64,      # equity to bet when not facing a bet (probability)
                 reraise_equity: float = 0.80,        # equity to raise when facing a bet (probability)
                 bet_pot_fraction: float = 0.90,      # bet size as a fraction of the pot after calling
                 call_margin: float = 0.02,           # equity edge over pot odds required to call (probability)
                 per_raise_factor: float = 0.75,      # equity multiplied by this per opponent raise this street
                 preflop_reraise_multiple: float = 3.0):  # preflop re-raise-to as a multiple of current bet
        self.num_samples = num_samples
        self.button_open_bb = button_open_bb
        self.limp_raise_equity = limp_raise_equity
        self.limp_raise_bb = limp_raise_bb
        self.three_bet_equity = three_bet_equity
        self.value_bet_equity = value_bet_equity
        self.reraise_equity = reraise_equity
        self.bet_pot_fraction = bet_pot_fraction
        self.call_margin = call_margin
        self.per_raise_factor = per_raise_factor
        self.preflop_reraise_multiple = preflop_reraise_multiple

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        eq = equity(obs, key, self.num_samples)
        street_actions = obs.history.is_valid & (obs.history.round == obs.current_round)
        opp_raises = jnp.sum(street_actions & (obs.history.action_type == Action.RAISE) & (obs.history.player_id != 0))
        # equity vs a random hand overstates our chances against someone who has raised;
        # preflop the first raise is just the opener, so discounting starts at the second
        preflop = obs.current_round == Round.PRE_FLOP
        counted_raises = jnp.where(preflop, jnp.maximum(opp_raises - 1, 0), opp_raises)
        eq_faced = eq * jnp.power(self.per_raise_factor, counted_raises)

        # --- shared postflop / facing-bet logic: call on pot odds, raise for value ---
        facing_bet = to_call(obs) > 0
        bet_total = obs.current_bet + self.bet_pot_fraction * (obs.pot + to_call(obs))
        raise_total = jnp.where(preflop, self.preflop_reraise_multiple * obs.current_bet, bet_total)
        raise_equity = jnp.where(preflop, self.three_bet_equity,
                                 jnp.where(facing_bet, self.reraise_equity, self.value_bet_equity))
        wants_raise = obs.can_raise & (eq >= raise_equity)
        wants_call = eq_faced >= pot_odds(obs) + self.call_margin
        action = choose(wants_call, check_call(), fold_or_check(obs))
        action = choose(wants_raise, raise_to(obs, raise_total), action)

        # --- preflop as button (small blind): open every hand for a min-raise-plus ---
        blinds_only = preflop & (obs.current_bet <= obs.big_blind)
        on_button = obs.button == 0
        action = choose(blinds_only & on_button, raise_to(obs, self.button_open_bb * obs.big_blind), action)

        # --- preflop as big blind vs a limp: raise the top of the range, check the rest ---
        limped_pot = blinds_only & ~on_button
        punish_limp = choose(eq >= self.limp_raise_equity, raise_to(obs, self.limp_raise_bb * obs.big_blind), check_call())
        action = choose(limped_pot, punish_limp, action)

        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "bassander_lin_glm",
            "version": 8,
            "parent": 7,
            "lineage": "phase_2_suneater/bassander_lin_glm",
            "model": "GLM-5.3",
            "prompt_version": "botgen-v2",
            "phase": "phase_2_suneater",
            "engine": "02abc49+wt sha256:295477eff7637056",
            "params": {
                "num_samples": self.num_samples,
                "button_open_bb": self.button_open_bb,
                "limp_raise_equity": self.limp_raise_equity,
                "limp_raise_bb": self.limp_raise_bb,
                "three_bet_equity": self.three_bet_equity,
                "value_bet_equity": self.value_bet_equity,
                "reraise_equity": self.reraise_equity,
                "bet_pot_fraction": self.bet_pot_fraction,
                "call_margin": self.call_margin,
                "per_raise_factor": self.per_raise_factor,
                "preflop_reraise_multiple": self.preflop_reraise_multiple,
            },
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GLM"},
            ],
        }
