# Frozen copy of meepo_claude/v5.py, the version deployed in that lineage's README.
# The shared equity.py helpers are inlined so this file stands alone.
# Source sha256 (first 16): v5.py 92a678bd3fa7a25e, equity.py b040cb274e3b4bdd
"""meepo_claude v5: v2 with a smaller range discount for preflop raises.

Change from v2: heads-up button opens are wide, so a preflop raise removes only
preflop_raise_discount of the range (not the postflop 0.25); the big blind defends more.
v2 change from v1: with weak hands, bet as a bluff at a fixed frequency when betting is free,
so opponents who fold to bets lose pots they would have checked down.
v1 summary:

Equity is measured against a random hand, then discounted for every bet or raise the
opponent has made this hand (their range is stronger than random). Preflop the button
opens wide and the big blind defends by pot odds; postflop the bot bets for value,
checks medium hands, and calls or folds on pot odds against the discounted equity.
"""
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


# ---- bot, verbatim from v5.py ----

class Bot:
    def __init__(self, num_samples=192, open_equity=0.42, preflop_raise_equity=0.62,
                 limp_raise_equity=0.55, value_bet_equity=0.64, value_raise_equity=0.78,
                 street_raise_discount=0.25, earlier_raise_discount=0.12, preflop_raise_discount=0.10, call_margin=0.02,
                 open_size_bb=2.5, reraise_multiple=3.0, bet_pot_fraction=0.66,
                 bluff_max_equity=0.35, bluff_frequency=0.35):
        self.num_samples = num_samples                    # Monte Carlo samples per decision (cap 256)
        self.open_equity = open_equity                    # button opens with equity vs random >= this
        self.preflop_raise_equity = preflop_raise_equity  # 3-bet / 4-bet with discounted equity >= this
        self.limp_raise_equity = limp_raise_equity        # big blind raises a limp with equity >= this
        self.value_bet_equity = value_bet_equity          # postflop bet when checked to, discounted equity
        self.value_raise_equity = value_raise_equity      # postflop raise facing a bet, discounted equity
        self.street_raise_discount = street_raise_discount    # weakest range fraction removed per raise this street
        self.earlier_raise_discount = earlier_raise_discount  # same, per raise on an earlier street
        self.preflop_raise_discount = preflop_raise_discount  # same, per preflop raise (wide heads-up opens)
        self.call_margin = call_margin                    # equity probability required above pot odds
        self.open_size_bb = open_size_bb                  # button open, raise-to in big blinds
        self.reraise_multiple = reraise_multiple          # preflop re-raise = this times the current bet
        self.bet_pot_fraction = bet_pot_fraction          # postflop bet/raise size as a pot fraction
        self.bluff_max_equity = bluff_max_equity          # hands below this equity may bluff (they cannot win by checking)
        self.bluff_frequency = bluff_frequency            # probability of bluffing such a hand when checked to

    def init_state(self, key):
        return None

    # ---- shared: equity against the opponent's likely range ----
    def range_equity(self, obs, key):
        h = obs.history
        their_raises = h.is_valid & (h.player_id == 1) & (h.action_type == Action.RAISE)
        preflop_raises = jnp.sum(their_raises & (h.round == Round.PRE_FLOP))
        postflop = their_raises & (h.round > Round.PRE_FLOP)
        this_street = jnp.sum(postflop & (h.round == obs.current_round))
        earlier = jnp.sum(postflop & (h.round < obs.current_round))
        # each raise removes the weakest part of their range; what remains is what we must beat
        remaining = (1 - self.street_raise_discount) ** this_street * (1 - self.earlier_raise_discount) ** earlier
        remaining = remaining * (1 - self.preflop_raise_discount) ** preflop_raises
        raw = equity(obs, key, self.num_samples)
        return jnp.clip((raw - (1 - remaining)) / remaining, 0.0, 1.0), raw

    def bet(self, obs, pot_fraction):
        pot_after_call = obs.pot + to_call(obs)
        return raise_to(obs, obs.current_bet + pot_fraction * pot_after_call)

    def call_or_fold(self, obs, win_prob):
        # call when equity beats the price; never fold when checking is free
        return choose(win_prob >= pot_odds(obs) + self.call_margin, check_call(), fold_or_check(obs))

    # ---- preflop as button, first to act ----
    def button_open(self, obs, raw):
        return choose(raw >= self.open_equity, raise_to(obs, self.open_size_bb * obs.big_blind), fold_or_check(obs))

    # ---- preflop facing a limp or a raise ----
    def preflop_response(self, obs, win_prob):
        facing_limp = obs.current_bet <= obs.big_blind
        threshold = jnp.where(facing_limp, self.limp_raise_equity, self.preflop_raise_equity)
        reraise = raise_to(obs, self.reraise_multiple * jnp.maximum(obs.current_bet, obs.big_blind))
        return choose(win_prob >= threshold, reraise, self.call_or_fold(obs, win_prob))

    # ---- postflop, checked to us or first to act ----
    def postflop_unopened(self, obs, win_prob, key):
        value = win_prob >= self.value_bet_equity
        # bluff with hands too weak to win at showdown; medium hands check and keep their equity
        bluff = (win_prob < self.bluff_max_equity) & (jax.random.uniform(key) < self.bluff_frequency)
        return choose(value | bluff, self.bet(obs, self.bet_pot_fraction), check_call())

    # ---- postflop facing a bet ----
    def postflop_facing_bet(self, obs, win_prob):
        raise_ok = win_prob >= self.value_raise_equity
        return choose(raise_ok, self.bet(obs, self.bet_pot_fraction), self.call_or_fold(obs, win_prob))

    def act(self, obs: PlayerTableState, state, key):
        equity_key, mix_key = jax.random.split(key)
        win_prob, raw = self.range_equity(obs, equity_key)
        preflop = obs.current_round == Round.PRE_FLOP
        button_first_in = preflop & (obs.button == 0) & (obs.current_bet <= obs.big_blind) & (to_call(obs) > 0)
        preflop_action = choose(button_first_in, self.button_open(obs, raw), self.preflop_response(obs, win_prob))
        facing_bet = to_call(obs) > 0
        postflop_action = choose(facing_bet, self.postflop_facing_bet(obs, win_prob),
                                 self.postflop_unopened(obs, win_prob, mix_key))
        return choose(preflop, preflop_action, postflop_action), state

    def metadata(self) -> dict:
        return {
            "bot_name": "meepo_claude", "version": 5, "parent": 2,
            "lineage": "phase_3_dota/meepo_claude", "model": "Claude Opus 5.5 (claude-opus-5-5)",
            "prompt_version": "botgen-v3", "phase": "phase_3_dota",
            "engine": "02abc49+wt sha256:295477eff7637056",
            "params": dict(vars(self)),
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "Claude desktop (Code)"}
            ],
        }
