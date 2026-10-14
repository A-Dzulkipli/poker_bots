# Frozen copy of windrunner_glm/v6.py, the version deployed in that lineage's README.
# The shared equity.py helpers are inlined so this file stands alone.
# Source sha256 (first 16): v6.py 8f0cfc703782e562, equity.py b040cb274e3b4bdd
"""v6: v5 with the aggression discount pushed deeper still (0.58 per unit of betting pressure).

Parent weakness: v3 still called postflop bets with roughly break-even equity, which loses
against value-only opponents whose betting ranges sit well above average (the pooled valka
result was about -14 bb/100 across three seeds). A deeper discount folds the second-best hands
that a value bettor collects on, at the cost of some forfeited equity against the always-betting
probe bot.
"""
import jax
import jax.numpy as jnp

from poker_bots.poker_game import Action, ActionData, PlayerTableState, Round

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


# ---- bot, verbatim from v6.py ----

class Bot:
    def __init__(self,
                 num_samples=128,             # Monte Carlo equity samples per decision (cap 256)
                 open_size_bb=2.5,            # button open raise-to, big blinds
                 preflop_reraise_multiple=3.0,  # raise-to as a multiple of the current bet preflop
                 three_bet_equity=0.63,       # equity vs a random hand to 3-bet an open (probability)
                 light_three_bet_freq=0.15,   # 3-bet bluffs with hands that would call (probability)
                 limp_raise_equity=0.55,      # equity to raise a button limp (probability)
                 value_bet_equity=0.62,       # equity to bet postflop when checked to (probability)
                 bluff_frequency=0.33,        # flop bluff rate with air when checked to (probability)
                 semi_bluff_frequency=0.50,   # flop bluff rate with draws or medium hands (probability)
                 bluff_street_decay=0.55,     # multiply the bluff rate by this per street past the flop
                 bluff_equity_split=0.45,     # boundary between air and semi-bluff (probability)
                 bet_pot_fraction=0.65,       # postflop bet size, fraction of pot after calling
                 reraise_equity=0.80,         # equity to raise a postflop bet (probability)
                 call_margin=0.02,            # equity edge over pot odds required to call (probability)
                 aggression_discount=0.58,    # equity multiplier per unit of betting pressure
                 prior_street_weight=0.5,     # weight of earlier-street bets in the pressure count
                 realization_factor=0.90):    # equity realization before the river
        self.num_samples = num_samples
        self.open_size_bb = open_size_bb
        self.preflop_reraise_multiple = preflop_reraise_multiple
        self.three_bet_equity = three_bet_equity
        self.light_three_bet_freq = light_three_bet_freq
        self.limp_raise_equity = limp_raise_equity
        self.value_bet_equity = value_bet_equity
        self.bluff_frequency = bluff_frequency
        self.semi_bluff_frequency = semi_bluff_frequency
        self.bluff_street_decay = bluff_street_decay
        self.bluff_equity_split = bluff_equity_split
        self.bet_pot_fraction = bet_pot_fraction
        self.reraise_equity = reraise_equity
        self.call_margin = call_margin
        self.aggression_discount = aggression_discount
        self.prior_street_weight = prior_street_weight
        self.realization_factor = realization_factor

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        eq_key, roll_key = jax.random.split(key)
        win_prob = equity(obs, eq_key, self.num_samples)
        roll = jax.random.uniform(roll_key)
        preflop = obs.current_round == Round.PRE_FLOP
        on_button = obs.button == 0
        facing_bet = to_call(obs) > 0
        unopened = preflop & (obs.current_bet <= obs.big_blind)

        # betting pressure: each opponent bet this street is a full unit, earlier streets half
        opp = obs.history.is_valid & (obs.history.player_id != 0)
        is_bet = opp & (obs.history.action_type == Action.RAISE)
        street_bets = jnp.sum(is_bet & (obs.history.round == obs.current_round))
        prior_bets = jnp.sum(is_bet & (obs.history.round < obs.current_round))
        my_raises = jnp.sum(obs.history.is_valid & (obs.history.player_id == 0)
                            & (obs.history.action_type == Action.RAISE) & (obs.history.round == obs.current_round))
        # preflop an open is wide, so only a re-raise counts; once I have raised, any opponent raise is one
        preflop_units = jnp.where(my_raises > 0, street_bets, jnp.maximum(street_bets - 1, 0))
        units = jnp.where(preflop, preflop_units, street_bets + self.prior_street_weight * prior_bets)
        discounted = win_prob * jnp.power(self.aggression_discount, units)

        # passive baseline: call when discounted equity beats the price, check when free
        opponent_can_bet = jnp.any(obs.seated[1:] & obs.in_hand[1:] & (obs.stacks[1:] > 0))
        future_betting = (obs.current_round < Round.RIVER) & (to_call(obs) < obs.stacks[0]) & opponent_can_bet
        realized = jnp.where(future_betting, self.realization_factor, 1.0)
        wants_call = discounted * realized >= pot_odds(obs) + self.call_margin
        passive = choose(wants_call, check_call(), fold_or_check(obs))

        # preflop as button: open every hand; facing a 3-bet, fall through to the call test
        action = choose(unopened & on_button, raise_to(obs, self.open_size_bb * obs.big_blind), passive)

        # preflop as big blind: 3-bet the top plus light bluffs, punish limps with the strong range
        three_bet = preflop & ~on_button & facing_bet & ((win_prob >= self.three_bet_equity)
                                                         | (wants_call & (roll < self.light_three_bet_freq)))
        raise_limp = unopened & ~on_button & (win_prob >= self.limp_raise_equity)
        reraise_total = self.preflop_reraise_multiple * obs.current_bet
        action = choose((three_bet | raise_limp) & obs.can_raise, raise_to(obs, reraise_total), action)

        # postflop when checked to: bet for value; bluff air and draws at a rate that decays by street
        bet_total = obs.current_bet + self.bet_pot_fraction * (obs.pot + to_call(obs))
        street_step = jnp.clip(obs.current_round - 1, 0, 2)  # 0 flop, 1 turn, 2 river
        bluff_scale = jnp.power(self.bluff_street_decay, street_step)
        value_bet = win_prob >= self.value_bet_equity
        air_bluff = (win_prob < self.bluff_equity_split) & (roll < self.bluff_frequency * bluff_scale)
        semi_bluff = (win_prob >= self.bluff_equity_split) & (win_prob < self.value_bet_equity) \
            & (roll < self.semi_bluff_frequency * bluff_scale)
        stab = (~preflop) & (~facing_bet) & obs.can_raise & (value_bet | air_bluff | semi_bluff)
        action = choose(stab, raise_to(obs, bet_total), action)

        # postflop facing a bet: raise only near the top of the range
        value_raise = (~preflop) & facing_bet & obs.can_raise & (win_prob >= self.reraise_equity)
        action = choose(value_raise, raise_to(obs, bet_total), action)
        return action, state

    def metadata(self) -> dict:
        return {
            "bot_name": "windrunner_glm",
            "version": 6,
            "parent": 5,
            "lineage": "phase_3_dota/windrunner_glm",
            "model": "GLM-5.3 (account:zai-individual-coding-plan/GLM-5.3)",
            "prompt_version": "botgen-v3",
            "phase": "phase_3_dota",
            "engine": "02abc49+wt sha256:295477eff7637056",
            "params": {
                "num_samples": self.num_samples,
                "open_size_bb": self.open_size_bb,
                "preflop_reraise_multiple": self.preflop_reraise_multiple,
                "three_bet_equity": self.three_bet_equity,
                "light_three_bet_freq": self.light_three_bet_freq,
                "limp_raise_equity": self.limp_raise_equity,
                "value_bet_equity": self.value_bet_equity,
                "bluff_frequency": self.bluff_frequency,
                "semi_bluff_frequency": self.semi_bluff_frequency,
                "bluff_street_decay": self.bluff_street_decay,
                "bluff_equity_split": self.bluff_equity_split,
                "bet_pot_fraction": self.bet_pot_fraction,
                "reraise_equity": self.reraise_equity,
                "call_margin": self.call_margin,
                "aggression_discount": self.aggression_discount,
                "prior_street_weight": self.prior_street_weight,
                "realization_factor": self.realization_factor,
            },
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GLM"},
            ],
        }
