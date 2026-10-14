# Frozen copy of scherezard_glm/v5.py, the version deployed in that lineage's README.
# The shared equity.py helpers are inlined so this file stands alone.
# Source sha256 (first 16): v5.py af41011818ad0824, equity.py b040cb274e3b4bdd
"""scherezard_glm v5 — decisions against an explicit, action-narrowed opponent range.

The opponent's range is a (52,52) weight matrix. The preflop line fixes its shape (opens keep
strong hands, limps keep weak ones, calls keep a band below what would have raised); every
postflop villain action multiplies it by a likelihood that rises with how strongly the holding
connects with the board and with the bet size. Holdings are sampled from the final range and
the hero's made hand is compared against them; bet/call/fold follows that range equity.
"""
import jax
import jax.numpy as jnp
from poker_bots.hand_eval import evaluate_hands
from poker_bots.poker_game import Action, ActionData

# ---- shared helpers, verbatim from equity.py ----
import jax
import jax.numpy as jnp

from poker_bots.hand_eval import evaluate_hands
from poker_bots.poker_game import Action, ActionData, PlayerTableState


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

CARD = jnp.arange(52)  # static card table; rank = c % 13, suit = c // 13


def preflop_quality():
    """(52,52) preflop hand quality, ~[0, 1.4]: pairs and big suited cards high, trash low."""
    rank, suit = CARD % 13, CARD // 13
    hi = jnp.maximum(rank[:, None], rank[None, :])
    lo = jnp.minimum(rank[:, None], rank[None, :])
    pair, suited = rank[:, None] == rank[None, :], suit[:, None] == suit[None, :]
    gap = jnp.where(pair, 0, hi - lo)
    pair_bonus = jnp.where(pair, 0.45 + hi / 13 * 0.35, 0.0)
    return hi / 12 * 0.42 + lo / 12 * 0.18 + pair_bonus + suited * 0.08 - jnp.minimum(gap, 4) * 0.022


def board_connection(obs):
    """(3, 52, 52) per street: how strongly each holding connects with the board dealt so far."""
    on_board = obs.board >= 0
    bcard = jnp.where(on_board, obs.board, 0)
    rank_count = jnp.zeros(13).at[bcard % 13].add(on_board)
    suit_count = jnp.zeros(4).at[bcard // 13].add(on_board)
    board_top = jnp.max(jnp.where(on_board, bcard % 13, -1))
    rank, suit = CARD % 13, CARD // 13
    card_value = (0.30 * (rank_count[rank] > 0)          # pairs the board
                  + 0.90 * (suit_count[suit] >= 3)       # made flush / heavy draw
                  + 0.35 * (suit_count[suit] == 2)       # flush draw
                  + 0.12 * (rank > board_top))           # overcard
    pocket_pair_over_board = (rank[:, None] == rank[None, :]) & (rank[:, None] > board_top)
    connection = card_value[:, None] + card_value[None, :] + 0.45 * pocket_pair_over_board
    return jnp.tile(jnp.minimum(connection, 1.4), (3, 1, 1))


def preflop_range(obs):
    """Weights over the villain's holdings given only the preflop line, as soft score bands."""
    h = obs.history
    pre = h.is_valid & (h.round == 0) & (h.player_id == 1)
    raises = pre & (h.action_type == Action.RAISE)
    open_bb = jnp.max(jnp.where(raises, h.amount, 0)) / obs.big_blind
    limp = (obs.button == 1) & (jnp.sum(pre & (h.action_type == Action.CHECK_CALL)) >= 1) & (jnp.sum(raises) == 0)
    flat = (jnp.sum(raises) == 0) & ~limp  # big blind checking or calling an open
    quality = preflop_quality()
    lower = jnp.where(jnp.sum(raises) >= 2, 0.48, jnp.where(jnp.sum(raises) == 1, 0.26 + 0.02 * (open_bb - 2.5),
                   jnp.where(limp, 0.08, jnp.where(flat, 0.30, 0.0))))
    trim = jnp.where(jnp.sum(raises) == 0, 0.62, 1.5)  # passive lines would have raised premiums
    weights = jax.nn.sigmoid((quality - lower) / 0.12) * (0.2 + 0.8 * jax.nn.sigmoid((trim - quality) / 0.15))
    return 0.03 + 0.97 * weights  # small floor: rare slowplays and wild lines


def narrow_postflop(obs, weights):
    """Multiply range weights by a likelihood for every villain action on every later street."""
    h = obs.history
    connection = board_connection(obs)
    street = jnp.clip(h.round - 1, 0, 2)
    size_bb = h.amount / obs.big_blind
    bet_pull = (0.30 + 0.07 * jnp.clip(size_bb, 0, 12))[:, None, None]  # bigger bets lean on connected hands
    is_mine = h.is_valid & (h.player_id == 1) & (h.round >= 1)
    conn = connection[street]
    raise_slot = (h.action_type == Action.RAISE)[:, None, None]
    check_slot = ((h.action_type == Action.CHECK_CALL) & (h.amount == 0))[:, None, None]
    per_slot = jnp.where(raise_slot, 0.05 + bet_pull * conn,
               jnp.where(check_slot, 0.90 - 0.35 * conn, 0.25 + 0.55 * conn))
    keep = jnp.where(is_mine[:, None, None], per_slot, 1.0)
    return weights * jnp.prod(keep, axis=0)


def range_equity(obs, key, weights, num_samples):
    """Hero's win share of current made hands against holdings sampled from the narrowed range."""
    flat = weights.reshape(-1)
    priority = jax.random.uniform(key, flat.shape) ** (1.0 / jnp.maximum(flat, 1e-6))
    top = jnp.argsort(priority)[::-1][:num_samples]
    villain_cards = jnp.stack([top // 52, top % 52], axis=1)
    board = jnp.maximum(obs.board, 0)

    def best_strength(holes):
        hands = jnp.concatenate([holes, jnp.broadcast_to(board, (holes.shape[0], 5))], axis=1)
        by_street = jnp.stack([evaluate_hands(hands[:, :5]), evaluate_hands(hands[:, :6]), evaluate_hands(hands)])
        return by_street[jnp.clip(obs.current_round - 1, 0, 2)]

    villain_strength = best_strength(villain_cards)
    hero_strength = best_strength(obs.hole_cards[None])[0]
    win = jnp.mean(jnp.where(villain_strength > hero_strength, 0.0, jnp.where(villain_strength == hero_strength, 0.5, 1.0)))
    on_board = obs.board >= 0
    suit_count = jnp.zeros(4).at[jnp.where(on_board, obs.board, 0) // 13].add(on_board)
    villain_flush_draw = (villain_cards[:, 0] // 13 == villain_cards[:, 1] // 13) & (suit_count[villain_cards[:, 0] // 13] == 2)
    streets_left = jnp.clip(3 - obs.current_round, 0, 2)
    behind_draw_share = jnp.mean(villain_flush_draw & (villain_strength < hero_strength))
    return win, behind_draw_share, suit_count


class Bot:
    def __init__(self, open_min_quality=0.26, threebet_min_quality=0.48, defend_min_quality=0.30,
                 value_bet_equity=0.72, bluff_frequency=0.45, bluff_equity_ceiling=0.40,
                 raise_equity=0.80, call_margin=0.02, villain_samples=80, uniform_range=False):
        self.open_min_quality, self.threebet_min_quality, self.defend_min_quality = open_min_quality, threebet_min_quality, defend_min_quality
        self.value_bet_equity, self.bluff_frequency, self.bluff_equity_ceiling = value_bet_equity, bluff_frequency, bluff_equity_ceiling
        self.raise_equity, self.call_margin, self.villain_samples = raise_equity, call_margin, villain_samples
        self.uniform_range = uniform_range

    def init_state(self, key):
        return None

    def act(self, obs, state, key):
        sample_key, bluff_key, semibluff_key = jax.random.split(key, 3)
        quality = preflop_quality()
        on_board = obs.board >= 0
        dead = jnp.zeros(52, bool).at[obs.hole_cards].set(True).at[jnp.where(on_board, obs.board, 0)].max(on_board)
        weights = narrow_postflop(obs, preflop_range(obs)) * ~dead[:, None] * ~dead[None, :]
        weights = jnp.where(self.uniform_range, jnp.ones_like(weights), weights)  # ablation: range off
        villain_quality = jnp.sum(weights * quality) / jnp.maximum(jnp.sum(weights), 1.0)

        # ---- preflop: thresholds on hand quality, shifted by how strong the villain's line is
        hero_quality = quality[obs.hole_cards[0], obs.hole_cards[1]]
        h, tc = obs.history, to_call(obs)
        raises = h.is_valid & (h.round == 0) & (h.player_id == 1) & (h.action_type == Action.RAISE)
        open_bb = jnp.max(jnp.where(raises, h.amount, 0)) / obs.big_blind
        unopened = (obs.button == 0) & (jnp.sum(h.is_valid & (h.player_id == 1)) == 0)
        limped_to_me = (obs.button == 0) & (jnp.sum(raises) == 0) & (jnp.sum(h.is_valid & (h.player_id == 1) & (h.action_type == Action.CHECK_CALL)) > 0)
        call_floor = jnp.where(unopened | limped_to_me, 0.0, self.defend_min_quality + 0.02 * open_bb)
        raise_bar = jnp.where(unopened | limped_to_me, self.open_min_quality,
                     jnp.where(jnp.sum(raises) == 0, self.open_min_quality,
                     jnp.where(jnp.sum(raises) == 1, self.threebet_min_quality + 0.03 * (open_bb - 3.0), 0.95)))
        pot_after_call = obs.pot + jnp.minimum(tc, obs.stacks[0])
        preflop_raise = raise_to(obs, jnp.where(obs.current_bet > 0, obs.current_bet + 0.75 * pot_after_call, 3 * obs.big_blind))
        shove_equity = 0.5 + 0.5 * (hero_quality - villain_quality)
        must_stand = jnp.where(tc >= obs.stacks[0], pot_odds(obs) + self.call_margin, call_floor)
        preflop_action = choose(hero_quality >= raise_bar, preflop_raise,
                       choose(shove_equity > must_stand, check_call(), fold_or_check(obs)))

        # ---- postflop: equity against the narrowed range, draws credited arithmetically
        win, behind_draw_share, suit_count = range_equity(obs, sample_key, weights, self.villain_samples)
        hero_suit = obs.hole_cards[0] // 13
        board_top = jnp.max(jnp.where(obs.board >= 0, obs.board % 13, -1))
        hero_outs = 9 * ((obs.hole_cards[0] // 13 == obs.hole_cards[1] // 13) & (suit_count[hero_suit] == 2))
        hero_outs = hero_outs + 3 * jnp.sum(obs.hole_cards % 13 > board_top)
        streets_left = jnp.clip(3 - obs.current_round, 0, 2)
        equity = jnp.clip(win + (1 - win) * jnp.minimum(hero_outs, 18) * 0.021 * streets_left * 0.85
                          - behind_draw_share * 9 * 0.021 * streets_left, 0.0, 1.0)
        passive_line = jnp.sum(h.is_valid & (h.player_id == 1) & (h.round >= 1) & (h.action_type == Action.RAISE)) == 0
        value_size = obs.current_bet + (0.45 + 0.55 * (equity - self.value_bet_equity) / 0.28) * pot_after_call
        want_bluff = (jax.random.uniform(bluff_key) < self.bluff_frequency) & (equity < self.bluff_equity_ceiling) & passive_line
        facing_bet = tc > 0
        unopposed = choose(equity > self.value_bet_equity, raise_to(obs, value_size),
                  choose(want_bluff & (obs.can_raise > 0), raise_to(obs, 0.6 * pot_after_call), check_call()))
        opposed = choose(equity > self.raise_equity, raise_to(obs, obs.current_bet + 0.9 * pot_after_call),
               choose(equity > pot_odds(obs) + self.call_margin, check_call(),
               choose(want_bluff & (jax.random.uniform(semibluff_key) < 0.15) & (obs.can_raise > 0), raise_to(obs, obs.current_bet + 0.8 * pot_after_call), fold_or_check(obs))))
        action = choose(obs.current_round == 0, preflop_action, choose(facing_bet, opposed, unopposed))
        return action, state

    def metadata(self):
        return {
            "bot_name": "scherezard_glm", "version": 5, "parent": 1, "lineage": "phase_6_trails/scherezard_glm",
            "model": "GLM-5.3", "prompt_version": "botgen-v6", "phase": "phase_6_trails",
            "engine": "decafb9 sha256:295477eff7637056",
            "params": {"open_min_quality": self.open_min_quality, "threebet_min_quality": self.threebet_min_quality,
                       "defend_min_quality": self.defend_min_quality, "value_bet_equity": self.value_bet_equity,
                       "bluff_frequency": self.bluff_frequency, "bluff_equity_ceiling": self.bluff_equity_ceiling,
                       "raise_equity": self.raise_equity, "call_margin": self.call_margin, "villain_samples": self.villain_samples,
                       "uniform_range": self.uniform_range},
            "contributions": [{"role": "requester", "who": "a"}, {"role": "generator", "interface": "GLM"}],
        }
