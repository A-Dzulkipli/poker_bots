# Frozen copy of kraken_gpt/v6.py, the version deployed in that lineage's README.
# The shared equity.py helpers are inlined so this file stands alone.
# Source sha256 (first 16): v6.py fd6e8fe5a11edc99, equity.py b040cb274e3b4bdd
"""Track free bets separately from reraises so broad betting need not imply broad raising."""
import jax
import jax.numpy as jnp
from poker_bots.poker_game import Action, Round

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


# ---- bot, verbatim from v6.py ----


class Bot:
    def __init__(self, num_samples=128, open_size_bb=2.5, reraise_multiple=3.0,
                 threebet_equity=0.63, light_threebet_frequency=0.15,
                 isolate_equity=0.55, value_equity=0.62, raise_equity=0.85,
                 bluff_frequency=0.35, semibluff_frequency=0.50,
                 bluff_split_equity=0.45, bluff_street_decay=1.0,
                 bet_pot_fraction=0.65, range_retention=0.58,
                 prior_street_weight=0.5, realization_fraction=0.90,
                 call_margin_equity=0.02, prior_events=20.0, prior_bet_rate=0.40,
                 prior_fold_rate=0.50, betting_adjustment=0.65, bluff_adjustment=1.25,
                 prior_raise_rate=0.15, reraise_range_retention=0.40):
        self.num_samples = num_samples  # Monte Carlo deals per decision
        self.open_size_bb = open_size_bb  # raise-to in big blinds
        self.reraise_multiple = reraise_multiple  # multiple of current bet
        self.threebet_equity = threebet_equity  # equity probabilities throughout
        self.light_threebet_frequency = light_threebet_frequency  # mixing probability
        self.isolate_equity = isolate_equity
        self.value_equity = value_equity
        self.raise_equity = raise_equity
        self.bluff_frequency = bluff_frequency  # mixing probability with air
        self.semibluff_frequency = semibluff_frequency  # mixing probability with medium equity
        self.bluff_split_equity = bluff_split_equity
        self.bluff_street_decay = bluff_street_decay  # frequency multiplier per later street
        self.bet_pot_fraction = bet_pot_fraction  # fraction of pot after calling
        self.range_retention = range_retention  # fraction of range retained per pressure unit
        self.prior_street_weight = prior_street_weight  # current-street bet equivalents
        self.realization_fraction = realization_fraction  # fraction of equity realized
        self.call_margin_equity = call_margin_equity
        self.prior_events = prior_events  # pseudo-count opportunities for each rate
        self.prior_bet_rate = prior_bet_rate  # probability of betting when checking is free
        self.prior_fold_rate = prior_fold_rate  # probability of folding when facing our aggression
        self.betting_adjustment = betting_adjustment  # retention change per unit of bet-rate deviation
        self.bluff_adjustment = bluff_adjustment  # bluff probability change per unit of fold-rate deviation
        self.prior_raise_rate = prior_raise_rate  # probability of raising when facing our bet
        self.reraise_range_retention = reraise_range_retention  # fraction of range retained per reraise
        if not 1 <= num_samples <= 256:
            raise ValueError("num_samples must be in 1..256")

    def init_state(self, key):
        return {
            "opponent_free_actions": jnp.asarray(self.prior_events, jnp.float32),
            "opponent_bets_when_free": jnp.asarray(self.prior_events * self.prior_bet_rate, jnp.float32),
            "opponent_facing_bets": jnp.asarray(self.prior_events, jnp.float32),
            "opponent_raises_facing_bets": jnp.asarray(self.prior_events * self.prior_raise_rate, jnp.float32),
            "opponent_folds_to_bets": jnp.asarray(self.prior_events * self.prior_fold_rate, jnp.float32),
        }

    # Tracking: count actual postflop opportunities, including hands where we folded.
    def observe(self, record, state, key):
        history = record.history
        opponent_action = history.is_valid[1:] & (history.player_id[1:] != 0)
        postflop_action = opponent_action & (history.round[1:] > Round.PRE_FLOP) & (history.action_type[1:] <= Action.RAISE)
        same_street = history.is_valid[:-1] & (history.round[:-1] == history.round[1:])
        our_previous_bet = (history.player_id[:-1] == 0) & (history.action_type[:-1] == Action.RAISE)
        facing_our_bet = postflop_action & same_street & our_previous_bet
        checking_free = postflop_action & ~facing_our_bet
        opponent_bet = history.action_type[1:] == Action.RAISE
        opponent_fold = history.action_type[1:] == Action.FOLD
        return {
            "opponent_free_actions": state["opponent_free_actions"] + jnp.sum(checking_free, dtype=jnp.float32),
            "opponent_bets_when_free": state["opponent_bets_when_free"] + jnp.sum(checking_free & opponent_bet, dtype=jnp.float32),
            "opponent_facing_bets": state["opponent_facing_bets"] + jnp.sum(facing_our_bet, dtype=jnp.float32),
            "opponent_raises_facing_bets": state["opponent_raises_facing_bets"] + jnp.sum(facing_our_bet & opponent_bet, dtype=jnp.float32),
            "opponent_folds_to_bets": state["opponent_folds_to_bets"] + jnp.sum(facing_our_bet & opponent_fold, dtype=jnp.float32),
        }

    def act(self, obs, state, key):
        equity_key, mix_key = jax.random.split(key)
        raw_equity = equity(obs, equity_key, self.num_samples)
        mix = jax.random.uniform(mix_key)
        preflop = obs.current_round == Round.PRE_FLOP
        on_button = obs.button == 0
        call_cost = to_call(obs)
        facing_bet = call_cost > 0
        hero_total = obs.contributed[0] + obs.stacks[0]
        opponent_total = jnp.max(jnp.where(obs.seated[1:] & obs.in_hand[1:], obs.contributed[1:] + obs.stacks[1:], 0))
        effective_total = jnp.minimum(hero_total, opponent_total)
        unmatched_bet = jnp.maximum(obs.current_bet - hero_total, 0)
        pot_after_call = jnp.maximum(obs.pot - unmatched_bet + call_cost, 1)
        call_odds = call_cost / pot_after_call

        # Betting pressure: opening preflop raises are wide; reraises narrow ranges.
        history = obs.history
        raises = history.is_valid & (history.action_type == Action.RAISE)
        opponent_raises = raises & (history.player_id != 0)
        this_street = history.round == obs.current_round
        street_bets = jnp.sum(opponent_raises & this_street)
        prior_bets = jnp.sum(opponent_raises & (history.round < obs.current_round))
        my_raises = jnp.sum(raises & this_street & (history.player_id == 0))
        preflop_pressure = jnp.where(my_raises > 0, street_bets, jnp.maximum(street_bets - 1, 0))
        pressure = jnp.where(preflop, preflop_pressure, street_bets + self.prior_street_weight * prior_bets)
        # Broad bettors get less credit; frequent folders face more mixed bluffs.
        opponent_bet_rate = state["opponent_bets_when_free"] / state["opponent_free_actions"]
        opponent_fold_to_bet_rate = state["opponent_folds_to_bets"] / state["opponent_facing_bets"]
        learned_retention = self.range_retention + self.betting_adjustment * (opponent_bet_rate - self.prior_bet_rate)
        retention = jnp.where(preflop, self.range_retention, jnp.clip(learned_retention, jnp.finfo(jnp.float32).eps, 1.0))
        # A re-raise is a different decision from betting into a check.
        opponent_raise_rate = state["opponent_raises_facing_bets"] / state["opponent_facing_bets"]
        learned_reraise_retention = self.reraise_range_retention + self.betting_adjustment * (opponent_raise_rate - self.prior_raise_rate)
        reraise_retention = jnp.where(preflop, self.range_retention, jnp.clip(learned_reraise_retention, jnp.finfo(jnp.float32).eps, 1.0))
        previous_is_our_bet = raises[:-1] & (history.player_id[:-1] == 0) & (history.round[:-1] == history.round[1:])
        follows_our_bet = jnp.concatenate((jnp.zeros(1, dtype=jnp.bool_), previous_is_our_bet))
        opponent_reraises = opponent_raises & follows_our_bet
        current_reraises = jnp.sum(opponent_reraises & this_street)
        previous_reraises = jnp.sum(opponent_reraises & (history.round < obs.current_round))
        reraise_pressure = current_reraises + self.prior_street_weight * previous_reraises
        ordinary_pressure = jnp.maximum(pressure - reraise_pressure, 0)
        remaining_range = jnp.power(retention, ordinary_pressure) * jnp.power(reraise_retention, reraise_pressure)
        # Remove the weakest part of a random range, preserving equity near the nuts.
        adjusted_equity = jnp.maximum(raw_equity + remaining_range - 1, 0) / remaining_range
        opponent_can_bet = jnp.any(obs.seated[1:] & obs.in_hand[1:] & (obs.stacks[1:] > 0))
        future_betting = (obs.current_round < Round.RIVER) & (call_cost < obs.stacks[0]) & opponent_can_bet
        realization = jnp.where(future_betting, self.realization_fraction, 1.0)
        wants_call = adjusted_equity * realization >= call_odds + self.call_margin_equity
        passive = choose(wants_call, check_call(), fold_or_check(obs))

        # Preflop as button: open every hand, then respect reraises.
        unopened = preflop & (obs.current_bet <= obs.big_blind)
        opening_total = jnp.minimum(self.open_size_bb * obs.big_blind, effective_total)
        action = choose(unopened & on_button & obs.can_raise, raise_to(obs, opening_total), passive)

        # Preflop as big blind: isolate limps and mix light three-bets with value.
        single_open = jnp.sum(raises & (history.round == Round.PRE_FLOP)) == 1
        light_threebet = single_open & wants_call & (mix < self.light_threebet_frequency)
        threebet = preflop & ~on_button & facing_bet & ((adjusted_equity >= self.threebet_equity) | light_threebet)
        isolate = unopened & ~on_button & (raw_equity >= self.isolate_equity)
        reraise_total = jnp.minimum(self.reraise_multiple * obs.current_bet, effective_total)
        action = choose((threebet | isolate) & obs.can_raise, raise_to(obs, reraise_total), action)

        # Postflop when checked to: bet value and mix air and medium-equity bluffs.
        street_step = jnp.clip(obs.current_round - Round.FLOP, 0, 2)
        bluff_scale = jnp.power(self.bluff_street_decay, street_step)
        bluff_rate = jnp.where(raw_equity < self.bluff_split_equity, self.bluff_frequency, self.semibluff_frequency)
        bluff_shift = self.bluff_adjustment * (opponent_fold_to_bet_rate - self.prior_fold_rate)
        learned_bluff_rate = jnp.clip(bluff_rate + bluff_shift, 0.0, 1.0)
        bluff = mix < learned_bluff_rate * bluff_scale
        value_bet = raw_equity >= self.value_equity
        wants_bet = ~facing_bet & (value_bet | bluff)

        # Postflop facing a bet: raise value only after accounting for pressure.
        value_raise = facing_bet & (adjusted_equity >= self.raise_equity)
        bet_total = jnp.minimum(obs.current_bet + self.bet_pot_fraction * pot_after_call, effective_total)
        postflop_raise = ~preflop & obs.can_raise & (wants_bet | value_raise)
        action = choose(postflop_raise, raise_to(obs, bet_total), action)
        return action, state

    def metadata(self):
        return {
            "bot_name": "kraken_gpt",
            "version": 6,
            "parent": 3,
            "lineage": "phase_4_sea_monsters/kraken_gpt",
            "model": "GPT-6 (Codex; exact serving variant unavailable)",
            "prompt_version": "botgen-v4",
            "phase": "phase_4_sea_monsters",
            "engine": "decafb9 sha256:295477eff7637056",
            "params": dict(vars(self)),
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GPT (Codex)"}
            ],
        }
