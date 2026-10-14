"""Aggressor probe: punishes over-folding.

It opens every button, raises every limp, and bets whenever checking is free, on every street.
Facing a bet or raise it never raises; it continues only with equity (against a random hand) of at
least `defend_equity` and enough to beat the pot odds, and folds otherwise. Its profit therefore
comes almost entirely from opponents folding to its bets: a bot that folds too often loses chips
to it, and a bot that defends correctly beats it, because most of its bets are air.

Self-contained (imports only jax and poker_bots), so it can be copied as an opponent file.
"""
import jax
import jax.numpy as jnp

from poker_bots.hand_eval import evaluate_hands
from poker_bots.poker_game import Action, ActionData, PlayerTableState, Round


def equity(obs: PlayerTableState, key: jax.Array, num_samples: int) -> jax.Array:
    """Monte Carlo share of the pot won at showdown against opponents holding uniformly random cards."""
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


def action(kind: int, value: jax.Array = 0) -> ActionData:
    return ActionData(type=jnp.asarray(kind, dtype=jnp.int32), value=jnp.asarray(jnp.round(value), dtype=jnp.int32))


class Bot:
    def __init__(self, open_size_bb: float = 2.5, bet_pot_fraction: float = 0.67, defend_equity: float = 0.65,
                 num_samples: int = 64):
        self.open_size_bb = open_size_bb            # preflop open / limp-raise size, raise-to in big blinds
        self.bet_pot_fraction = bet_pot_fraction    # postflop bet size as a fraction of the pot
        self.defend_equity = defend_equity          # minimum equity probability to continue facing aggression
        self.num_samples = num_samples              # Monte Carlo samples for call decisions

    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array):
        to_call = jnp.minimum(obs.current_bet - obs.contributed[0], obs.stacks[0])
        preflop = obs.current_round == Round.PRE_FLOP
        unopened = preflop & (obs.current_bet <= obs.big_blind)
        # first in on the button, a limped pot in the big blind, or checked to postflop: always bet
        attack = (unopened & (obs.button == 0)) | (to_call == 0)
        bet_total = jnp.where(preflop, self.open_size_bb * obs.big_blind, self.bet_pot_fraction * obs.pot)
        bet = action(Action.RAISE, jnp.clip(bet_total, obs.raise_lower_bound, obs.raise_upper_bound))
        # facing real aggression: continue only with enough equity against a random hand; never fold a free check
        price = to_call / jnp.maximum(obs.pot + to_call, 1)
        win_prob = equity(obs, key, self.num_samples)
        continues = (to_call == 0) | ((win_prob >= price) & (win_prob >= self.defend_equity))
        call_or_fold = jnp.where(continues, Action.CHECK_CALL, Action.FOLD)
        response = action(call_or_fold)
        chosen = jax.tree.map(lambda a, b: jnp.where(attack & obs.can_raise, a, b), bet, response)
        return chosen, state

    def metadata(self) -> dict:
        return {"bot_name": "aggressor", "kind": "bot", "role": "probe",
                "description": "opens every button, bets whenever checked to, continues on pot odds above defend_equity",
                "params": {"open_size_bb": self.open_size_bb, "bet_pot_fraction": self.bet_pot_fraction,
                           "defend_equity": self.defend_equity, "num_samples": self.num_samples}}
