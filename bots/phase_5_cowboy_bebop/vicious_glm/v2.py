"""vicious_glm v2 — same equity core as v1, but it stops playing weak-passive.

One closely related group of changes: respect that a bettor's range beats random cards (much
larger equity discount when facing bets), and take the initiative instead of checking behind
(open-raise nearly every button instead of limping, bet more marginal hands, bluff more air).
"""
import jax
import jax.numpy as jnp
from poker_bots.hand_eval import evaluate_hands
from poker_bots.poker_game import Action, ActionData
from .equity import check_call, choose, equity, fold_or_check, hand_category, made_hand, pot_odds, raise_to, to_call

# Rank windows that can host a straight: 2-6 through T-A (rows 0-8) and the wheel A-5 (row 9).
WINDOWS = jnp.zeros((10, 13), jnp.int32)
WINDOWS = WINDOWS.at[jnp.arange(9)[:, None], jnp.arange(9)[:, None] + jnp.arange(5)[None, :]].set(1)
WINDOWS = WINDOWS.at[9, jnp.array([12, 0, 1, 2, 3])].set(1)


class Bot:
    def __init__(self,
                 open_equity: float = 0.33,            # minimum equity to open-raise on the button (raise ~always)
                 limp_raise_equity: float = 0.50,      # minimum equity to raise a limped button as big blind
                 three_bet_equity: float = 0.60,       # minimum equity to three-bet or four-bet
                 defend_equity: float = 0.42,          # minimum equity to call a preflop raise
                 open_size_bb: float = 2.5,            # open-raise size in big blinds
                 three_bet_multiplier: float = 2.5,    # raise increment over current bet, as a multiple of it
                 value_bet_equity: float = 0.56,       # minimum equity to bet when checked to us
                 monster_equity: float = 0.78,         # minimum equity to raise a bet for value
                 base_bet_fraction: float = 0.55,      # bet size as a fraction of pot on dry boards
                 wet_bet_extra: float = 0.30,          # extra bet fraction per unit of board wetness
                 raise_face_base: float = 0.65,        # raise sizing vs a bet, fraction of pot after call
                 wet_raise_extra: float = 0.15,        # extra raise fraction per unit of wetness
                 call_margin: float = 0.03,            # equity cushion over pot odds to call a bet
                 wet_call_extra: float = 0.10,         # extra cushion per unit wetness times faced bet fraction
                 paired_call_relief: float = 0.02,     # cushion removed on paired boards
                 bet_discount: float = 0.22,           # equity discount per unit of faced bet fraction
                 semi_bluff_outs: int = 6,             # improving cards needed to semi-bluff when checked to
                 semi_bluff_freq: float = 0.75,        # probability of betting a qualifying draw
                 semi_bluff_min_equity: float = 0.35,  # minimum equity to bet or raise a draw
                 air_bluff_fold_scale: float = 1.00,   # air-bluff probability per unit of opponent fold rate
                 air_bluff_max_equity: float = 0.40,   # equity above which a hand is too good to bluff
                 draw_raise_outs: int = 8,             # improving cards needed to raise a bet as a semi-bluff
                 draw_call_credit: float = 0.05,       # equity cushion forgiven when holding a big draw
                 texture_effect_scale: float = 1.0,    # 0 disables every board-texture effect (ablation)
                 draw_effect_scale: float = 1.0,       # 0 disables every draw-based effect (ablation)
                 prior_hands: float = 8.0,             # pseudo-hands behind the fold-rate prior
                 prior_fold_rate: float = 0.45,        # assumed opponent fold rate before evidence
                 equity_samples: int = 128):           # Monte Carlo samples per decision (cap 256)
        self.open_equity = open_equity
        self.limp_raise_equity = limp_raise_equity
        self.three_bet_equity = three_bet_equity
        self.defend_equity = defend_equity
        self.open_size_bb = open_size_bb
        self.three_bet_multiplier = three_bet_multiplier
        self.value_bet_equity = value_bet_equity
        self.monster_equity = monster_equity
        self.base_bet_fraction = base_bet_fraction
        self.wet_bet_extra = wet_bet_extra
        self.raise_face_base = raise_face_base
        self.wet_raise_extra = wet_raise_extra
        self.call_margin = call_margin
        self.wet_call_extra = wet_call_extra
        self.paired_call_relief = paired_call_relief
        self.bet_discount = bet_discount
        self.semi_bluff_outs = semi_bluff_outs
        self.semi_bluff_freq = semi_bluff_freq
        self.semi_bluff_min_equity = semi_bluff_min_equity
        self.air_bluff_fold_scale = air_bluff_fold_scale
        self.air_bluff_max_equity = air_bluff_max_equity
        self.draw_raise_outs = draw_raise_outs
        self.draw_call_credit = draw_call_credit
        self.texture_effect_scale = texture_effect_scale
        self.draw_effect_scale = draw_effect_scale
        self.prior_hands = prior_hands
        self.prior_fold_rate = prior_fold_rate
        self.equity_samples = equity_samples

    def init_state(self, key):
        return {"hands_seen": jnp.zeros((), jnp.float32),
                "opponent_folded_hands": jnp.zeros((), jnp.float32)}

    def observe(self, record, state, key):
        opponent_folded = record.seated[1] & ~record.in_hand[1]
        return {"hands_seen": state["hands_seen"] + 1.0,
                "opponent_folded_hands": state["opponent_folded_hands"] + opponent_folded.astype(jnp.float32)}

    def board_texture(self, obs):
        """Board wetness in [0,1] from suitedness, connectedness and high cards, plus a paired flag."""
        valid = obs.board >= 0
        suits = jnp.where(valid, obs.board // 13, -1)
        ranks = jnp.where(valid, obs.board % 13, -1)
        suit_counts = jnp.sum(suits[None, :] == jnp.arange(4)[:, None], axis=1)
        rank_counts = jnp.sum(ranks[None, :] == jnp.arange(13)[:, None], axis=1)
        flushiness = jnp.clip(jnp.max(suit_counts) / 3.0, 0.0, 1.0)
        window_counts = jnp.sum(WINDOWS * rank_counts[None, :], axis=1)
        connectedness = jnp.clip(jnp.max(window_counts) / 3.0, 0.0, 1.0)
        top_rank = jnp.max(ranks)
        high_cards = jnp.clip((top_rank - 5) / 7.0, 0.0, 1.0)
        paired = (jnp.max(rank_counts) >= 2).astype(jnp.float32)
        wetness = 0.45 * flushiness + 0.35 * connectedness + 0.20 * high_cards
        return wetness, paired

    def draw_outs(self, obs):
        """Number of distinct next-street cards that promote the made hand to two pair or better."""
        valid = obs.board >= 0
        cards = jnp.concatenate([obs.hole_cards, jnp.maximum(obs.board, 0)])
        six_cards = jnp.tile(cards[:5], (52, 1)).at[:, 5].set(jnp.arange(52))
        seven_cards = jnp.tile(cards[:6], (52, 1)).at[:, 6].set(jnp.arange(52))
        category_next = jnp.where(obs.current_round == 1, evaluate_hands(six_cards), evaluate_hands(seven_cards)) // 13 ** 5
        category_now = hand_category(made_hand(obs))
        known = jnp.zeros(52, jnp.bool_).at[obs.hole_cards].set(True).at[jnp.maximum(obs.board, 0)].max(valid)
        improves = (category_next >= 2) & (category_next > category_now) & ~known
        on_draw_street = (obs.current_round >= 1) & (obs.current_round <= 2)
        return jnp.sum(improves & on_draw_street)

    def preflop_play(self, obs, showdown_equity):
        call_chips = to_call(obs)
        open_raise = raise_to(obs, self.open_size_bb * obs.big_blind)
        # button (owing exactly the blind completion): open-raise almost every hand, complete the worst few
        button_play = choose(showdown_equity > self.open_equity, open_raise, check_call())
        # big blind against a limp: raise the top of range, check the rest
        bb_vs_limp = choose(showdown_equity > self.limp_raise_equity, open_raise, check_call())
        # facing a raise: three-bet premiums, defend hands with equity and price, fold the rest
        three_bet = raise_to(obs, obs.current_bet * (1.0 + self.three_bet_multiplier))
        defense_needed = jnp.maximum(self.defend_equity, pot_odds(obs) + 0.02)
        defend = choose(showdown_equity > defense_needed, check_call(), fold_or_check(obs))
        vs_raise = choose(showdown_equity > self.three_bet_equity, three_bet, defend)
        unraised = choose(obs.button == 0, button_play, bb_vs_limp)
        return choose(call_chips >= 2, vs_raise, unraised)

    def postflop_play(self, obs, showdown_equity, wetness, paired, outs, opponent_fold_rate, key):
        wet = wetness * self.texture_effect_scale
        pair_flag = paired * self.texture_effect_scale
        effective_outs = outs * self.draw_effect_scale
        call_chips = to_call(obs)
        pot_after_call = obs.pot + call_chips
        faced_fraction = jnp.minimum(call_chips / jnp.maximum(obs.pot, 1), 1.5)
        dice = jax.random.uniform(key)
        # checked to us: bet strong equity bigger on wet boards, semi-bluff draws, bluff air on quiet boards
        bet_fraction = self.base_bet_fraction + self.wet_bet_extra * wet
        value_bet = showdown_equity > self.value_bet_equity
        semi_bluff = (effective_outs >= self.semi_bluff_outs) & (showdown_equity > self.semi_bluff_min_equity) & (dice < self.semi_bluff_freq)
        air_bluff = (showdown_equity < self.air_bluff_max_equity) & (wet < 0.45) & (dice < self.air_bluff_fold_scale * opponent_fold_rate)
        bet = raise_to(obs, obs.current_bet + bet_fraction * pot_after_call)
        checked_to = choose(value_bet | semi_bluff | air_bluff, bet, check_call())
        # facing a bet: raise monsters and huge draws, call when equity clears pot odds plus a texture-scaled margin
        range_discount = self.bet_discount * faced_fraction
        monster = showdown_equity > self.monster_equity
        draw_raise = (effective_outs >= self.draw_raise_outs) & (showdown_equity > self.semi_bluff_min_equity) & (dice > 0.5)
        raise_fraction = self.raise_face_base + self.wet_raise_extra * wet
        big_raise = raise_to(obs, obs.current_bet + raise_fraction * pot_after_call)
        margin = self.call_margin + self.wet_call_extra * wet * faced_fraction - self.paired_call_relief * pair_flag
        call_needed = pot_odds(obs) + margin - self.draw_call_credit * (effective_outs >= self.draw_raise_outs)
        good_call = (showdown_equity - range_discount) > call_needed
        vs_bet = choose(monster | draw_raise, big_raise, choose(good_call, check_call(), fold_or_check(obs)))
        return choose(call_chips > 0, vs_bet, checked_to)

    def act(self, obs, state, key):
        equity_key, action_key = jax.random.split(key)
        showdown_equity = equity(obs, equity_key, self.equity_samples)
        wetness, paired = self.board_texture(obs)
        outs = self.draw_outs(obs)
        opponent_fold_rate = (state["opponent_folded_hands"] + self.prior_fold_rate * self.prior_hands) / (state["hands_seen"] + self.prior_hands)
        preflop = self.preflop_play(obs, showdown_equity)
        postflop = self.postflop_play(obs, showdown_equity, wetness, paired, outs, opponent_fold_rate, action_key)
        return choose(obs.current_round == 0, preflop, postflop), state

    def metadata(self) -> dict:
        return {
            "bot_name": "vicious_glm",
            "version": 2,
            "parent": 1,
            "lineage": "phase_5_cowboy_bebop/vicious_glm",
            "model": "GLM-5.3",
            "prompt_version": "botgen-v5",
            "phase": "phase_5_cowboy_bebop",
            "engine": "decafb9 sha256:295477eff7637056",
            "params": {name: value for name, value in vars(self).items()},
            "contributions": [
                {"role": "requester", "who": "a"},
                {"role": "generator", "interface": "GLM"},
            ],
        }
