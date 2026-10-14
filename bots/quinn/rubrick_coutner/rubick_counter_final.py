"""rubick_counter: kraken_gpt's adaptive play plus three pressure plays aimed at Rubick.

Why it works. rubick_strict plays windrunner_glm on ~90% of its hands, and rubick (argmax) on ~30%.
windrunner discounts its equity by a fixed factor (0.58) per unit of our betting pressure, whatever
the bet size, and folds unless the discounted equity beats the pot odds. So:
    - a big 3-bet from the big blind (20bb) folds even its aces,
    - a 4-bet all-in over its 3-bet folds everything (an all-in also takes away its re-raise),
    - a ~0.8-pot flop bet after we opened folds everything below ~0.8 equity (the rest raises).
The other bots Rubick switches to call these plays more, so each play is gated: we track, per spot,
how often the opponent folds, re-raises or calls our raise, and press only when the expected value
beats the passive line. Everything else is kraken_gpt (phase 4's best bot), unchanged.
"""
import jax
import jax.numpy as jnp
from poker_bots.poker_game import Action, Round

from bots.phase_4_sea_monsters.kraken_gpt_final import Bot as Kraken, equity, to_call, raise_to, choose

PREFLOP_SPOTS = ("threebet", "fourbet_jam")
POSTFLOP_SPOTS = tuple(f"{street}_{line}" for street in ("flop", "turn", "river") for line in ("bet", "raise"))
SPOTS = PREFLOP_SPOTS + POSTFLOP_SPOTS


class Bot(Kraken):
    def __init__(self,
                 threebet_bb=20.0,                          # big-blind 3-bet, raise-to in big blinds
                 postflop_pot_fraction=(0.8, 0.45, 0.35),   # flop / turn / river pressure bet, pot fraction
                 prior_hands=6.0,                           # pseudo-count opportunities per spot
                 preflop_prior_fold_rate=0.85,              # optimistic: exploring preflop is cheap
                 postflop_prior_fold_rate=0.5,
                 prior_reraise_rate=0.05,                   # prior chance they re-raise our pressure
                 preflop_equity_drop=0.15,                  # preflop equity lost vs a calling range, per unit fold rate
                 ev_margin_pot=0.1,                         # pressure must beat the passive line by this pot fraction
                 tight_threebet_fold_rate=0.9,              # fold rate to our 3-bets above which their 3-bets are treated as light
                 plays=("threebet", "jam", "postflop"),
                 **kraken_params):
        super().__init__(**kraken_params)
        self.threebet_bb = threebet_bb
        self.postflop_pot_fraction = postflop_pot_fraction
        self.prior_hands = prior_hands
        self.preflop_prior_fold_rate = preflop_prior_fold_rate
        self.postflop_prior_fold_rate = postflop_prior_fold_rate
        self.prior_reraise_rate = prior_reraise_rate
        self.preflop_equity_drop = preflop_equity_drop
        self.ev_margin_pot = ev_margin_pot
        self.tight_threebet_fold_rate = tight_threebet_fold_rate
        self.plays = tuple(plays)

    def init_state(self, key):
        state = super().init_state(key)
        for spot in SPOTS:
            fold = self.preflop_prior_fold_rate if spot in PREFLOP_SPOTS else self.postflop_prior_fold_rate
            state[f"{spot}_chances"] = jnp.asarray(self.prior_hands, jnp.float32)
            state[f"{spot}_folds"] = jnp.asarray(self.prior_hands * fold, jnp.float32)
            state[f"{spot}_reraises"] = jnp.asarray(self.prior_hands * self.prior_reraise_rate, jnp.float32)
        return state

    # ---- tracking: after each of our raises, the opponent's next action (fold / re-raise / call) ----
    def observe(self, record, state, key):
        state = {**state, **super().observe(record, state, key)}
        h = record.history
        ours = h.is_valid & (h.player_id == 0) & (h.action_type == Action.RAISE)
        theirs = h.is_valid & (h.player_id != 0)
        nothing = jnp.zeros(1, bool)
        next_fold = jnp.concatenate([theirs[1:] & (h.action_type[1:] == Action.FOLD), nothing])
        next_reraise = jnp.concatenate([theirs[1:] & (h.action_type[1:] == Action.RAISE), nothing])
        preflop = h.round == Round.PRE_FLOP
        their_preflop_raise_before = jnp.cumsum(theirs & (h.action_type == Action.RAISE) & preflop) > 0
        our_preflop_raises_before = jnp.cumsum(ours & preflop) - ours
        spots = {
            "threebet": ours & preflop & their_preflop_raise_before & (our_preflop_raises_before == 0),
            "fourbet_jam": ours & preflop & (our_preflop_raises_before >= 1) & h.is_all_in,
        }
        # postflop, per street: we bet first ("bet") or raised their bet ("raise")
        same_street_as_previous = jnp.concatenate([nothing, h.round[:-1] == h.round[1:]])
        after_their_bet = jnp.concatenate([nothing, (theirs & (h.action_type == Action.RAISE))[:-1]]) & same_street_as_previous
        for street, number in (("flop", Round.FLOP), ("turn", Round.TURN), ("river", Round.RIVER)):
            on_street = ours & (h.round == number)
            spots[f"{street}_bet"] = on_street & ~after_their_bet
            spots[f"{street}_raise"] = on_street & after_their_bet
        for name, mask in spots.items():
            state[f"{name}_chances"] += jnp.sum(mask, dtype=jnp.float32)
            state[f"{name}_folds"] += jnp.sum(mask & next_fold, dtype=jnp.float32)
            state[f"{name}_reraises"] += jnp.sum(mask & next_reraise, dtype=jnp.float32)
        return state

    def response_rates(self, state, spot):
        return state[f"{spot}_folds"] / state[f"{spot}_chances"], state[f"{spot}_reraises"] / state[f"{spot}_chances"]

    # ---- expected value of raising to `target`: they fold, re-raise (we give up), or call ----
    def pressure_ev(self, obs, target, raw_equity, fold_rate, reraise_rate, preflop_drop_scale):
        hero_total = obs.contributed[0] + obs.stacks[0]
        opponent_total = obs.contributed[1] + obs.stacks[1]
        target = jnp.minimum(target, jnp.minimum(hero_total, opponent_total))
        ours = target - obs.contributed[0]
        theirs = target - obs.contributed[1]
        all_in = target >= jnp.minimum(hero_total, opponent_total)
        reraise_rate = jnp.where(all_in, 0.0, reraise_rate)     # nobody can re-raise an all-in
        call_rate = jnp.clip(1.0 - fold_rate - reraise_rate, 0.0, 1.0)
        # the hands that continue are the strongest part of their range
        keep = jnp.clip(1.0 - fold_rate, 0.02, 1.0)
        postflop_called = jnp.clip((raw_equity - (1.0 - keep)) / keep, 0.0, 1.0)
        # preflop equities are compressed (aces are ~80% even against a tight range), so the drop is mild
        preflop_called = jnp.clip(raw_equity - preflop_drop_scale * self.preflop_equity_drop * fold_rate, 0.0, 1.0)
        equity_when_called = jnp.where(obs.current_round == Round.PRE_FLOP, preflop_called, postflop_called)
        called = equity_when_called * (obs.pot + ours + theirs) - ours
        ev = fold_rate * obs.pot + call_rate * called - reraise_rate * ours
        return ev, target

    def passive_ev(self, obs, raw_equity):
        """Rough value of checking or calling: our share of the pot at showdown, never below folding."""
        cost = to_call(obs)
        return jnp.maximum(0.0, raw_equity * (obs.pot + cost) - cost)

    def act(self, obs, state, key):
        kraken_key, equity_key = jax.random.split(key)
        kraken_action, state = super().act(obs, state, kraken_key)
        raw = equity(obs, equity_key, 128)
        h = obs.history
        preflop = obs.current_round == Round.PRE_FLOP
        on_button = obs.button == 0
        raises = h.is_valid & (h.action_type == Action.RAISE)
        their_preflop_raises = jnp.sum(raises & (h.player_id != 0) & (h.round == Round.PRE_FLOP))
        our_preflop_raises = jnp.sum(raises & (h.player_id == 0) & (h.round == Round.PRE_FLOP))
        our_street_raises = jnp.sum(raises & (h.player_id == 0) & (h.round == obs.current_round))
        their_street_raises = jnp.sum(raises & (h.player_id != 0) & (h.round == obs.current_round))

        # Preflop as big blind facing an open: a big 3-bet.
        threebet = preflop & ~on_button & (their_preflop_raises == 1) & (our_preflop_raises == 0)
        # Preflop as button facing a 3-bet: all in.
        jam = preflop & on_button & (our_preflop_raises == 1) & (their_preflop_raises >= 1)
        # Postflop: our first raise of the street, betting or raising once; never in a raising war.
        postflop = ~preflop & (our_street_raises == 0) & (their_street_raises <= 1)
        street = jnp.clip(obs.current_round - Round.FLOP, 0, 2)
        postflop_target = obs.current_bet + jnp.asarray(self.postflop_pot_fraction)[street] * (obs.pot + to_call(obs))
        target = jnp.where(threebet, self.threebet_bb * obs.big_blind,
                           jnp.where(jam, obs.contributed[0] + obs.stacks[0], postflop_target))

        postflop_rates = jnp.stack([jnp.stack(self.response_rates(state, s)) for s in POSTFLOP_SPOTS]).reshape(3, 2, 2)
        facing_bet = (to_call(obs) > 0).astype(jnp.int32)
        rates = jnp.where(threebet, jnp.stack(self.response_rates(state, "threebet")),
                          jnp.where(jam, jnp.stack(self.response_rates(state, "fourbet_jam")), postflop_rates[street, facing_bet]))
        # A 3-bettor who calls our jam has a doubly narrowed range, so the equity drop is larger,
        # unless they fold to our 3-bets almost always (then their 3-bets are rarely value-heavy either).
        threebet_fold_rate = self.response_rates(state, "threebet")[0]
        loose_threebettor = jnp.clip((self.tight_threebet_fold_rate - threebet_fold_rate) / 0.3, 0.0, 1.0)
        drop_scale = jnp.where(jam, 1.0 + loose_threebettor, 1.0)
        ev, target = self.pressure_ev(obs, target, raw, rates[0], rates[1], drop_scale)

        enabled = ((threebet & ("threebet" in self.plays)) | (jam & ("jam" in self.plays))
                   | (postflop & ("postflop" in self.plays)))
        press = enabled & obs.can_raise & (ev > self.passive_ev(obs, raw) + self.ev_margin_pot * obs.pot)
        return choose(press, raise_to(obs, target), kraken_action), state

    def metadata(self):
        return {"bot_name": "rubick_counter", "kind": "bot", "version": 9, "base": "phase_4_sea_monsters/kraken_gpt",
                "description": "kraken_gpt plus tracked, EV-gated pressure plays that exploit windrunner-style equity discounting",
                "params": {"threebet_bb": self.threebet_bb, "postflop_pot_fraction": list(self.postflop_pot_fraction),
                           "prior_hands": self.prior_hands, "preflop_prior_fold_rate": self.preflop_prior_fold_rate,
                           "postflop_prior_fold_rate": self.postflop_prior_fold_rate,
                           "prior_reraise_rate": self.prior_reraise_rate, "preflop_equity_drop": self.preflop_equity_drop,
                           "ev_margin_pot": self.ev_margin_pot, "tight_threebet_fold_rate": self.tight_threebet_fold_rate, "plays": list(self.plays)}}
