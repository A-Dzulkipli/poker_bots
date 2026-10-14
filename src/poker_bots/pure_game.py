"""Plain-Python rebuild of poker_game.py for bots written as ordinary classes.

Same rules, observation fields and hand records as the JAX engine, checked in lockstep by
tests/test_pure_game.py. Bots hold their own state on self and use their own randomness; one
instance sits in each seat.
"""
import random
from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence

from poker_bots.hand_eval import NUM_RANKS, NUM_SUITS, build_rank_tables, get_multiset_index
from poker_bots.poker_game import Action, ActionData, Round

VISIBLE_BOARD = (0, 3, 4, 5)


def evaluate_hand(cards: Sequence[int]) -> int:
    """Scores 5 to 7 cards exactly as hand_eval.evaluate_hands; higher wins."""
    multiset_table, flush_table = build_rank_tables(len(cards))
    for suit in range(NUM_SUITS):
        suited = [card % NUM_RANKS for card in cards if card // NUM_RANKS == suit]
        if len(suited) >= 5:
            return int(flush_table[sum(1 << rank for rank in suited)])
    return int(multiset_table[get_multiset_index(tuple(sorted(card % NUM_RANKS for card in cards)))])


@dataclass
class ActionRecord:
    player_id: int
    action_type: int
    amount: int
    requested_type: int
    requested: int
    is_all_in: bool
    round: int


@dataclass
class Observation:
    """A seat's view, in the same fields as PlayerTableState. Index 0 is the observing seat."""
    hole_cards: list[int]
    board: list[int]
    history: list[ActionRecord]
    stacks: list[int]
    seated: list[bool]
    in_hand: list[bool]
    button: int
    sb_seat: int
    bb_seat: int
    contributed: list[int]
    current_round: int
    min_raise: int
    current_bet: int
    pot: int
    can_raise: bool
    raise_lower_bound: int
    raise_upper_bound: int
    big_blind: int
    hand_count: int

    @property
    def to_call(self) -> int:
        return min(self.current_bet - self.contributed[0], self.stacks[0])


@dataclass
class HandRecord:
    hand_index: int
    button: int
    sb_seat: int
    bb_seat: int
    seated: list[bool]
    starting_stacks: list[int]
    hole_cards: list[list[int]]
    board: list[int]
    board_dealt: int
    history: list[ActionRecord]
    in_hand: list[bool]
    went_to_showdown: bool
    hand_ranks: list[int]
    payouts: list[int]
    ending_stacks: list[int]
    rebuys: list[int]
    cash_outs: list[int]


class Bot(Protocol):
    """act returns ActionData(type, value) with the JAX engine's semantics: FOLD (and NO_OP) always
    folds, CHECK_CALL checks or calls, RAISE takes value as this street's raise-to total, clipped to
    [raise_lower_bound, raise_upper_bound], and is a call when can_raise is False.

    Optional: observe(record: HandRecord) -> None, called at the end of every hand with the seat's
    relative record (hidden cards and unshown hand_ranks are -1), and metadata() -> dict."""
    def act(self, obs: Observation) -> ActionData: ...


class Table:
    def __init__(
        self,
        bots: Sequence[Bot | None],
        small_blind: int = 1,
        big_blind: int = 2,
        starting_stacks: Sequence[int] | None = None,
        button: int = 0,
        min_stack: int = 0,
        max_stack: int | None = None,
        rebuy: bool = True,
        seed: int | None = None,
        rng: random.Random | None = None
    ):
        self.bots = list(bots)
        self.max_players = len(self.bots)
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.rebuy = rebuy
        self.rng = rng or random.Random(seed)
        if starting_stacks is None:
            starting_stacks = [100 * big_blind] * self.max_players
        self.starting_stacks = list(starting_stacks)
        self.stacks = list(starting_stacks)
        self.buy_in = list(starting_stacks)
        self.seated = [bot is not None and stack > 0 for bot, stack in zip(self.bots, self.stacks)]
        self.min_stack = min_stack
        self.max_stack = 300 * big_blind if max_stack is None else max_stack
        self.hand_count = 0

        seat_order = [(button + 1 + i) % self.max_players for i in range(self.max_players)]
        seated_ordered = [seat for seat in seat_order if self.seated[seat]]
        self.button = button
        if len(seated_ordered) == 2:
            self.sb_seat, self.bb_seat = button, seated_ordered[0]
        else:
            self.sb_seat, self.bb_seat = (seated_ordered + [seat_order[0]] * 2)[:2]
        self.current_player = self.bb_seat
        self.start_hand()

    # positions and masks

    def first_after(self, seat: int, mask: Sequence[bool]) -> int:
        order = [(seat + i) % self.max_players for i in range(1, self.max_players + 1)]
        return next((s for s in order if mask[s]), order[0])

    def eligible(self) -> list[bool]:
        return [h and s for h, s in zip(self.in_hand, self.seated)]

    def can_act(self) -> list[bool]:
        return [e and stack > 0 for e, stack in zip(self.eligible(), self.stacks)]

    def to_act(self) -> list[bool]:
        can_act = self.can_act()
        many = sum(can_act) >= 2
        return [
            c and (not a or owes) and (many or owes)
            for c, a, owes in zip(can_act, self.acted, (x < self.current_bet for x in self.contributed))
        ]

    def can_raise(self, seat: int) -> bool:
        to_call = self.current_bet - self.contributed[seat]
        reopened = to_call >= self.min_raise
        return (reopened or not self.acted[seat]) and to_call < self.stacks[seat] and sum(self.can_act()) > 1

    def raise_bounds(self, seat: int) -> tuple[int, int]:
        upper = self.contributed[seat] + self.stacks[seat]
        return min(upper, self.current_bet + self.min_raise), upper

    # hand flow

    def start_hand(self):
        deck = list(range(52))
        self.rng.shuffle(deck)
        n = self.max_players
        self.hole_cards = [[deck[i], deck[n + i]] for i in range(n)]
        self.runout = deck[2 * n:2 * n + 5]
        self.history: list[ActionRecord] = []
        self.in_hand = list(self.seated)
        self.acted = [False] * n
        self.contributed = [0] * n
        self.current_round = Round.PRE_FLOP
        self.min_raise = self.big_blind
        self.current_bet = 0
        if sum(self.seated) <= 1:
            return
        for seat, blind, kind in ((self.sb_seat, self.small_blind, Action.SMALL_BLIND), (self.bb_seat, self.big_blind, Action.BIG_BLIND)):
            amount = min(self.stacks[seat], blind)
            self.contributed[seat] = amount
            self.stacks[seat] -= amount
            if amount > 0:
                self.history.append(ActionRecord(seat, kind, amount, kind, amount, self.stacks[seat] == 0, Round.PRE_FLOP))
        self.current_bet = self.big_blind
        self.current_player = self.first_after(self.bb_seat, self.to_act())

    def next_event(self) -> int:
        """0 table finished, 1 hand over, 2 street over, 3 player to act."""
        if sum(self.seated) <= 1:
            return 0
        street_over = not any(self.to_act())
        if (street_over and self.current_round == Round.RIVER) or sum(self.eligible()) <= 1:
            return 1
        return 2 if street_over else 3

    def apply_action(self, action: ActionData):
        seat = self.current_player
        lower, upper = self.raise_bounds(seat)
        is_fold = action.type in (Action.FOLD, Action.NO_OP)
        is_raise = action.type == Action.RAISE and self.can_raise(seat)
        if is_fold:
            target = self.contributed[seat]
        elif is_raise:
            target = min(max(action.value, lower), upper)
        else:
            target = min(self.current_bet, upper)
        delta = target - self.contributed[seat]
        increment = target - self.current_bet
        kind = Action.FOLD if is_fold else Action.RAISE if is_raise else Action.CHECK_CALL
        self.history.append(ActionRecord(seat, kind, target, int(action.type), int(action.value), delta == self.stacks[seat], self.current_round))
        self.stacks[seat] -= delta
        self.contributed[seat] = target
        self.in_hand[seat] = self.in_hand[seat] and not is_fold
        self.acted[seat] = True
        if increment >= self.min_raise:
            self.min_raise = increment
        self.current_bet = max(self.current_bet, target)
        self.current_player = self.first_after(seat, self.to_act())

    def next_street(self):
        self.current_round = (self.current_round + 1) % (Round.RIVER + 1)
        self.contributed = [0] * self.max_players
        self.current_bet = 0
        self.min_raise = self.big_blind
        self.acted = [False] * self.max_players
        self.current_player = self.first_after(self.button, self.to_act())

    def get_payouts(self, hand_ranks: list[int]) -> list[int]:
        committed = [start - stack for start, stack in zip(self.starting_stacks, self.stacks)]
        eligible = self.eligible()
        button_order = [(self.button + i) % self.max_players for i in range(1, self.max_players + 1)]
        payouts = [0] * self.max_players
        previous = 0
        for level in sorted(committed):
            pot_slice = sum(min(max(c, previous), level) - previous for c in committed)
            contenders = [e and c >= level for e, c in zip(eligible, committed)]
            best = max((r for r, c in zip(hand_ranks, contenders) if c), default=-1)
            winners = [c and r == best for c, r in zip(contenders, hand_ranks)]
            share = pot_slice // max(sum(winners), 1)
            odd_chips = pot_slice - share * sum(winners)
            for seat in range(self.max_players):
                payouts[seat] += share if winners[seat] else 0
            for seat in [s for s in button_order if winners[s]][:odd_chips]:
                payouts[seat] += 1
            previous = level
        return payouts

    def get_hand_record(self) -> HandRecord:
        hand_ranks = [evaluate_hand(hole + self.runout) for hole in self.hole_cards]
        payouts = self.get_payouts(hand_ranks)
        ending = [stack + payout for stack, payout in zip(self.stacks, payouts)]
        if self.rebuy:
            rebuys = [max(buy_in - end, 0) if seated and end <= self.min_stack else 0 for buy_in, end, seated in zip(self.buy_in, ending, self.seated)]
            cash_outs = [max(end + rebuy - self.max_stack, 0) for end, rebuy in zip(ending, rebuys)]
        else:
            rebuys = [0] * self.max_players
            cash_outs = [0] * self.max_players
        eligible = self.eligible()
        return HandRecord(
            hand_index=self.hand_count,
            button=self.button,
            sb_seat=self.sb_seat,
            bb_seat=self.bb_seat,
            seated=list(self.seated),
            starting_stacks=list(self.starting_stacks),
            hole_cards=[list(hole) for hole in self.hole_cards],
            board=list(self.runout),
            board_dealt=VISIBLE_BOARD[self.current_round],
            history=list(self.history),
            in_hand=eligible,
            went_to_showdown=sum(eligible) >= 2,
            hand_ranks=hand_ranks,
            payouts=payouts,
            ending_stacks=ending,
            rebuys=rebuys,
            cash_outs=cash_outs
        )

    def next_hand(self, record: HandRecord):
        self.stacks = [end + rebuy - cash for end, rebuy, cash in zip(record.ending_stacks, record.rebuys, record.cash_outs)]
        self.seated = [seated and stack > 0 for seated, stack in zip(self.seated, self.stacks)]
        live = [seated and stack > 0 for seated, stack in zip(self.seated, self.stacks)]
        next_bb = self.first_after(self.bb_seat, live)
        if sum(live) == 2:
            self.button = self.sb_seat = self.first_after(next_bb, live)
        else:
            self.button, self.sb_seat = self.sb_seat, self.bb_seat
        self.bb_seat = next_bb
        self.starting_stacks = list(self.stacks)
        self.hand_count += 1
        self.start_hand()

    def step(self, action: ActionData | None = None) -> HandRecord | None:
        """Advances one event. A player-to-act step uses action if given, else asks the seat's bot.
        Returns the finished hand's record on the step that ends it."""
        event = self.next_event()
        if event == 1:
            record = self.get_hand_record()
            for seat, bot in enumerate(self.bots):
                if record.seated[seat] and hasattr(bot, "observe"):
                    bot.observe(relative_record(record, seat))
            self.next_hand(record)
            return record
        if event == 2:
            self.next_street()
        elif event == 3:
            if action is None:
                action = self.bots[self.current_player].act(self.observation(self.current_player))
            self.apply_action(action)
        return None

    def play_hand(self) -> HandRecord | None:
        """Steps to the end of the current hand; None if the table has finished."""
        while self.next_event() != 0:
            record = self.step()
            if record is not None:
                return record
        return None

    def run(self, num_hands: int) -> list[HandRecord]:
        records = []
        while len(records) < num_hands and (record := self.play_hand()) is not None:
            records.append(record)
        return records

    def observation(self, seat: int) -> Observation:
        """For a seat other than current_player, can_raise and the raise bounds are as if it were to act."""
        n = self.max_players
        rotate = lambda values: [values[(seat + i) % n] for i in range(n)]
        lower, upper = self.raise_bounds(seat)
        can_raise = self.can_raise(seat)
        call = min(self.current_bet, upper)
        return Observation(
            hole_cards=list(self.hole_cards[seat]),
            board=[card if i < VISIBLE_BOARD[self.current_round] else -1 for i, card in enumerate(self.runout)],
            history=relative_history(self.history, seat, n),
            stacks=rotate(self.stacks),
            seated=rotate(self.seated),
            in_hand=rotate(self.in_hand),
            button=(self.button - seat) % n,
            sb_seat=(self.sb_seat - seat) % n,
            bb_seat=(self.bb_seat - seat) % n,
            contributed=rotate(self.contributed),
            current_round=self.current_round,
            min_raise=self.min_raise,
            current_bet=self.current_bet,
            pot=sum(start - stack for start, stack in zip(self.starting_stacks, self.stacks)),
            can_raise=can_raise,
            raise_lower_bound=lower if can_raise else call,
            raise_upper_bound=upper if can_raise else call,
            big_blind=self.big_blind,
            hand_count=self.hand_count
        )


def relative_history(history: list[ActionRecord], seat: int, max_players: int) -> list[ActionRecord]:
    return [
        replace(record, player_id=(record.player_id - seat) % max_players, requested_type=record.action_type, requested=record.amount)
        for record in history
    ]


def relative_record(record: HandRecord, seat: int) -> HandRecord:
    n = len(record.seated)
    rotate = lambda values: [values[(seat + i) % n] for i in range(n)]
    shown = [record.went_to_showdown and in_hand for in_hand in record.in_hand]
    return replace(
        record,
        button=(record.button - seat) % n,
        sb_seat=(record.sb_seat - seat) % n,
        bb_seat=(record.bb_seat - seat) % n,
        seated=rotate(record.seated),
        starting_stacks=rotate(record.starting_stacks),
        hole_cards=rotate([hole if shown[s] or s == seat else [-1, -1] for s, hole in enumerate(record.hole_cards)]),
        board=[card if i < record.board_dealt else -1 for i, card in enumerate(record.board)],
        history=relative_history(record.history, seat, n),
        in_hand=rotate(record.in_hand),
        hand_ranks=rotate([rank if shown[s] else -1 for s, rank in enumerate(record.hand_ranks)]),
        payouts=rotate(record.payouts),
        ending_stacks=rotate(record.ending_stacks),
        rebuys=rotate(record.rebuys),
        cash_outs=rotate(record.cash_outs)
    )


class RandomBot:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> ActionData:
        action_type = self.rng.choices([Action.FOLD, Action.CHECK_CALL, Action.RAISE], weights=[0.15, 0.6, 0.25])[0]
        small_raise_cap = min(obs.raise_upper_bound, obs.raise_lower_bound + 3 * obs.big_blind)
        if self.rng.random() < 0.1:
            value = obs.raise_upper_bound
        else:
            value = self.rng.randint(obs.raise_lower_bound, small_raise_cap)
        return ActionData(type=action_type, value=value)

    def metadata(self) -> dict:
        return {"bot_type": "random_bot"}
