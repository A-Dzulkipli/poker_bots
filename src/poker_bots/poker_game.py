from collections import namedtuple
import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable
from typing import Protocol, TypeVar

from poker_bots.hand_eval import evaluate_hands

Card = namedtuple('Card', ['Rank', 'Suit'])

RANKS = "2 3 4 5 6 7 8 9 10 J Q K A".split()
SUITS = "Diamonds Clubs Hearts Spades".split()
ID_TO_CARD = [(rank, suit) for suit in SUITS for rank in RANKS]
CARD_TO_ID = {
    card: i for i, card in enumerate(ID_TO_CARD)
}


# convenience structs

class Action:
    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2
    NO_OP = 3
    SMALL_BLIND = 4
    BIG_BLIND = 5

class Round:
    PRE_FLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

# ground truth data

class DeckState(NamedTuple):
    hole_cards: jax.Array
    runout: jax.Array

class ActionRecord(NamedTuple):
    player_id: jax. Array
    action_type: jax.Array
    amount: jax.Array
    is_all_in: jax.Array
    round: jax.Array
    is_valid: jax.Array

    @classmethod
    def init(cls, max_actions: int) -> "ActionRecord":
        return cls(
            player_id=jnp.full((max_actions,), fill_value=-1, dtype=jnp.int32),
            action_type=jnp.full((max_actions,), fill_value=Action.NO_OP, dtype=jnp.int32),
            amount=jnp.zeros((max_actions,), dtype=jnp.int32),
            is_all_in=jnp.zeros((max_actions,), dtype=jnp.bool_),
            round=jnp.zeros((max_actions,), dtype=jnp.int32),
            is_valid=jnp.zeros((max_actions,), dtype=jnp.bool_)
        )

class PlayerTableState(NamedTuple):
    hole_cards: jax.Array
    board: jax.Array
    history: ActionRecord
    stacks: jax.Array
    seated: jax.Array
    in_hand: jax.Array
    button: jax.Array
    sb_seat: jax.Array
    bb_seat: jax.Array
    contributed: jax.Array
    current_round: jax.Array
    min_raise: jax.Array
    current_bet: jax.Array
    pot: jax.Array
    can_raise: jax.Array
    raise_lower_bound: jax.Array
    raise_upper_bound: jax.Array
    big_blind: jax.Array


class TableState(NamedTuple):
    deck: DeckState
    history: ActionRecord
    starting_stacks: jax.Array
    seated: jax.Array
    stacks: jax.Array
    in_hand: jax.Array
    acted: jax.Array
    button: jax.Array
    sb_seat: jax.Array
    bb_seat: jax.Array
    action_count: jax.Array
    contributed: jax.Array
    master_key: jax.Array
    player_states: tuple
    seat_bot: jax.Array
    current_player: jax.Array
    current_round: jax.Array
    min_raise: jax.Array
    current_bet: jax.Array

    @property
    def eligible_mask(self) -> jax.Array:
        return self.in_hand & self.seated

    @property
    def can_act_mask(self) -> jax.Array:
        return self.eligible_mask & (self.stacks > 0)

    @property
    def num_eligible(self) -> jax.Array:
        return jnp.sum(self.eligible_mask, dtype=jnp.int32)

    @property
    def num_can_act(self) -> jax.Array:
        return jnp.sum(self.can_act_mask, dtype=jnp.int32)

    @property
    def is_early_fold_out(self) -> jax.Array:
        return self.num_eligible <= 1

    @property
    def is_all_in_runout(self) -> jax.Array:
        return (self.num_eligible >= 2) & (self.num_can_act <= 1)

    @property
    def betting_active(self) -> jax.Array:
        return (self.num_eligible >= 2) & (self.num_can_act >= 2)

    @property
    def num_seated(self) -> jax.Array:
        return jnp.sum(self.seated)

    @property
    def to_act_mask(self):
        owes = self.contributed < self.current_bet
        return self.can_act_mask & (~self.acted | owes) & ((self.num_can_act >= 2) | owes)

class ActionData(NamedTuple):
    type: jax.Array
    value: jax.Array

class PlayerActionResponse(NamedTuple):
    table_state: TableState
    action: ActionData

class PokerEnv(NamedTuple):
    init: Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, tuple], TableState]
    get_payouts: Callable[[TableState, jax.Array], jax.Array]
    step: Callable[[TableState], TableState]
    player_metadata: tuple

# bot protocol

S = TypeVar("S")

class Bot(Protocol[S]):
    def init_state(self, key: jax.Array) -> S: ...
    def act(self, obs: PlayerTableState, state: S, key: jax.Array) -> tuple[ActionData, S]: ...
    def metadata(self) -> dict: ...

def make_poker_env(bots: tuple[Bot, ...], max_players: int = 9, max_actions: int = 100, small_blind: int = 1, big_blind: int = 2) -> PokerEnv:
    num_hole = 2 * max_players
    board_end = num_hole + 5

    def extract_player_meta(bot: Bot) -> dict:
        return bot.metadata()

    def get_deck_state(key: jax.Array) -> DeckState:
        deck_order = jax.random.permutation(key=key, x=52)
        hole_cards = deck_order[:num_hole].reshape(2, max_players).T
        runout = deck_order[num_hole : board_end]
        return DeckState(hole_cards, runout)

    player_metadata = tuple(extract_player_meta(bot) for bot in bots)

    def post_blinds(table_state: TableState) -> TableState:
        sb_seat = table_state.sb_seat
        bb_seat = table_state.bb_seat
        stacks = table_state.stacks
        sb_amt = jnp.minimum(stacks[sb_seat], small_blind)
        bb_amt = jnp.minimum(stacks[bb_seat], big_blind)

        contributed = jnp.zeros((max_players,), dtype=jnp.int32)
        contributed = contributed.at[sb_seat].set(sb_amt)
        contributed = contributed.at[bb_seat].set(bb_amt)

        stacks = stacks.at[sb_seat].add(-sb_amt)
        stacks = stacks.at[bb_seat].add(-bb_amt)

        blind_seats = jnp.stack([sb_seat, bb_seat])
        blind_amts = jnp.stack([sb_amt, bb_amt])
        blind_records = ActionRecord(
            player_id=blind_seats,
            action_type=jnp.array([Action.SMALL_BLIND, Action.BIG_BLIND], dtype=jnp.int32),
            amount=blind_amts,
            is_all_in=stacks[blind_seats] == 0,
            round=jnp.full((2,), Round.PRE_FLOP, dtype=jnp.int32),
            is_valid=blind_amts > 0
        )
        record_indices = table_state.action_count + jnp.arange(2)
        history = jax.tree.map(
            lambda buffer, x: buffer.at[record_indices].set(x),
            table_state.history,
            blind_records
        )

        table_state = table_state._replace(
            history=history,
            action_count=table_state.action_count + 2,
            stacks=stacks,
            contributed=contributed,
            current_bet=jnp.asarray(big_blind, dtype=jnp.int32)
        )
        return table_state._replace(
            current_player=first_after(bb_seat, table_state.to_act_mask)
        )

    def init(
        key: jax.Array,
        seated_mask: jax.Array,
        starting_stacks: jax.Array,
        button: jax.Array | int = 0,
        seat_bot: jax.Array | None = None,
        player_states: tuple | None = None
    ) -> TableState:
        button = jnp.asarray(button, dtype=jnp.int32)
        starting_stacks = jnp.asarray(starting_stacks, dtype=jnp.int32)
        seated_mask = seated_mask & (starting_stacks > 0)
        if seat_bot is None:
            seat_bot = jnp.zeros((max_players,), dtype=jnp.int32)
        seat_bot = jnp.asarray(seat_bot, dtype=jnp.int32)
        
        keys = jax.random.split(key, 2 + max_players)
        master_key, deck_key = keys[0], keys[1]
        player_keys = keys[2:]
        deck = get_deck_state(deck_key)
        history = ActionRecord.init(max_actions=max_actions)

        seat_order = (button + 1 + jnp.arange(max_players)) % max_players
        is_seated_ordered = seated_mask[seat_order]
        seated_rank = jnp.cumsum(is_seated_ordered)
        num_seated = seated_rank[-1]

        is_heads_up = num_seated == 2
        first_seat_idx = jnp.argmax(seated_rank == 1)
        second_seat_idx = jnp.argmax(seated_rank == 2)

        sb_seat = jnp.where(is_heads_up, button, seat_order[first_seat_idx])
        bb_seat = jnp.where(is_heads_up, seat_order[first_seat_idx], seat_order[second_seat_idx])

        if player_states is None:
            player_states = tuple(
                jax.vmap(bot.init_state)(player_keys) for bot in bots
            )

        acted = jnp.zeros(shape=(max_players,), dtype=jnp.bool_)
        action_count = jnp.array(0, dtype=jnp.int32)
        min_raise = jnp.asarray(big_blind, dtype=jnp.int32)

        table_state = TableState(
            deck=deck,
            history=history,
            starting_stacks=starting_stacks,
            seated=seated_mask,
            stacks=starting_stacks,
            in_hand=seated_mask,
            acted=acted,
            button=button,
            sb_seat=sb_seat,
            bb_seat=bb_seat,
            action_count=action_count,
            contributed=jnp.zeros((max_players,), dtype=jnp.int32),
            master_key=master_key,
            player_states=player_states,
            seat_bot=seat_bot,
            current_player=bb_seat,
            current_round=jnp.array(Round.PRE_FLOP, dtype=jnp.int32),
            min_raise=min_raise,
            current_bet=jnp.asarray(0, dtype=jnp.int32)
        )
        return post_blinds(table_state)

    def get_hole_cards_for_player(hole_cards: jax.Array, current_player: jax.Array) -> jax.Array:
        return hole_cards[current_player]
    
    def get_board_at_round(runout: jax.Array, current_round: jax.Array) -> jax.Array:
        visible_counts = jnp.array([0, 3, 4, 5], dtype=jnp.int32)
        num_visible = visible_counts[current_round]
        board_indices = jnp.arange(5, dtype=jnp.int32)
        return jnp.where(
            board_indices < num_visible,
            runout,
            -1
        )

    def get_relative_history(history: ActionRecord, current_player: jax.Array) -> ActionRecord:
        shifted_ids = (history.player_id - current_player) % max_players

        rel_player_id = jnp.where(history.is_valid, shifted_ids, -1)

        return history._replace(player_id=rel_player_id)

    def get_relative_indices(current_player: jax.Array) -> jax.Array:
        return (current_player + jnp.arange(max_players)) % max_players
    
    def get_relative_stacks(stacks: jax.Array, current_player: jax.Array) -> jax.Array:
        return stacks[get_relative_indices(current_player)]

    def get_relative_seated(seated: jax.Array, current_player: jax.Array) -> jax.Array:
        return seated[get_relative_indices(current_player)]

    def get_relative_in_hand(in_hand: jax.Array, current_player: jax.Array) -> jax.Array:
        return in_hand[get_relative_indices(current_player)]
    
    def get_relative_seat(seat: jax.Array, current_player: jax.Array) -> jax.Array:
        return (seat - current_player) % max_players

    def get_relative_contributed(contributed: jax.Array, current_player: jax.Array) -> jax.Array:
        return contributed[get_relative_indices(current_player)]
    
    def get_pot_total(starting_stacks: jax.Array, stacks: jax.Array) -> jax.Array:
        return jnp.sum(starting_stacks - stacks)

    def get_current_player_can_raise(acted: jax.Array, contributed: jax.Array, current_bet: jax.Array, min_raise: jax.Array, stacks: jax.Array, num_can_act: jax.Array, current_player: jax.Array) -> jax.Array:
        to_call = current_bet - contributed[current_player]
        action_reopened = to_call >= min_raise
        acted_already = acted[current_player]
        would_be_all_in = to_call >= stacks[current_player]
        other_players = num_can_act > 1
        return (action_reopened | ~acted_already) & ~would_be_all_in & other_players

    def get_raise_lower_bound(
        contributed: jax.Array,
        current_bet: jax.Array,
        min_raise: jax.Array,
        stacks: jax.Array,
        current_player: jax.Array
    ) -> jax.Array:
        return jnp.minimum(contributed[current_player] + stacks[current_player], current_bet + min_raise)

    def get_raise_upper_bound(
        contributed: jax.Array,
        stacks: jax.Array,
        current_player: jax.Array
    ) -> jax.Array:
        return contributed[current_player] + stacks[current_player]


    def construct_player_table_state(table_state: TableState) -> PlayerTableState:
        hole_cards = get_hole_cards_for_player(table_state.deck.hole_cards, table_state.current_player)
        board = get_board_at_round(table_state.deck.runout, table_state.current_round)
        history = get_relative_history(table_state.history, table_state.current_player)
        stacks = get_relative_stacks(table_state.stacks, table_state.current_player)
        seated = get_relative_seated(table_state.seated, table_state.current_player)
        in_hand = get_relative_in_hand(table_state.in_hand, table_state.current_player)
        button = get_relative_seat(table_state.button, table_state.current_player)
        sb_seat = get_relative_seat(table_state.sb_seat, table_state.current_player)
        bb_seat = get_relative_seat(table_state.bb_seat, table_state.current_player)
        contributed = get_relative_contributed(table_state.contributed, table_state.current_player)
        current_round = table_state.current_round
        min_raise = table_state.min_raise
        current_bet = table_state.current_bet
        pot = get_pot_total(table_state.starting_stacks, table_state.stacks)
        can_raise = get_current_player_can_raise(table_state.acted, table_state.contributed, table_state.current_bet, table_state.min_raise, table_state.stacks, table_state.num_can_act, table_state.current_player)
        raise_lower_bound = get_raise_lower_bound(
            table_state.contributed,
            table_state.current_bet,
            table_state.min_raise,
            table_state.stacks,
            table_state.current_player
        )
        raise_upper_bound = get_raise_upper_bound(
            table_state.contributed, 
            table_state.stacks, 
            table_state.current_player
        )
        call_amount = jnp.minimum(table_state.current_bet, raise_upper_bound)
        raise_lower_bound = jnp.where(can_raise, raise_lower_bound, call_amount)
        raise_upper_bound = jnp.where(can_raise, raise_upper_bound, call_amount)
        bb = jnp.asarray(big_blind, dtype=jnp.int32)

        return PlayerTableState(
            hole_cards=hole_cards,
            board=board,
            history=history,
            stacks=stacks,
            seated=seated,
            in_hand=in_hand,
            button=button,
            sb_seat=sb_seat,
            bb_seat=bb_seat,
            contributed=contributed,
            current_round=current_round,
            min_raise=min_raise,
            current_bet=current_bet,
            pot=pot,
            can_raise=can_raise,
            raise_lower_bound=raise_lower_bound,
            raise_upper_bound=raise_upper_bound,
            big_blind=bb
        )

    def make_act_branch(b: int) -> Callable[[PlayerTableState, tuple, jax.Array, jax.Array], tuple[ActionData, tuple]]:
        def branch(player_table_state: PlayerTableState, player_states: tuple, seat: jax.Array, key: jax.Array) -> tuple[ActionData, tuple]:
            bot_states = player_states[b]
            seat_state = jax.tree.map(lambda x: x[seat], bot_states)
            action, seat_state = bots[b].act(
                player_table_state,
                seat_state,
                key
            )
            bot_states = jax.tree.map(lambda x, y: x.at[seat].set(y), bot_states, seat_state)
            return action, player_states[:b] + (bot_states,) + player_states[b + 1:]
        return branch

    act_branches = tuple(make_act_branch(b) for b in range(len(bots)))

    def get_player_move(table_state: TableState) -> PlayerActionResponse:
        player_states = table_state.player_states
        master_key, action_key = jax.random.split(table_state.master_key, 2)
        player_table_state = construct_player_table_state(table_state)
        action: ActionData
        action, player_states = jax.lax.switch(
            table_state.seat_bot[table_state.current_player],
            act_branches,
            player_table_state,
            player_states,
            table_state.current_player,
            action_key
        )
        return PlayerActionResponse(
            table_state=table_state._replace(
                master_key=master_key,
                player_states=player_states
            ),
            action=action
        )

    # def create_action_record(player_action_response: PlayerActionResponse, table_state: TableState) -> ActionRecord:
    #     player_id = table_state.current_player

    

    # class ActionRecord(NamedTuple):
    #     player_id: jax. Array
    #     action_type: jax.Array
    #     amount: jax.Array
    #     is_all_in: jax.Array
    #     is_valid: jax.Array

    #     @classmethod
    #     def init(cls, max_actions: int) -> "ActionRecord":
    #         return cls(
    #             player_id=jnp.full((max_actions,), fill_value=-1, dtype=jnp.int32),
    #             action_type=jnp.full((max_actions,), fill_value=Action.NO_OP, dtype=jnp.int32),
    #             amount=jnp.zeros((max_actions,), dtype=jnp.int32),
    #             is_all_in=jnp.zeros((max_actions,), dtype=jnp.bool_),
    #             is_valid=jnp.zeros((max_actions,), dtype=jnp.bool_)
    #         )
    def process_player_move(player_action_response: PlayerActionResponse):
        current_player = player_action_response.table_state.current_player
        contributed = player_action_response.table_state.contributed
        stacks = player_action_response.table_state.stacks
        current_bet = player_action_response.table_state.current_bet
        min_raise = player_action_response.table_state.min_raise
        acted = player_action_response.table_state.acted
        num_can_act = player_action_response.table_state.num_can_act
        in_hand = player_action_response.table_state.in_hand
        history = player_action_response.table_state.history
        action_count = player_action_response.table_state.action_count
        current_round = player_action_response.table_state.current_round

        all_in_to = contributed[current_player] + stacks[current_player]
        call_to = jnp.minimum(current_bet, all_in_to)
        raise_to = jnp.clip(
            player_action_response.action.value, 
            get_raise_lower_bound(
                contributed,
                current_bet, 
                min_raise,
                stacks,
                current_player
            ),
            get_raise_upper_bound(
                contributed,
                stacks,
                current_player
            )
        )
        can_raise = get_current_player_can_raise(
            acted,
            contributed,
            current_bet,
            min_raise,
            stacks,
            num_can_act,
            current_player
        )
        is_fold = (player_action_response.action.type == Action.FOLD) | (player_action_response.action.type == Action.NO_OP)
        is_raise = (player_action_response.action.type == Action.RAISE) & can_raise
        target_contribution = jnp.where(
            is_fold,
            contributed[current_player],
            jnp.where(
                is_raise,
                raise_to,
                call_to
            )
        )
        delta = target_contribution - contributed[current_player]
        increment = target_contribution - current_bet
        full_raise = increment >= min_raise

        player_id = current_player
        action_type = jnp.where(
            is_fold,
            Action.FOLD,
            jnp.where(
                is_raise,
                Action.RAISE,
                Action.CHECK_CALL
            )
        )
        amount = target_contribution
        is_all_in = delta == stacks[current_player]
        is_valid = jnp.asarray(True, dtype=jnp.bool_)
        action_record = ActionRecord(
            player_id=player_id,
            action_type=action_type,
            amount=amount,
            is_all_in=is_all_in,
            round=current_round,
            is_valid=is_valid
        )

        history = jax.tree.map(
            lambda buffer, x: buffer.at[action_count].set(x),
            history,
            action_record
        )
        action_count = action_count + 1
        stacks = stacks.at[current_player].add(-delta)
        contributed = contributed.at[current_player].set(target_contribution)
        in_hand = in_hand.at[current_player].set(in_hand[current_player] & ~is_fold)
        acted = acted.at[current_player].set(True)
        min_raise = jnp.where(
            full_raise,
            increment,
            min_raise
        )
        current_bet = jnp.maximum(current_bet, target_contribution)

        return player_action_response.table_state._replace(
            history=history,
            action_count=action_count,
            stacks=stacks,
            contributed=contributed,
            in_hand=in_hand,
            acted=acted,
            min_raise=min_raise,
            current_bet=current_bet
        )

    def is_street_over(table_state: TableState) -> jax.Array:
        return ~jnp.any(table_state.to_act_mask)

    def is_hand_over(table_state: TableState) -> jax.Array:
        river_complete = is_street_over(table_state) & (table_state.current_round == Round.RIVER)
        fold_out = jnp.sum(table_state.eligible_mask) <= 1
        return river_complete | fold_out

    def get_hand_ranks(table_state: TableState) -> jax.Array:
        board = jnp.broadcast_to(table_state.deck.runout, (max_players, 5))
        cards = jnp.concatenate([table_state.deck.hole_cards, board], axis=1)
        return evaluate_hands(cards)

    def get_payouts(table_state: TableState, hand_ranks: jax.Array) -> jax.Array:
        committed = table_state.starting_stacks - table_state.stacks
        levels = jnp.sort(committed)
        payouts = jnp.zeros((max_players,), dtype=jnp.int32)
        button_order = (table_state.button + jnp.arange(1, max_players + 1)) % max_players
        previous_level = jnp.asarray(0, dtype=jnp.int32)
        for i in range(max_players):
            level = levels[i]
            pot_slice = jnp.sum(jnp.clip(committed, previous_level, level) - previous_level)
            contenders = table_state.eligible_mask & (committed >= level)
            best_rank = jnp.max(jnp.where(contenders, hand_ranks, -1))
            winners = contenders & (hand_ranks == best_rank)
            num_winners = jnp.sum(winners, dtype=jnp.int32)
            share = pot_slice // jnp.maximum(num_winners, 1)
            odd_chips = pot_slice - share * num_winners
            winners_ordered = winners[button_order]
            gets_odd_chip_ordered = winners_ordered & (jnp.cumsum(winners_ordered) <= odd_chips)
            gets_odd_chip = jnp.zeros((max_players,), dtype=jnp.bool_).at[button_order].set(gets_odd_chip_ordered)
            payouts = payouts + jnp.where(winners, share, 0) + gets_odd_chip
            previous_level = level
        return payouts

    # def get_next_button(table_state: TableState) -> jax.Array:
    #     # needs dead button stuff
    #     button = table_state.button
    #     has_chips = table_state.stacks > 0
    #     button_relative_indices = (button + jnp.arange(1, max_players + 1)) % max_players
    #     has_chips_relative= has_chips[button_relative_indices]
    #     next_button = button_relative_indices[jnp.argmax(has_chips_relative)]
    #     return next_button

    # def get_next_player(table_state: TableState) -> jax.Array:
    #     street_over = is_street_over(table_state)
    #     current_player = table_state.current_player
    #     current_player_relative_indices = (current_player + jnp.arange(1, max_players + 1)) % max_players
    #     can_act_relative = table_state.can_act_mask[current_player_relative_indices]
    #     next_player = current_player_relative_indices[jnp.argmax(can_act_relative)]
    #     return jnp.where(
    #         ~street_over,
    #         next_player,
    #         jnp.asarray(0, dtype=jnp.int32)
    #     )

    def first_after(seat: jax.Array, mask: jax.Array) -> jax.Array:
        order = (seat + jnp.arange(1, max_players + 1)) % max_players
        return order[jnp.argmax(mask[order])]

    def get_next_player(table_state: TableState) -> jax.Array:
        return first_after(table_state.current_player, table_state.to_act_mask)

    def get_next_positions(table_state: TableState) -> tuple[jax.Array, jax.Array, jax.Array]:
        live = table_state.seated & (table_state.stacks > 0)
        next_bb_seat = first_after(table_state.bb_seat, live)
        # Heads-up: the button posts the small blind. The big blind still moves
        # forward one live seat, so nobody skips or repeats it going from 3 to 2.
        is_heads_up = jnp.sum(live) == 2
        heads_up_button = first_after(next_bb_seat, live)
        next_button = jnp.where(is_heads_up, heads_up_button, table_state.sb_seat)
        next_sb_seat = jnp.where(is_heads_up, heads_up_button, table_state.bb_seat)
        return next_button, next_sb_seat, next_bb_seat

    def next_player(table_state: TableState) -> TableState:
        return table_state._replace(
            current_player=get_next_player(table_state)
        )
    
    def next_street(table_state: TableState) -> TableState:
        next_round = (table_state.current_round + 1) % (Round.RIVER + 1)
        # next_player = get_next_player(table_state)
        new_round_contribution = jnp.zeros((max_players,), dtype=jnp.int32)
        new_current_bet = jnp.asarray(0, dtype=jnp.int32)
        new_min_raise = jnp.asarray(big_blind, dtype=jnp.int32)
        new_acted = jnp.zeros(shape=(max_players,), dtype=jnp.bool_)
        new_state = table_state._replace(
            acted=new_acted,
            current_round=next_round,
            contributed=new_round_contribution,
            current_bet=new_current_bet,
            min_raise=new_min_raise
        )
        return new_state._replace(
            current_player=first_after(new_state.button, new_state.to_act_mask)
        )

    def next_hand(table_state: TableState) -> TableState:
        stacks = table_state.stacks + get_payouts(table_state, get_hand_ranks(table_state))
        seated = table_state.seated & (stacks > 0)
        table_state = table_state._replace(stacks=stacks, seated=seated)
        button, sb_seat, bb_seat = get_next_positions(table_state)
        master_key, deck_key = jax.random.split(table_state.master_key)
        table_state = table_state._replace(
            deck=get_deck_state(deck_key),
            history=ActionRecord.init(max_actions=max_actions),
            starting_stacks=stacks,
            in_hand=seated,
            acted=jnp.zeros((max_players,), dtype=jnp.bool_),
            button=button,
            sb_seat=sb_seat,
            bb_seat=bb_seat,
            action_count=jnp.asarray(0, dtype=jnp.int32),
            contributed=jnp.zeros((max_players,), dtype=jnp.int32),
            master_key=master_key,
            current_round=jnp.asarray(Round.PRE_FLOP, dtype=jnp.int32),
            min_raise=jnp.asarray(big_blind, dtype=jnp.int32),
            current_bet=jnp.asarray(0, dtype=jnp.int32)
        )
        table_finished = table_state.num_seated <= 1
        return jax.tree.map(
            lambda finished, posted: jnp.where(table_finished, finished, posted),
            table_state,
            post_blinds(table_state)
        )

    def take_action(table_state: TableState) -> TableState:
        return next_player(process_player_move(get_player_move(table_state)))

    def step(table_state: TableState) -> TableState:
        table_finished = table_state.num_seated <= 1
        case = jnp.where(
            table_finished,
            0,
            jnp.where(
                is_hand_over(table_state),
                1,
                jnp.where(
                    is_street_over(table_state),
                    2,
                    3
                )
            )
        )
        return jax.lax.switch(
            case,
            (lambda table_state: table_state, next_hand, next_street, take_action),
            table_state
        )

    





# class TableState(NamedTuple):
#     deck: DeckState
#     history: ActionRecord
#     starting_stacks: jax.Array
#     seated: jax.Array
#     stacks: jax.Array
#     in_hand: jax.Array
#     acted: jax.Array
#     button: jax.Array
#     action_count: jax.Array
#     contributed: jax.Array
#     master_key: jax.Array
#     player_states: tuple
#     current_player: jax.Array
#     current_round: jax.Array
#     min_raise: jax.Array
#     current_bet: jax.Array


    return PokerEnv(
        init=init, 
        get_payouts=get_payouts,
        step=step,
        player_metadata=player_metadata
    )



# class PlayerTableState(NamedTuple):
#     hole_cards: jax.Array
#     board: jax.Array
#     history: ActionRecord
#     stacks: jax.Array
#     in_hand: jax.Array
#     button: jax.Array
#     contributed: jax.Array
#     current_round: jax.Array
#     min_raise: jax.Array
#     current_bet: jax.Array
#     pot: jax.Array
#     can_raise: jax.Array
#     raise_lower_bound: jax.Array
#     raise_upper_bound: jax.Array
#     big_blind: jax.Array

    # class TableState(NamedTuple):
    #     deck: DeckState
    #     history: ActionRecord
    #     starting_stacks: jax.Array
    #     stacks: jax.Array
    #     in_hand: jax.Array
    #     acted: jax.Array
    #     button: jax.Array
    #     action_count: jax.Array
    #     contributed: jax.Array
    #     master_key: jax.Array
    #     player_states: tuple
    #     current_player: jax.Array
    #     current_round: jax.Array
    #     min_raise: jax.Array
    #     current_bet: jax.Array

    # @jax.jit
    # def init(key, seated_mask, starting_stacks, button, player_states=None):
    #     keys = jax.random.split(key, 2 + max_players)
    #     master_key, deck_key = keys[0], keys[1]
    #     player_keys = keys[2:]
    #     deck = get_deck_state(deck_key)
    #     history = ActionRecord.init(max_actions=max_actions)

    #     seat_order = (button + 1 + jnp.arange(max_players)) % max_players
    #     is_seated_ordered = seated_mask[seat_order]
    #     seated_rank = jnp.cumsum(is_seated_ordered)
    #     num_seated = seated_rank[-1]

    #     is_heads_up = num_seated == 2
    #     first_seat_idx = jnp.argmax(seated_rank == 1)
    #     second_seat_idx = jnp.argmax(seated_rank == 2)

    #     sb_seat = jnp.where(is_heads_up, button, seat_order[first_seat_idx])
    #     bb_seat = jnp.where(is_heads_up, seat_order[first_seat_idx], seat_order[second_seat_idx])

    #     sb_amt = jnp.minimum(starting_stacks[sb_seat], small_blind)
    #     bb_amt = jnp.minimum(starting_stacks[bb_seat], big_blind)

    #     bets = jnp.zeros((max_players,), dtype=jnp.int32)
    #     bets = bets.at[sb_seat].set(sb_amt)
    #     bets = bets.at[bb_seat].set(bb_amt)

    #     stacks = starting_stacks.at[sb_seat].add(-sb_amt)
    #     stacks = stacks.at[bb_seat].add(-bb_amt)

    #     if player_states is None:
    #         player_states = tuple(
    #             player.init_state(player_keys[i]) for i, player in enumerate(players)
    #         )

    #     third_seat_idx = jnp.argmax(seated_rank == 3)
    #     utg_seat = seat_order[third_seat_idx]
    #     current_player = jnp.where(is_heads_up, sb_seat, utg_seat)

    #     return TableState(
    #         deck=deck,
    #         history=history,
    #         seated_mask=seated_mask,
    #         starting_stacks=starting_stacks,
    #         stacks=stacks,
    #         active_mask=seated_mask,
    #         button=button,
    #         pot=jnp.array(0, dtype=jnp.int32),
    #         action_count=jnp.array(0, dtype=jnp.int32),
    #         small_blind=jnp.array(small_blind, dtype=jnp.int32),
    #         big_blind=jnp.array(big_blind, dtype=jnp.int32),
    #         bets=bets,
    #         master_key=master_key,
    #         player_states=player_states,
    #         current_player=current_player,
    #         current_round=jnp.array(Round.PRE_FLOP, dtype=jnp.int32),
    #         min_raise=jnp.array(big_blind, dtype=jnp.int32)
    #     )

# class RoundState(NamedTuple):
#     action_history: jax.Array
#     action_count: jax.Array

# # class TableState(NamedTuple):
# #     deck: DeckState
# #     history: ActionRecord
# #     seated_mask: jax.Array
# #     starting_stacks: jax.Array
# #     stacks: jax.Array
# #     active_mask: jax.Array
# #     button: jax.Array
# #     pot: jax.Array
# #     action_count: jax.Array
# #     small_blind: jax.Array
# #     big_blind: jax.Array
# #     bets: jax.Array
# #     master_key: jax.Array
# #     player_states: tuple
# #     current_player: jax.Array
# #     current_round: jax.Array
# #     min_raise: jax.Array



# class PlayerTableState(NamedTuple):
#     hole_cards: jax.Array
#     history: ActionRecord
#     stacks: jax.Array
#     active_players: jax.Array
#     button: jax.Array
#     pot: jax.Array
#     current_bet: jax.Array
#     big_blind: jax.Array
#     min_raise: jax.Array
#     current_round: jax.Array
#     bets: jax.Array
#     current_board: jax.Array

# class ActionData(NamedTuple):
#     type: jax.Array
#     value: jax.Array

# class PlayerActionResponse(NamedTuple):
#     player_state: jax.Array
#     action: ActionData

# def extract_player_meta(player):
#     return player.metadata()

# def make_poker_env(players, max_players: int = 9, max_actions: int = 100, small_blind=1, big_blind=2):
#     num_hole = 2 * max_players
#     board_end = num_hole + 5

#     def get_deck_state(key):
#         deck_order = jax.random.permutation(key=key, x=52)
#         hole_cards = deck_order[:num_hole].reshape(2, max_players).T
#         runout = deck_order[num_hole : board_end]
#         return DeckState(hole_cards, runout)

#     player_metadata = tuple(extract_player_meta(player) for player in players)

#     @jax.jit
#     def init(key, seated_mask, starting_stacks, button, player_states=None):
#         keys = jax.random.split(key, 2 + max_players)
#         master_key, deck_key = keys[0], keys[1]
#         player_keys = keys[2:]
#         deck = get_deck_state(deck_key)
#         history = ActionRecord.init(max_actions=max_actions)

#         seat_order = (button + 1 + jnp.arange(max_players)) % max_players
#         is_seated_ordered = seated_mask[seat_order]
#         seated_rank = jnp.cumsum(is_seated_ordered)
#         num_seated = seated_rank[-1]

#         is_heads_up = num_seated == 2
#         first_seat_idx = jnp.argmax(seated_rank == 1)
#         second_seat_idx = jnp.argmax(seated_rank == 2)

#         sb_seat = jnp.where(is_heads_up, button, seat_order[first_seat_idx])
#         bb_seat = jnp.where(is_heads_up, seat_order[first_seat_idx], seat_order[second_seat_idx])

#         sb_amt = jnp.minimum(starting_stacks[sb_seat], small_blind)
#         bb_amt = jnp.minimum(starting_stacks[bb_seat], big_blind)

#         bets = jnp.zeros((max_players,), dtype=jnp.int32)
#         bets = bets.at[sb_seat].set(sb_amt)
#         bets = bets.at[bb_seat].set(bb_amt)

#         stacks = starting_stacks.at[sb_seat].add(-sb_amt)
#         stacks = stacks.at[bb_seat].add(-bb_amt)

#         if player_states is None:
#             player_states = tuple(
#                 player.init_state(player_keys[i]) for i, player in enumerate(players)
#             )

#         third_seat_idx = jnp.argmax(seated_rank == 3)
#         utg_seat = seat_order[third_seat_idx]
#         current_player = jnp.where(is_heads_up, sb_seat, utg_seat)

#         return TableState(
#             deck=deck,
#             history=history,
#             seated_mask=seated_mask,
#             starting_stacks=starting_stacks,
#             stacks=stacks,
#             active_mask=seated_mask,
#             button=button,
#             pot=jnp.array(0, dtype=jnp.int32),
#             action_count=jnp.array(0, dtype=jnp.int32),
#             small_blind=jnp.array(small_blind, dtype=jnp.int32),
#             big_blind=jnp.array(big_blind, dtype=jnp.int32),
#             bets=bets,
#             master_key=master_key,
#             player_states=player_states,
#             current_player=current_player,
#             current_round=jnp.array(Round.PRE_FLOP, dtype=jnp.int32),
#             min_raise=jnp.array(big_blind, dtype=jnp.int32)
#         )
    
#     # def get_hole_cards(hole_cards: jax.Array, current_player: jax.Array) -> jax.Array:
#     #     return hole_cards[current_player]

    # def get_history(history: ActionRecord, current_player: jax.Array):
    #     shifted_ids = (history.player_id - current_player) % max_players

    #     rel_player_id = jnp.where(history.is_valid, shifted_ids, -1)

    #     return history._replace(player_id=rel_player_id)

#     def get_stacks(stacks: jax.Array, current_player: jax.Array):
#         relative_indices = (current_player + jnp.arange(max_players)) % max_players
#         return stacks[relative_indices]

#     def get_active_players(active_players: jax.Array, current_player: jax.Array):
#         relative_indices = (current_player + jnp.arange(max_players)) % max_players
#         return active_players[relative_indices]

    # def get_button(button: jax.Array, current_player: jax.Array):
    #     return (button - current_player) % max_players

#     # def get_pot(pot: jax.Array):
#     #     return pot

#     def get_current_bet(bets: jax.Array):
#         return jnp.max(bets)

#     # def get_big_blind(big_blind: jax.Array):
#     #     return big_blind

#     def get_bets(bets: jax.Array, current_player: jax.Array):
#         relative_indices = (current_player + jnp.arange(max_players)) % max_players
#         return bets[relative_indices]

#     def get_current_board(runout: jax.Array, current_round: jax.Array):
        # visible_counts = jnp.array([0, 3, 4, 5], dtype=jnp.int32)
        # num_visible = visible_counts[current_round]
        # board_indices = jnp.arange(5, dtype=jnp.int32)
        # return jnp.where(
        #     board_indices < num_visible,
        #     runout,
        #     -1
        # )

#     def filter_player_state(table_state: TableState):
#         return PlayerTableState(
#             hole_cards=table_state.deck.hole_cards[table_state.current_player],
#             history=get_history(table_state.history, table_state.current_player),
#             stacks=get_stacks(table_state.stacks, table_state.current_player),
#             active_players=get_active_players(table_state.active_mask, table_state.current_player),
#             button=get_button(table_state.button, table_state.current_player),
#             pot=table_state.pot,
#             current_bet=get_current_bet(table_state.bets),
#             big_blind=table_state.big_blind,
#             min_raise=table_state.min_raise,
#             current_round=table_state.current_round,
#             bets=get_bets(table_state.bets, table_state.current_player),
#             current_board=get_current_board(table_state.deck.runout, table_state.current_round)
#         )

#     def get_action(table_state: TableState):
#         return players[table_state.current_player].act(filter_player_state(table_state), table_state.player_states[table_state.current_player])

#     # def get_call_or_raise_value(table_state: TableState, action_data: ActionData):
#     #     to_call = get_current_bet(table_state.bets) - table_state.bets[table_state.current_player]
#     #     return jnp.where(action_data.type == Action.CHECK_CALL, to_call, )

#     # def clamp_action_to_nearest_legal(table_state: TableState, action_data: ActionData):
#     #     clamped_action = jnp.where(
#     #         action_data.type == Action.RAISE,
#     #         jnp.where(
#     #             action_data.value - get_current_bet(table_state.bets) >= table_state.min_raise,
#     #             action_data,
#     #             ActionData(
#     #                 type=Action.RAISE,
#     #                 value=get_current_bet(table_state.bets) + table_state.min_raise
#     #             )
#     #         ),
#     #         action_data
#     #     )
#     #     return clamped_action

#     def clamp_action_to_nearest_legal(table_state: TableState, action_data: ActionData) -> ActionData:
#         current_bet = get_current_bet(table_state.bets)
        
#         # 1. Minimum legal raise threshold
#         min_legal_raise = current_bet + table_state.min_raise
        
#         # 2. Maximum possible commitment (all-in)
#         hero_max = table_state.bets[table_state.current_player] + table_state.stacks[table_state.current_player]
        
#         # Effective lower bound cannot exceed their all-in stack
#         effective_min = jnp.minimum(min_legal_raise, hero_max)
        
#         # Clamp the proposed raise amount into [effective_min, hero_max]
#         clamped_raise_val = jnp.clip(action_data.value, effective_min, hero_max)
        
#         # Only modify the payload if the action type was RAISE; otherwise leave value as-is
#         new_value = jnp.where(
#             action_data.type == Action.RAISE,
#             clamped_raise_val,
#             action_data.value
#         )
        
#         return action_data._replace(value=new_value)

#     def make_new_action_record(table_state: TableState, player_action: ActionData):
#         hero = table_state.current_player
#         hero_bet = table_state.bets[hero]
#         hero_stack = table_state.stacks[hero]

#         action_type = player_action.type
#         raw_amount = player_action.value

#         # Use | instead of python 'or'
#         is_committing = (action_type == Action.RAISE) | (action_type == Action.CHECK_CALL)

#         # Clean amount: non-committing actions (FOLD/NO_OP) commit 0 chips
#         amount = jnp.where(is_committing, raw_amount, 0)

#         # Boolean logic without nested jnp.where
#         is_all_in = is_committing & (raw_amount - hero_bet >= hero_stack)

#         return ActionRecord(
#             player_id=hero,
#             action_type=action_type,
#             amount=amount,
#             is_all_in=is_all_in,
#             is_valid=jnp.bool_(True)
#         )

#     def make_new_player_bet_amount(table_state: TableState, player_action: ActionData) -> jax.Array:
#         hero = table_state.current_player
#         hero_bet = table_state.bets[hero]
#         hero_stack = table_state.stacks[hero]
#         current_bet = get_current_bet(table_state.bets)

#         # CALL matches current_bet up to total stack ceiling
#         call_amount = jnp.minimum(hero_bet + hero_stack, current_bet)

#         # Branchless selection across the three outcomes:
#         # 1. CHECK_CALL -> call_amount
#         # 2. RAISE -> player_action.value (already clamped to [min_raise, hero_max])
#         # 3. FOLD / NO_OP -> hero_bet (chips committed earlier on this street stay on the table)
#         return jnp.where(
#             player_action.type == Action.CHECK_CALL,
#             call_amount,
#             jnp.where(
#                 player_action.type == Action.RAISE,
#                 player_action.value,
#                 hero_bet
#             )
#         )



#     # def get_new_action_record(table_state: TableState, player_action: ActionData):
#     #     player_id = table_state.current_player
#     #     action_type = player_action.type
#     #     current_gap = get_current_bet(table_state.bets) - table_state.bets[table_state.current_player]
#     #     amount = jnp.where(
#     #         player_action.type == Action.CHECK_CALL,
#     #         get_current_bet(table_state.bets),
#     #         jnp.where(
#     #             player_action.type == Action.RAISE,
#     #             player_action.value,
#     #             jnp.array(0, dtype=jnp.int32)
#     #         )
#     #     )
#     #     is_all_in = jnp.where(
#     #         player_action
#     #     )
    
#     # def update_table_state_in_response_to_action(table_state: TableState, player_response: PlayerActionResponse):

#         # new_history = table_state.history.at[table_state.action_count].set(
#         #     ActionRecord(
#         #         player_id=table_state.current_player,
#         #         action_type=player_response.action.type,
#         #         amount=jnp.where(player_response.action.type == Action.CHECK_CALL, get_current_bet(table_state.bets), jnp.where(player_response.action.type == Action.RAISE, player_response.action.value, 0)),
#         #         is_all_in=jnp.where(
#         #              table_state.bets[table_state.current_player])
#         #     )
#         # )
#     # player_id: jax. Array
#     # action_type: jax.Array
#     # amount: jax.Array
#     # is_all_in: jax.Array
#     # is_valid: jax.Array
#     return PokerEnv(init=init, player_metadata=player_metadata)













class TestBot:
    def __init__(self):
        pass
    def init_state(self, key):
        return None
    
    def act(self, obs, state, key):
        action = ActionData(
            type=jnp.asarray(Action.CHECK_CALL, dtype=jnp.int32),
            value=jnp.asarray(0, dtype=jnp.int32)
        )
        return action, state
    
    def observe(self, data):
        return None
    
    def metadata(self):
        return {
            "bot_type": "test_bot"
        }

class RandomBot:
    def init_state(self, key: jax.Array) -> None:
        return None

    def act(self, obs: PlayerTableState, state: None, key: jax.Array) -> tuple[ActionData, None]:
        type_key, value_key, shove_key = jax.random.split(key, 3)
        action_type = jax.random.choice(
            type_key,
            jnp.array([Action.FOLD, Action.CHECK_CALL, Action.RAISE], dtype=jnp.int32),
            p=jnp.array([0.15, 0.6, 0.25])
        )
        small_raise_cap = jnp.minimum(obs.raise_upper_bound, obs.raise_lower_bound + 3 * obs.big_blind)
        value = jnp.where(
            jax.random.bernoulli(shove_key, 0.1),
            obs.raise_upper_bound,
            jax.random.randint(value_key, (), obs.raise_lower_bound, small_raise_cap + 1)
        )
        return ActionData(type=action_type, value=value), state

    def metadata(self) -> dict:
        return {
            "bot_type": "random_bot"
        }




# if __name__ == "__main__":
#     # print(ID_TO_CARD)
#     # print(CARD_TO_ID)
#     # get_deck_state = make_poker_env(9).get_deck_state
#     # key = jax.random.PRNGKey(0)
#     # print(get_deck_state(key))
#     # test_bot = TestBot()
#     # print(make_poker_env((test_bot,), max_players=1))