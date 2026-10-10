from msilib import Table
from collections import namedtuple
import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable

Card = namedtuple('Card', ['Rank', 'Suit'])

RANKS = "2 3 4 5 6 7 8 9 10 J Q K A".split()
SUITS = "Diamonds Clubs Hearts Spades".split()
ID_TO_CARD = [(rank, suit) for suit in SUITS for rank in RANKS]
CARD_TO_ID = {
    card: i for i, card in enumerate(ID_TO_CARD)
}

class DeckState(NamedTuple):
    hole_cards: jax.Array
    runout: jax.Array

class Action:
    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2
    NO_OP = 3

class ActionRecord(NamedTuple):
    player_id: jax. Array
    action_type: jax.Array
    amount: jax.Array
    is_all_in: jax.Array
    is_valid: jax.Array

    @classmethod
    def init(cls, max_actions: int) -> "ActionRecord":
        return cls(
            player_id=jnp.full((max_actions,), fill_value=-1, dtype=jnp.int32),
            action_type=jnp.full((max_actions,), fill_value=Action.NO_OP, dtype=jnp.int32),
            amount=jnp.zeros((max_actions,), dtype=jnp.int32),
            is_all_in=jnp.zeros((max_actions,), dtype=jnp.bool_),
            is_valid=jnp.zeros((max_actions,), dtype=jnp.bool_)
        )

class Round:
    PRE_FLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

class RoundState(NamedTuple):
    action_history: jax.Array
    action_count: jax.Array

class TableState(NamedTuple):
    deck: DeckState
    history: ActionRecord
    seated_mask: jax.Array
    starting_stacks: jax.Array
    stacks: jax.Array
    active_mask: jax.Array
    button: jax.Array
    pot: jax.Array
    action_count: jax.Array
    small_blind: jax.Array
    big_blind: jax.Array
    bets: jax.Array
    master_key: jax.Array
    player_states: tuple
    current_player: jax.Array
    current_round: jax.Array
    min_raise: jax.Array

class PokerEnv(NamedTuple):
    # get_deck_state: Callable[[jax.Array], DeckState]
    # action_sequence: jax.Array
    init: Callable[[jax.Array, jax.Array, jax.Array, jax.Array, tuple], TableState]
    player_metadata: tuple

class PlayerTableState(NamedTuple):
    hole_cards: jax.Array
    history: ActionRecord
    stacks: jax.Array
    active_players: jax.Array
    button: jax.Array
    pot: jax.Array
    current_bet: jax.Array
    big_blind: jax.Array
    min_raise: jax.Array
    current_round: jax.Array
    bets: jax.array
    current_board: jax.Array

class ActionData(NamedTuple):
    type: jax.Array
    value: jax.Array

class PlayerActionResponse(NamedTuple):
    player_state: jax.Array
    action: ActionData

def extract_player_meta(player):
    return player.metadata()

def make_poker_env(players, max_players: int = 9, max_actions: int = 100, small_blind=1, big_blind=2):
    num_hole = 2 * max_players
    board_end = num_hole + 5

    def get_deck_state(key):
        deck_order = jax.random.permutation(key=key, x=52)
        hole_cards = deck_order[:num_hole].reshape(2, max_players).T
        runout = deck_order[num_hole : board_end]
        return DeckState(hole_cards, runout)

    player_metadata = tuple(extract_player_meta(player) for player in players)

    @jax.jit
    def init(key, seated_mask, starting_stacks, button, player_states=None):
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

        sb_amt = jnp.minimum(starting_stacks[sb_seat], small_blind)
        bb_amt = jnp.minimum(starting_stacks[bb_seat], big_blind)

        bets = jnp.zeros((max_players,), dtype=jnp.int32)
        bets = bets.at[sb_seat].set(sb_amt)
        bets = bets.at[bb_seat].set(bb_amt)

        stacks = starting_stacks.at[sb_seat].add(-sb_amt)
        stacks = stacks.at[bb_seat].add(-bb_amt)

        if player_states is None:
            player_states = tuple(
                player.init_state(player_keys[i]) for i, player in enumerate(players)
            )

        third_seat_idx = jnp.argmax(seated_rank == 3)
        utg_seat = seat_order[third_seat_idx]
        current_player = jnp.where(is_heads_up, sb_seat, utg_seat)

        return TableState(
            deck=deck,
            history=history,
            seated_mask=seated_mask,
            starting_stacks=starting_stacks,
            stacks=stacks,
            active_mask=seated_mask,
            button=button,
            pot=jnp.array(0, dtype=jnp.int32),
            action_count=jnp.array(0, dtype=jnp.int32),
            small_blind=jnp.array(small_blind, dtype=jnp.int32),
            big_blind=jnp.array(big_blind, dtype=jnp.int32),
            bets=bets,
            master_key=master_key,
            player_states=player_states,
            current_player=current_player,
            current_round=jnp.array(Round.PRE_FLOP, dtype=jnp.int32),
            min_raise=jnp.array(big_blind, dtype=jnp.int32)
        )
    
    # def get_hole_cards(hole_cards: jax.Array, current_player: jax.Array) -> jax.Array:
    #     return hole_cards[current_player]

    def get_history(history: ActionRecord, current_player: jax.Array):
        shifted_ids = (history.player_id - current_player) % max_players

        rel_player_id = jnp.where(history.is_valid, shifted_ids, -1)

        return history._replace(player_id=rel_player_id)

    def get_stacks(stacks: jax.Array, current_player: jax.Array):
        relative_indices = (current_player + jnp.arange(max_players)) % max_players
        return stacks[relative_indices]

    def get_active_players(active_players: jax.Array, current_player: jax.Array):
        relative_indices = (current_player + jnp.arange(max_players)) % max_players
        return active_players[relative_indices]

    def get_button(button: jax.Array, current_player: jax.Array):
        return (button - current_player) % max_players

    # def get_pot(pot: jax.Array):
    #     return pot

    def get_current_bet(bets: jax.Array):
        return jnp.max(bets)

    # def get_big_blind(big_blind: jax.Array):
    #     return big_blind

    def get_bets(bets: jax.Array, current_player: jax.Array):
        relative_indices = (current_player + jnp.arange(max_players)) % max_players
        return bets[relative_indices]

    def get_current_board(runout: jax.Array, current_round: jax.Array):
        visible_counts = jnp.array([0, 3, 4, 5], dtype=jnp.int32)
        num_visible = visible_counts[current_round]
        board_indices = jnp.arange(5, dtype=jnp.int32)
        return jnp.where(
            board_indices < num_visible,
            runout,
            -1
        )

    def filter_player_state(table_state: TableState):
        return PlayerTableState(
            hole_cards=table_state.deck.hole_cards[table_state.current_player],
            history=get_history(table_state.history, table_state.current_player),
            stacks=get_stacks(table_state.stacks, table_state.current_player),
            active_players=get_active_players(table_state.active_mask, table_state.current_player),
            button=get_button(table_state.button, table_state.current_player),
            pot=table_state.pot,
            current_bet=get_current_bet(table_state.bets),
            big_blind=table_state.big_blind,
            min_raise=table_state.min_raise,
            current_round=table_state.current_round,
            bets=get_bets(table_state.bets, table_state.current_player),
            current_board=get_current_board(table_state.deck.runout, table_state.current_round)
        )

    def get_action(table_state: TableState):
        return players[table_state.current_player].act(filter_player_state(table_state), table_state.player_states[table_state.current_player])

    # def get_call_or_raise_value(table_state: TableState, action_data: ActionData):
    #     to_call = get_current_bet(table_state.bets) - table_state.bets[table_state.current_player]
    #     return jnp.where(action_data.type == Action.CHECK_CALL, to_call, )

    # def clamp_action_to_nearest_legal(table_state: TableState, action_data: ActionData):
    #     clamped_action = jnp.where(
    #         action_data.type == Action.RAISE,
    #         jnp.where(
    #             action_data.value - get_current_bet(table_state.bets) >= table_state.min_raise,
    #             action_data,
    #             ActionData(
    #                 type=Action.RAISE,
    #                 value=get_current_bet(table_state.bets) + table_state.min_raise
    #             )
    #         ),
    #         action_data
    #     )
    #     return clamped_action

    def clamp_action_to_nearest_legal(table_state: TableState, action_data: ActionData) -> ActionData:
        current_bet = get_current_bet(table_state.bets)
        
        # 1. Minimum legal raise threshold
        min_legal_raise = current_bet + table_state.min_raise
        
        # 2. Maximum possible commitment (all-in)
        hero_max = table_state.bets[table_state.current_player] + table_state.stacks[table_state.current_player]
        
        # Effective lower bound cannot exceed their all-in stack
        effective_min = jnp.minimum(min_legal_raise, hero_max)
        
        # Clamp the proposed raise amount into [effective_min, hero_max]
        clamped_raise_val = jnp.clip(action_data.value, effective_min, hero_max)
        
        # Only modify the payload if the action type was RAISE; otherwise leave value as-is
        new_value = jnp.where(
            action_data.type == Action.RAISE,
            clamped_raise_val,
            action_data.value
        )
        
        return action_data._replace(value=new_value)

    def make_new_action_record(table_state: TableState, player_action: ActionData):
        hero = table_state.current_player
        hero_bet = table_state.bets[hero]
        hero_stack = table_state.stacks[hero]

        action_type = player_action.type
        raw_amount = player_action.value

        # Use | instead of python 'or'
        is_committing = (action_type == Action.RAISE) | (action_type == Action.CHECK_CALL)

        # Clean amount: non-committing actions (FOLD/NO_OP) commit 0 chips
        amount = jnp.where(is_committing, raw_amount, 0)

        # Boolean logic without nested jnp.where
        is_all_in = is_committing & (raw_amount - hero_bet >= hero_stack)

        return ActionRecord(
            player_id=hero,
            action_type=action_type,
            amount=amount,
            is_all_in=is_all_in,
            is_valid=jnp.bool_(True)
        )

    def make_new_player_bet_amount(table_state: TableState, player_action: ActionData) -> jax.Array:
        hero = table_state.current_player
        hero_bet = table_state.bets[hero]
        hero_stack = table_state.stacks[hero]
        current_bet = get_current_bet(table_state.bets)

        # CALL matches current_bet up to total stack ceiling
        call_amount = jnp.minimum(hero_bet + hero_stack, current_bet)

        # Branchless selection across the three outcomes:
        # 1. CHECK_CALL -> call_amount
        # 2. RAISE -> player_action.value (already clamped to [min_raise, hero_max])
        # 3. FOLD / NO_OP -> hero_bet (chips committed earlier on this street stay on the table)
        return jnp.where(
            player_action.type == Action.CHECK_CALL,
            call_amount,
            jnp.where(
                player_action.type == Action.RAISE,
                player_action.value,
                hero_bet
            )
        )



    # def get_new_action_record(table_state: TableState, player_action: ActionData):
    #     player_id = table_state.current_player
    #     action_type = player_action.type
    #     current_gap = get_current_bet(table_state.bets) - table_state.bets[table_state.current_player]
    #     amount = jnp.where(
    #         player_action.type == Action.CHECK_CALL,
    #         get_current_bet(table_state.bets),
    #         jnp.where(
    #             player_action.type == Action.RAISE,
    #             player_action.value,
    #             jnp.array(0, dtype=jnp.int32)
    #         )
    #     )
    #     is_all_in = jnp.where(
    #         player_action
    #     )
    
    def update_table_state_in_response_to_action(table_state: TableState, player_response: PlayerActionResponse):

        # new_history = table_state.history.at[table_state.action_count].set(
        #     ActionRecord(
        #         player_id=table_state.current_player,
        #         action_type=player_response.action.type,
        #         amount=jnp.where(player_response.action.type == Action.CHECK_CALL, get_current_bet(table_state.bets), jnp.where(player_response.action.type == Action.RAISE, player_response.action.value, 0)),
        #         is_all_in=jnp.where(
        #              table_state.bets[table_state.current_player])
        #     )
        # )
    # player_id: jax. Array
    # action_type: jax.Array
    # amount: jax.Array
    # is_all_in: jax.Array
    # is_valid: jax.Array
    return PokerEnv(init=init, player_metadata=player_metadata)













class TestBot:
    def __init__(self):
        pass
    def init_state(key):
        return None
    
    def act(self, action_sequence, state):
        return None
    
    def observe(self, data):
        return None
    
    def metadata(self):
        return {
            "bot_type": "test_bot"
        }




if __name__ == "__main__":
    # print(ID_TO_CARD)
    # print(CARD_TO_ID)
    # get_deck_state = make_poker_env(9).get_deck_state
    # key = jax.random.PRNGKey(0)
    # print(get_deck_state(key))
    test_bot = TestBot()
    print(make_poker_env((test_bot,), max_players=1))