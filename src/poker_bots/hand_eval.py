from functools import cache
from itertools import combinations, combinations_with_replacement
from math import comb

import jax
import jax.numpy as jnp
import numpy as np

NUM_RANKS = 13
NUM_SUITS = 4
CATEGORY_BASE = NUM_RANKS ** 5

class Category:
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    TRIPS = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    QUADS = 7
    STRAIGHT_FLUSH = 8

SHAPE_CATEGORIES = {
    (4, 1): Category.QUADS,
    (3, 2): Category.FULL_HOUSE,
    (3, 1, 1): Category.TRIPS,
    (2, 2, 1): Category.TWO_PAIR,
    (2, 1, 1, 1): Category.PAIR,
    (1, 1, 1, 1, 1): Category.HIGH_CARD
}

def get_straight_high(ordered_ranks: tuple[int, ...]) -> int:
    if ordered_ranks == (12, 3, 2, 1, 0):
        return 3
    if len(ordered_ranks) == 5 and ordered_ranks[0] - ordered_ranks[4] == 4:
        return ordered_ranks[0]
    return -1

@cache
def score_five(ranks: tuple[int, ...], is_flush: bool) -> int:
    groups = sorted(((ranks.count(r), r) for r in set(ranks)), reverse=True)
    shape = tuple(count for count, _ in groups)
    ordered_ranks = tuple(r for _, r in groups)
    straight_high = get_straight_high(ordered_ranks)
    if straight_high >= 0:
        category = Category.STRAIGHT_FLUSH if is_flush else Category.STRAIGHT
        ordered_ranks = (straight_high,)
    elif is_flush:
        category = Category.FLUSH
    else:
        category = SHAPE_CATEGORIES[shape]
    tiebreak = 0
    for r in ordered_ranks:
        tiebreak = tiebreak * NUM_RANKS + r
    return category * CATEGORY_BASE + tiebreak

def score_seven_reference(cards: tuple[int, ...]) -> int:
    return max(
        score_five(
            tuple(sorted(card % NUM_RANKS for card in five)),
            len({card // NUM_RANKS for card in five}) == 1
        )
        for five in combinations(cards, 5)
    )

def get_multiset_index(sorted_ranks: tuple[int, ...]) -> int:
    return sum(comb(r + i, i + 1) for i, r in enumerate(sorted_ranks))

@cache
def build_rank_tables(num_cards: int = 7) -> tuple[np.ndarray, np.ndarray]:
    multiset_table = np.zeros(comb(NUM_RANKS + num_cards - 1, num_cards), dtype=np.int32)
    for multiset in combinations_with_replacement(range(NUM_RANKS), num_cards):
        if max(multiset.count(r) for r in multiset) > 4:
            continue
        multiset_table[get_multiset_index(multiset)] = max(
            score_five(five, False) for five in combinations(multiset, 5)
        )
    flush_table = np.zeros(1 << NUM_RANKS, dtype=np.int32)
    for mask in range(1 << NUM_RANKS):
        suited = tuple(r for r in range(NUM_RANKS) if mask >> r & 1)
        if len(suited) >= 5:
            flush_table[mask] = max(score_five(five, True) for five in combinations(suited, 5))
    return multiset_table, flush_table

COMB_TABLE = np.array([[comb(n, k) for k in range(8)] for n in range(NUM_RANKS + 6)], dtype=np.int32)

def evaluate_hands(cards: jax.Array) -> jax.Array:
    """Scores hands of 5, 6 or 7 cards along the last axis; higher is better and rank // CATEGORY_BASE
    is the Category. Every entry must be a real card: -1 padding gives garbage."""
    num_cards = cards.shape[-1]
    assert 5 <= num_cards <= 7, f"evaluate_hands takes 5 to 7 cards, got {num_cards}"
    multiset_table, flush_table = build_rank_tables(num_cards)
    ranks = cards % NUM_RANKS
    suits = cards // NUM_RANKS

    suit_counts = jnp.sum(suits[..., None] == jnp.arange(NUM_SUITS), axis=-2)
    flush_suit = jnp.argmax(suit_counts, axis=-1)
    is_flush = jnp.max(suit_counts, axis=-1) >= 5
    flush_mask = jnp.sum(
        jnp.where(suits == flush_suit[..., None], jnp.left_shift(1, ranks), 0),
        axis=-1
    )

    sorted_ranks = jnp.sort(ranks, axis=-1)
    positions = jnp.arange(num_cards)
    multiset_index = jnp.sum(jnp.asarray(COMB_TABLE)[sorted_ranks + positions, positions + 1], axis=-1)

    return jnp.where(
        is_flush,
        jnp.asarray(flush_table)[flush_mask],
        jnp.asarray(multiset_table)[multiset_index]
    )
