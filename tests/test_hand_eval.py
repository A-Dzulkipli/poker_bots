import jax
import jax.numpy as jnp
import numpy as np

from poker_bots.hand_eval import Category, CATEGORY_BASE, evaluate_hands, score_seven_reference
from poker_bots.poker_game import CARD_TO_ID


def to_cards(hand: str) -> list[int]:
    suits = {"d": "Diamonds", "c": "Clubs", "h": "Hearts", "s": "Spades"}
    return [CARD_TO_ID[(card[:-1], suits[card[-1]])] for card in hand.split()]


def evaluate(*hands: str) -> list[int]:
    return [int(rank) for rank in evaluate_hands(jnp.array([to_cards(hand) for hand in hands]))]


def test_category_order():
    ranks = evaluate(
        "2d 7c 9h Js Kd 3c 4h",    # high card
        "2d 2c 9h Js Kd 3c 4h",    # pair
        "2d 2c 9h 9s Kd 3c 4h",    # two pair
        "2d 2c 2h Js Kd 3c 4h",    # trips
        "2d 3c 4h 5s 6d Jc Kh",    # straight
        "2d 7d 9d Jd Kd 3c 4h",    # flush
        "2d 2c 2h Js Jd 3c 4h",    # full house
        "2d 2c 2h 2s Kd 3c 4h",    # quads
        "2d 3d 4d 5d 6d Jc Kh",    # straight flush
    )
    assert ranks == sorted(ranks)
    assert [rank // CATEGORY_BASE for rank in ranks] == list(range(Category.STRAIGHT_FLUSH + 1))


def test_wheel_is_lowest_straight():
    wheel, six_high = evaluate("Ad 2c 3h 4s 5d 9c Jh", "2d 3c 4h 5s 6d 9c Jh")
    assert wheel < six_high
    assert wheel // CATEGORY_BASE == Category.STRAIGHT


def test_flush_beats_straight_in_same_seven():
    (rank,) = evaluate("5h 6h 7h 8c 9h Jh 2s")
    assert rank // CATEGORY_BASE == Category.FLUSH


def test_kickers_and_ties():
    better_kicker, worse_kicker, board_plays_a, board_plays_b = evaluate(
        "Ad Kc 9h 9s 4d 3c 2h",
        "Ad Qc 9h 9s 4d 3c 2h",
        "2d 3c Ah Ks Qd Jc 9h",
        "4d 5c Ah Ks Qd Jc 9h",
    )
    assert better_kicker > worse_kicker
    assert board_plays_a == board_plays_b


def test_matches_reference_on_random_hands():
    keys = jax.random.split(jax.random.PRNGKey(0), 2000)
    hands = jax.vmap(lambda key: jax.random.permutation(key, 52)[:7])(keys)
    ranks = np.asarray(jax.jit(evaluate_hands)(hands))
    reference = [score_seven_reference(tuple(int(card) for card in hand)) for hand in np.asarray(hands)]
    np.testing.assert_array_equal(ranks, reference)
