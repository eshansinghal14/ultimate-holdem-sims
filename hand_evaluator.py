"""
5-card and 7-card hand evaluation for Ultimate Texas Hold'em.
Returns an integer score (higher = better) and a hand-type constant.
"""
from itertools import combinations
from typing import List, Tuple
from deck import Card

HIGH_CARD      = 1
PAIR           = 2
TWO_PAIR       = 3
TRIPS          = 4
STRAIGHT       = 5
FLUSH          = 6
FULL_HOUSE     = 7
QUADS          = 8
STRAIGHT_FLUSH = 9
ROYAL_FLUSH    = 10

HAND_NAMES = {
    HIGH_CARD: "High Card", PAIR: "Pair", TWO_PAIR: "Two Pair",
    TRIPS: "Trips", STRAIGHT: "Straight", FLUSH: "Flush",
    FULL_HOUSE: "Full House", QUADS: "Quads",
    STRAIGHT_FLUSH: "Straight Flush", ROYAL_FLUSH: "Royal Flush",
}


def evaluate_5(cards: List[Card]) -> Tuple[int, int]:
    """Evaluate exactly 5 cards. Returns (score, hand_type)."""
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    is_flush = len(set(suits)) == 1

    # Check straight
    is_straight = False
    straight_high = 0
    if ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5:
        is_straight = True
        straight_high = ranks[0]
    # Wheel: A-2-3-4-5
    if set(ranks) == {14, 2, 3, 4, 5}:
        is_straight = True
        straight_high = 5

    from collections import Counter
    counts = Counter(ranks)
    freq = sorted(counts.values(), reverse=True)
    # sort by (frequency desc, rank desc) for tiebreaker
    grouped = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    primary_ranks = [r for r, _ in grouped]

    if is_straight and is_flush:
        hand_type = ROYAL_FLUSH if straight_high == 14 else STRAIGHT_FLUSH
        score = _score(hand_type, [straight_high])
    elif freq == [4, 1]:
        hand_type = QUADS
        score = _score(hand_type, primary_ranks)
    elif freq == [3, 2]:
        hand_type = FULL_HOUSE
        score = _score(hand_type, primary_ranks)
    elif is_flush:
        hand_type = FLUSH
        score = _score(hand_type, ranks)
    elif is_straight:
        hand_type = STRAIGHT
        score = _score(hand_type, [straight_high])
    elif freq == [3, 1, 1]:
        hand_type = TRIPS
        score = _score(hand_type, primary_ranks)
    elif freq == [2, 2, 1]:
        hand_type = TWO_PAIR
        score = _score(hand_type, primary_ranks)
    elif freq == [2, 1, 1, 1]:
        hand_type = PAIR
        score = _score(hand_type, primary_ranks)
    else:
        hand_type = HIGH_CARD
        score = _score(hand_type, ranks)

    return score, hand_type


def _score(hand_type: int, ranks: List[int]) -> int:
    """
    Pack hand_type + exactly 5 rank slots into one integer so hand_type
    always dominates tiebreakers (e.g. ROYAL_FLUSH > any QUADS).
    Ranks are padded with 0 to fill 5 slots.
    """
    padded = list(ranks[:5]) + [0] * (5 - min(5, len(ranks)))
    s = hand_type
    for r in padded:
        s = s * 15 + r
    return s


def best_of_7(cards: List[Card]) -> Tuple[int, int]:
    """Return (best_score, hand_type) across all C(7,5)=21 five-card combos."""
    best_score, best_type = 0, 0
    for combo in combinations(cards, 5):
        score, hand_type = evaluate_5(list(combo))
        if score > best_score:
            best_score, best_type = score, hand_type
    return best_score, best_type
