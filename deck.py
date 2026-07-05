from collections import namedtuple
from typing import List, Tuple
import numpy as np

Card = namedtuple("Card", ["rank", "suit"])
# rank: 2-14 (14=Ace), suit: 0-3

RANK_NAMES = {2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"T",11:"J",12:"Q",13:"K",14:"A"}
SUIT_NAMES = {0:"c",1:"d",2:"h",3:"s"}


def card_str(c: Card) -> str:
    return RANK_NAMES[c.rank] + SUIT_NAMES[c.suit]


def create_deck() -> List[Card]:
    return [Card(rank, suit) for rank in range(2, 15) for suit in range(4)]


def remove_cards(deck: List[Card], cards) -> List[Card]:
    dead = set(cards)
    return [c for c in deck if c not in dead]


def deal(rng: np.random.Generator, deck: List[Card], n: int) -> Tuple[List[Card], List[Card]]:
    idx = rng.choice(len(deck), size=n, replace=False)
    idx_set = set(idx)
    drawn = [deck[i] for i in idx]
    remaining = [deck[i] for i in range(len(deck)) if i not in idx_set]
    return drawn, remaining


def canonical_hand_key(c1: Card, c2: Card) -> Tuple[int, int, bool]:
    """Return (high_rank, low_rank, suited) canonical key for a 2-card hand."""
    r1, r2 = c1.rank, c2.rank
    if r1 < r2:
        r1, r2 = r2, r1
    suited = c1.suit == c2.suit
    return (r1, r2, suited)


def hand_label(r1: int, r2: int, suited: bool) -> str:
    s = "s" if suited else "o"
    if r1 == r2:
        return RANK_NAMES[r1] * 2
    return RANK_NAMES[r1] + RANK_NAMES[r2] + s


def all_canonical_hands() -> List[Tuple[int, int, bool]]:
    """All 169 canonical 2-card starting hands (high_rank, low_rank, suited)."""
    hands = []
    for r1 in range(14, 1, -1):
        for r2 in range(r1, 1, -1):
            if r1 == r2:
                hands.append((r1, r2, False))
            else:
                hands.append((r1, r2, True))
                hands.append((r1, r2, False))
    return hands


def representative_cards(r1: int, r2: int, suited: bool) -> Tuple[Card, Card]:
    """Pick one concrete Card pair representing the canonical hand."""
    if suited:
        return Card(r1, 0), Card(r2, 0)
    else:
        return Card(r1, 0), Card(r2, 1)
