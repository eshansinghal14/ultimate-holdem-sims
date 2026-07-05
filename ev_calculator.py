"""
Monte Carlo EV calculator for a UTH preflop decision.

All EVs are in ante units.
known_cards: cards already "dead" (colluders' hole cards) — excluded from deck,
             so dealer outs are naturally computed against the reduced deck.

Flop raise 2x conditions:
  1. Two pair or better
  2. Hidden pair (at least one hole card pairs with flop or pocket pair)
     — EXCEPT pocket 2s
  3. Four to a flush where at least one hole card in the flush is T or better

River raise 1x condition:
  Fewer than 21 cards in the remaining unknown deck (dealer's potential pool)
  can beat the player using best_of_6(that card + 5 community).
  Known colluder cards are already excluded, so their ranks don't count as outs.
"""
from itertools import combinations
from typing import List
import numpy as np

from deck import Card, create_deck, remove_cards, deal
from hand_evaluator import best_of_7, evaluate_5, PAIR, TWO_PAIR
from uth_rules import resolve_hand


def _remaining_deck(player_cards: List[Card], known_cards: List[Card]) -> List[Card]:
    dead = list(player_cards) + list(known_cards)
    return remove_cards(create_deck(), dead)


# ── Flop decision helpers ────────────────────────────────────────────────────

def _is_pocket_twos(player_cards: List[Card]) -> bool:
    return player_cards[0].rank == 2 and player_cards[1].rank == 2


def _has_hole_card_pair(player_cards: List[Card], flop: List[Card]) -> bool:
    """True if any hole card pairs with a flop card, or hole cards pair each other."""
    h1, h2 = player_cards[0].rank, player_cards[1].rank
    flop_ranks = {c.rank for c in flop}
    return h1 == h2 or h1 in flop_ranks or h2 in flop_ranks


def _has_four_flush_hidden_ten_plus(player_cards: List[Card], flop: List[Card]) -> bool:
    """
    True if 4+ of the 5 cards (player + flop) share a suit, and at least one
    of the player's hole cards contributes to that flush with rank >= T(10).
    """
    from collections import Counter
    all_cards = list(player_cards) + list(flop)
    suit_counts = Counter(c.suit for c in all_cards)
    for suit, count in suit_counts.items():
        if count >= 4:
            for hc in player_cards:
                if hc.suit == suit and hc.rank >= 10:
                    return True
    return False


def _flop_raise(player_cards: List[Card], flop: List[Card]) -> bool:
    """Return True if player should raise 2x on the flop."""
    _, flop_type = evaluate_5(player_cards + flop)  # exactly 5 cards

    if flop_type >= TWO_PAIR:
        return True

    if flop_type == PAIR:
        if _has_hole_card_pair(player_cards, flop) and not _is_pocket_twos(player_cards):
            return True

    if _has_four_flush_hidden_ten_plus(player_cards, flop):
        return True

    return False


# ── River decision helpers ───────────────────────────────────────────────────

def _best_of_6(cards: List[Card]):
    """Best 5-card hand from exactly 6 cards (C(6,5)=6 combos)."""
    best_score, best_type = 0, 0
    for combo in combinations(cards, 5):
        s, t = evaluate_5(list(combo))
        if s > best_score:
            best_score, best_type = s, t
    return best_score, best_type


def _count_dealer_outs(player_score: int, community: List[Card], outs_deck: List[Card]) -> int:
    """
    Count cards in outs_deck where that single card + 5 community
    (evaluated as best of 6) beats the player's score.

    outs_deck is the pool of unknown cards dealer could draw from —
    colluder cards are already excluded, so known-dead ranks don't inflate the count.
    """
    outs = 0
    for c in outs_deck:
        d_score, _ = _best_of_6([c] + community)
        if d_score > player_score:
            outs += 1
    return outs


# ── EV functions ─────────────────────────────────────────────────────────────

def ev_raise4x(
    player_cards: List[Card],
    known_cards: List[Card],
    rng: np.random.Generator,
    n_sims: int = 500,
) -> float:
    deck = _remaining_deck(player_cards, known_cards)
    total = 0.0
    for _ in range(n_sims):
        drawn, _ = deal(rng, deck, 7)  # 5 community + 2 dealer
        community = drawn[:5]
        dealer_hole = drawn[5:7]
        p_score, p_type = best_of_7(player_cards + community)
        d_score, d_type = best_of_7(dealer_hole + community)
        total += resolve_hand(p_score, p_type, d_score, d_type, play_multiple=4)
    return total / n_sims


def ev_check_path(
    player_cards: List[Card],
    known_cards: List[Card],
    rng: np.random.Generator,
    n_sims: int = 500,
) -> float:
    """
    Simulate the check-path strategy:
      Flop: raise 2x if 2pair+, hidden pair (not pocket 2s), or 4-flush with hidden T+
      River: raise 1x if dealer has < 21 outs that beat player (using known dead cards)
             else fold (−2)
    """
    deck = _remaining_deck(player_cards, known_cards)
    total = 0.0

    for _ in range(n_sims):
        drawn, _ = deal(rng, deck, 7)
        community = drawn[:5]
        dealer_hole = drawn[5:7]
        flop = community[:3]

        if _flop_raise(player_cards, flop):
            play_multiple = 2
            p_score, p_type = best_of_7(player_cards + community)
            d_score, d_type = best_of_7(dealer_hole + community)
            total += resolve_hand(p_score, p_type, d_score, d_type, play_multiple)
        else:
            # River decision: count outs from unknown pool (excludes colluders + community)
            outs_deck = remove_cards(deck, community)  # dealer cards still unknown here
            p_score, p_type = best_of_7(player_cards + community)
            n_outs = _count_dealer_outs(p_score, community, outs_deck)

            if n_outs < 21:
                play_multiple = 1
                d_score, d_type = best_of_7(dealer_hole + community)
                total += resolve_hand(p_score, p_type, d_score, d_type, play_multiple)
            else:
                total += -2.0  # fold

    return total / n_sims


def calc_edge(
    player_cards: List[Card],
    known_cards: List[Card],
    rng: np.random.Generator,
    n_sims: int = 500,
) -> float:
    """EV(raise 4x) − EV(check path). Positive = raise is better."""
    r = ev_raise4x(player_cards, known_cards, rng, n_sims)
    c = ev_check_path(player_cards, known_cards, rng, n_sims)
    return r - c
