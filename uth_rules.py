"""
Ultimate Texas Hold'em payout rules and hand resolution.

All monetary values are in units of the Ante bet.

Bet structure per hand:
  - Player posts Ante (1 unit) + Blind (1 unit)
  - Preflop: Play bet = 4x Ante, OR check
  - Flop (if checked preflop): Play bet = 2x Ante, OR check
  - Turn/River (if checked flop): Play bet = 1x Ante, OR fold
  - Fold: lose Ante + Blind (−2 units)

Blind pays only when player wins the hand.
Ante pushes (no gain/loss) when dealer does not qualify.
"""
from hand_evaluator import (
    HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH,
    FULL_HOUSE, QUADS, STRAIGHT_FLUSH, ROYAL_FLUSH,
)

# Blind payout multipliers (applied to 1 unit blind bet)
BLIND_PAYOUT = {
    ROYAL_FLUSH:    500.0,
    STRAIGHT_FLUSH:  50.0,
    QUADS:           10.0,
    FULL_HOUSE:       3.0,
    FLUSH:            1.5,
    STRAIGHT:         1.0,
}

# Dealer must have at least a pair to qualify
_PAIR_THRESHOLD = PAIR


def dealer_qualifies(dealer_score: int, dealer_type: int) -> bool:
    return dealer_type >= _PAIR_THRESHOLD


def blind_bonus(player_type: int) -> float:
    """Return blind payout multiplier for the player's hand type (0 = push)."""
    return BLIND_PAYOUT.get(player_type, 0.0)


def resolve_hand(
    player_score: int,
    player_type: int,
    dealer_score: int,
    dealer_type: int,
    play_multiple: int,
) -> float:
    """
    Return net profit/loss in ante units for a completed UTH hand.

    play_multiple: 0 (fold), 1, 2, or 4
    When play_multiple == 0 the player folded; return -2 immediately.
    """
    if play_multiple == 0:
        return -2.0  # lost ante + blind

    qualifies = dealer_qualifies(dealer_score, dealer_type)

    if player_score > dealer_score:
        # Player wins
        play_win = float(play_multiple)
        ante_win = 1.0 if qualifies else 0.0  # push if dealer doesn't qualify
        blind_win = blind_bonus(player_type)
        return play_win + ante_win + blind_win

    elif player_score < dealer_score:
        # Dealer wins — player loses play + ante + blind regardless of qualify
        return -float(play_multiple) - 1.0 - 1.0

    else:
        # Tie — everything pushes
        return 0.0
