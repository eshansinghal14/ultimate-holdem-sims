"""
Flush Draw Flop Edge Calculator for Ultimate Texas Hold'em.

Simulates the flop raise decision (2x vs check) when the player holds four
to a flush: (high_card)c + 2c in hand, with exactly 2 clubs on the flop.

For each count of flush outs visible in colluder hands (0..max-outs), reports:
  EV(raise 2x on flop) - EV(check flop -> optimal river play)

Args:
  --high-card   : rank of the highest flush-suit card in hand (A K Q J T 9 8 7 6 5 4 3)
  --num-players : total players at the table [1-6]
  --max-outs    : max flush outs visible in colluder hands to test [default: 4]
  --n-config    : flop/colluder samples per outs-seen cell [default: 200]
  --n-sims      : MC runouts per EV estimate [default: 200]
  --seed        : random seed [default: 42]
"""

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deck import Card, create_deck, deal, remove_cards, RANK_NAMES
from hand_evaluator import best_of_7, evaluate_5
from uth_rules import resolve_hand

Z_99 = 2.576
FLUSH_SUIT = 0  # clubs — arbitrary

_NAME_TO_RANK: Dict[str, int] = {v: k for k, v in RANK_NAMES.items()}


def _parse_rank(s: str) -> int:
    r = _NAME_TO_RANK.get(s.upper())
    if r is None:
        raise ValueError(f"Unknown rank '{s}'. Use A K Q J T 9 8 7 6 5 4 3 2")
    return r


# ── EV helpers ────────────────────────────────────────────────────────────────

def _best_of_6(cards: List[Card]) -> Tuple[int, int]:
    best_score, best_type = 0, 0
    for combo in combinations(cards, 5):
        s, t = evaluate_5(list(combo))
        if s > best_score:
            best_score, best_type = s, t
    return best_score, best_type


def _count_dealer_outs(player_score: int, community: List[Card], outs_deck: List[Card]) -> int:
    outs = 0
    for c in outs_deck:
        d_score, _ = _best_of_6([c] + community)
        if d_score > player_score:
            outs += 1
    return outs


def _after_dead(player_cards: List[Card], col_cards: List[Card], flop: List[Card]) -> List[Card]:
    dead = list(player_cards) + list(col_cards) + list(flop)
    return remove_cards(create_deck(), dead)


def ev_raise2x(
    player_cards: List[Card], col_cards: List[Card], flop: List[Card],
    rng: np.random.Generator, n_sims: int,
) -> float:
    """EV of raising 2x on the flop."""
    deck = _after_dead(player_cards, col_cards, flop)
    total = 0.0
    for _ in range(n_sims):
        drawn, _ = deal(rng, deck, 4)         # turn, river, dealer x2
        community = flop + drawn[:2]
        dealer_hole = drawn[2:4]
        p_score, p_type = best_of_7(player_cards + community)
        d_score, d_type = best_of_7(dealer_hole + community)
        total += resolve_hand(p_score, p_type, d_score, d_type, play_multiple=2)
    return total / n_sims


def ev_check(
    player_cards: List[Card], col_cards: List[Card], flop: List[Card],
    rng: np.random.Generator, n_sims: int,
) -> float:
    """EV of checking the flop, then raising 1x or folding at the river."""
    deck = _after_dead(player_cards, col_cards, flop)
    total = 0.0
    for _ in range(n_sims):
        drawn, _ = deal(rng, deck, 4)
        turn_river = drawn[:2]
        community = flop + turn_river
        dealer_hole = drawn[2:4]
        p_score, p_type = best_of_7(player_cards + community)
        # outs_deck: full pool of unknowns at river decision time
        outs_deck = remove_cards(deck, turn_river)
        n_outs = _count_dealer_outs(p_score, community, outs_deck)
        if n_outs < 21:
            d_score, d_type = best_of_7(dealer_hole + community)
            total += resolve_hand(p_score, p_type, d_score, d_type, play_multiple=1)
        else:
            total += -2.0  # fold
    return total / n_sims


# ── Constrained dealing ───────────────────────────────────────────────────────

def _deal_colluders_constrained(
    rng: np.random.Generator, deck: List[Card],
    n_colluders: int, n_flush_seen: int, flush_suit: int,
    max_attempts: int = 500,
) -> Optional[List[Card]]:
    n_cards = n_colluders * 2
    for _ in range(max_attempts):
        drawn, _ = deal(rng, deck, n_cards)
        if sum(1 for c in drawn if c.suit == flush_suit) == n_flush_seen:
            return drawn
    return None


def _deal_flop_constrained(
    rng: np.random.Generator, deck: List[Card], flush_suit: int,
    max_attempts: int = 500,
) -> Optional[List[Card]]:
    """Deal a 3-card flop with exactly 2 cards of flush_suit (making 4-flush with player's 2)."""
    for _ in range(max_attempts):
        drawn, _ = deal(rng, deck, 3)
        if sum(1 for c in drawn if c.suit == flush_suit) == 2:
            return drawn
    return None


# ── CI ────────────────────────────────────────────────────────────────────────

def _ci99(values: List[float]) -> Tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if n == 1:
        return mean, mean, mean
    se = float(np.std(values, ddof=1)) / np.sqrt(n)
    return mean, mean - Z_99 * se, mean + Z_99 * se


# ── Worker ────────────────────────────────────────────────────────────────────

def _compute_task(args_tuple: tuple) -> tuple:
    n_outs_seen, num_players, high_card_rank, n_config, n_sims, seed = args_tuple
    rng = np.random.default_rng(seed)

    # Player holds high_card + 2 of the flush suit (both clubs)
    player_cards = [Card(high_card_rank, FLUSH_SUIT), Card(2, FLUSH_SUIT)]
    base_deck = remove_cards(create_deck(), player_cards)
    n_colluders = num_players - 1

    edges = []
    for _ in range(n_config):
        if n_colluders > 0:
            col_cards = _deal_colluders_constrained(
                rng, base_deck, n_colluders, n_outs_seen, FLUSH_SUIT
            )
            if col_cards is None:
                continue
        else:
            col_cards = []

        after_col = remove_cards(base_deck, col_cards)
        flop = _deal_flop_constrained(rng, after_col, FLUSH_SUIT)
        if flop is None:
            continue

        ev2  = ev_raise2x(player_cards, col_cards, flop, rng, n_sims)
        ev_c = ev_check(player_cards, col_cards, flop, rng, n_sims)
        edges.append(ev2 - ev_c)

    if not edges:
        return n_outs_seen, None

    mean, ci_low, ci_high = _ci99(edges)
    return n_outs_seen, {
        "mean":      round(mean, 4),
        "ci_low":    round(ci_low, 4),
        "ci_high":   round(ci_high, 4),
        "n_samples": len(edges),
        "decision":  "raise" if mean > 0 else "check",
    }


# ── Main computation ──────────────────────────────────────────────────────────

def compute_edges(
    high_card_rank: int,
    num_players: int,
    max_outs: int,
    n_config: int,
    n_sims: int,
    seed: int,
    verbose: bool = True,
) -> Dict[int, dict]:
    # colluders can hold at most (num_players-1)*2 flush cards total
    max_colluder_cards = (num_players - 1) * 2
    effective_max = min(max_outs, max_colluder_cards) if num_players > 1 else 0
    outs_range = list(range(0, effective_max + 1))

    work_items = [
        (n_outs, num_players, high_card_rank, n_config, n_sims, seed * 10_000 + n_outs)
        for n_outs in outs_range
    ]

    if verbose:
        hc = RANK_NAMES[high_card_rank]
        print(f"  Tasks: {len(work_items)}  (high_card={hc}, players={num_players}, "
              f"max_outs={effective_max}, cores={os.cpu_count()})")

    results: Dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = {pool.submit(_compute_task, item): item[0] for item in work_items}
        for fut in as_completed(futures):
            n_outs, entry = fut.result()
            if entry is None:
                continue
            results[n_outs] = entry
            if verbose:
                e = entry
                dec = "RAISE 2x" if e["decision"] == "raise" else "check  "
                print(f"  outs_seen={n_outs}: {e['mean']:+.3f} "
                      f"[{e['ci_low']:+.3f}, {e['ci_high']:+.3f}] -> {dec}  n={e['n_samples']}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UTH Flush Draw Flop Edge — EV(raise 2x) - EV(check) on the flop "
                    "when holding four to a flush."
    )
    p.add_argument("--high-card", required=True,
                   help="Rank of highest flush-suit card in hand (A K Q J T 9 8 7 6 5 4 3)")
    p.add_argument("--num-players", type=int, default=6,
                   help="Total players at table [default: 6]")
    p.add_argument("--max-outs", type=int, default=4,
                   help="Max flush outs visible in colluder hands to test [default: 4]")
    p.add_argument("--n-config", type=int, default=200,
                   help="Colluder/flop samples per outs-seen cell [default: 200]")
    p.add_argument("--n-sims", type=int, default=200,
                   help="MC runouts per EV estimate [default: 200]")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed [default: 42]")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        high_card_rank = _parse_rank(args.high_card)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if high_card_rank < 3:
        print("--high-card must be 3 or higher (rank 2 is used as the second hole card)")
        sys.exit(1)
    if not (1 <= args.num_players <= 6):
        print("--num-players must be between 1 and 6")
        sys.exit(1)

    hc_name = RANK_NAMES[high_card_rank]
    print(f"UTH flush draw edge: {hc_name}-high, {args.num_players} players, "
          f"max_outs={args.max_outs}")
    print(f"  n_config={args.n_config}, n_sims={args.n_sims}, seed={args.seed}")

    results = compute_edges(
        high_card_rank=high_card_rank,
        num_players=args.num_players,
        max_outs=args.max_outs,
        n_config=args.n_config,
        n_sims=args.n_sims,
        seed=args.seed,
        verbose=True,
    )

    if not results:
        print("No results computed.")
        sys.exit(1)

    print("\nSummary:")
    for n_outs in sorted(results):
        e = results[n_outs]
        dec = "RAISE 2x" if e["decision"] == "raise" else "check  "
        print(f"  outs_seen={n_outs}: edge={e['mean']:+.3f} "
              f"[{e['ci_low']:+.3f}, {e['ci_high']:+.3f}] -> {dec}")


if __name__ == "__main__":
    main()
