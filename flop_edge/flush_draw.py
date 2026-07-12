"""
Flush Draw Flop Edge Calculator for Ultimate Texas Hold'em.

Tests two flush draw configurations on the flop:
  2+2: player holds (high_card)c + 2c, flop has exactly 2 clubs
  1+3: player holds (high_card)c + 2d, flop is monotone (3 clubs)

For each high card in --high-cards and each count of flush outs visible in
colluder hands (0..max-outs), computes:
  EV(raise 2x on flop) - EV(check flop -> optimal river play)

Args:
  --high-cards  : ranks to test, e.g. A K Q J T 9  (required)
  --num-players : total players at the table [1-6, default: 6]
  --max-outs    : max flush outs visible in colluder hands to test [default: 4]
  --n-config    : flop/colluder samples per cell [default: 200]
  --n-sims      : MC runouts per EV estimate [default: 200]
  --seed        : random seed [default: 42]
  --both        : test both 2+2 and 1+3 configs (default: 1+3 only)
  --json-out    : output JSON path [default: flush_draw_P<N>.json in script dir]
"""

import argparse
import json
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
FLUSH_SUIT = 0              # clubs — arbitrary
DRAW_TYPES = ["2+2", "1+3"]

_NAME_TO_RANK: Dict[str, int] = {v: k for k, v in RANK_NAMES.items()}


def _parse_rank(s: str) -> int:
    r = _NAME_TO_RANK.get(s.upper())
    if r is None:
        raise ValueError(f"Unknown rank '{s}'. Use A K Q J T 9 8 7 6 5 4 3 2")
    return r


def _player_cards(high_card_rank: int, draw_type: str) -> List[Card]:
    """
    2+2: high_card♣ + 2♣  (both suited, need 2-flush flop)
    1+3: high_card♣ + 2♦  (one suited card, need monotone flop)
    """
    if draw_type == "2+2":
        return [Card(high_card_rank, FLUSH_SUIT), Card(2, FLUSH_SUIT)]
    else:
        return [Card(high_card_rank, FLUSH_SUIT), Card(2, 1)]  # high♣ + 2♦


def _n_flop_flush(draw_type: str) -> int:
    return 2 if draw_type == "2+2" else 3


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
    deck = _after_dead(player_cards, col_cards, flop)
    total = 0.0
    for _ in range(n_sims):
        drawn, _ = deal(rng, deck, 4)        # turn, river, dealer x2
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
    """Check flop -> raise 1x or fold at river."""
    deck = _after_dead(player_cards, col_cards, flop)
    total = 0.0
    for _ in range(n_sims):
        drawn, _ = deal(rng, deck, 4)
        turn_river = drawn[:2]
        community = flop + turn_river
        dealer_hole = drawn[2:4]
        p_score, p_type = best_of_7(player_cards + community)
        outs_deck = remove_cards(deck, turn_river)
        n_outs = _count_dealer_outs(p_score, community, outs_deck)
        if n_outs < 21:
            d_score, d_type = best_of_7(dealer_hole + community)
            total += resolve_hand(p_score, p_type, d_score, d_type, play_multiple=1)
        else:
            total += -2.0
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
    rng: np.random.Generator, deck: List[Card], flush_suit: int, n_flush: int,
    max_attempts: int = 500,
) -> Optional[List[Card]]:
    """Deal a 3-card flop with exactly n_flush cards of flush_suit."""
    for _ in range(max_attempts):
        drawn, _ = deal(rng, deck, 3)
        if sum(1 for c in drawn if c.suit == flush_suit) == n_flush:
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
    high_card_rank, draw_type, n_outs_seen, num_players, n_config, n_sims, seed = args_tuple
    rng = np.random.default_rng(seed)

    p_cards = _player_cards(high_card_rank, draw_type)
    n_flop = _n_flop_flush(draw_type)
    base_deck = remove_cards(create_deck(), p_cards)
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
        flop = _deal_flop_constrained(rng, after_col, FLUSH_SUIT, n_flop)
        if flop is None:
            continue

        ev2  = ev_raise2x(p_cards, col_cards, flop, rng, n_sims)
        ev_c = ev_check(p_cards, col_cards, flop, rng, n_sims)
        edges.append(ev2 - ev_c)

    if not edges:
        return (high_card_rank, draw_type, n_outs_seen), None

    mean, ci_low, ci_high = _ci99(edges)
    std = float(np.std(edges, ddof=1)) if len(edges) > 1 else 0.0
    return (high_card_rank, draw_type, n_outs_seen), {
        "mean":      round(mean, 4),
        "ci_low":    round(ci_low, 4),
        "ci_high":   round(ci_high, 4),
        "std":       round(std, 4),
        "n_samples": len(edges),
        "decision":  "raise" if mean > 0 else "check",
    }


# ── Main computation ──────────────────────────────────────────────────────────

def compute_edges(
    high_card_ranks: List[int],
    num_players: int,
    max_outs: int,
    n_config: int,
    n_sims: int,
    seed: int,
    draw_types: List[str] = DRAW_TYPES,
    verbose: bool = True,
) -> Tuple[Dict, List[int]]:
    """
    Returns:
      (results, outs_range)
      results: {high_card_rank: {draw_type: {n_outs_seen: entry}}}
    """
    max_colluder_cards = (num_players - 1) * 2
    effective_max = min(max_outs, max_colluder_cards) if num_players > 1 else 0
    outs_range = list(range(0, effective_max + 1))

    work_items = []
    for hc in high_card_ranks:
        for dt in draw_types:
            for n_outs in outs_range:
                i = len(work_items)
                work_items.append(
                    (hc, dt, n_outs, num_players, n_config, n_sims, seed * 100_000 + i)
                )

    if verbose:
        hc_names = [RANK_NAMES[r] for r in high_card_ranks]
        print(f"  Tasks: {len(work_items)}  "
              f"(high_cards={hc_names}, players={num_players}, "
              f"max_outs={effective_max}, cores={os.cpu_count()})")

    results: Dict = {}
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = {pool.submit(_compute_task, item): item for item in work_items}
        for fut in as_completed(futures):
            (hc, dt, n_outs), entry = fut.result()
            if entry is None:
                continue
            results.setdefault(hc, {}).setdefault(dt, {})[n_outs] = entry
            if verbose:
                e = entry
                dec = "RAISE 2x" if e["decision"] == "raise" else "check  "
                print(f"  {RANK_NAMES[hc]:2s} {dt} outs_seen={n_outs}: "
                      f"{e['mean']:+.3f} [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}] "
                      f"-> {dec}  std={e['std']:.3f}  n={e['n_samples']}")

    return results, outs_range


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UTH Flush Draw Flop Edge — EV(raise 2x) - EV(check) on the flop "
                    "when holding four to a flush. Tests both 2+2 and 1+3 draw configs."
    )
    p.add_argument("--high-cards", nargs="+", required=True,
                   help="Ranks of highest flush-suit card to test (e.g. A K Q J T 9)")
    p.add_argument("--num-players", type=int, default=6,
                   help="Total players at table [default: 6]")
    p.add_argument("--max-outs", type=int, default=4,
                   help="Max flush outs visible in colluder hands to test [default: 4]")
    p.add_argument("--n-config", type=int, default=200,
                   help="Colluder/flop samples per cell [default: 200]")
    p.add_argument("--n-sims", type=int, default=200,
                   help="MC runouts per EV estimate [default: 200]")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed [default: 42]")
    p.add_argument("--both", action="store_true",
                   help="Test both 2+2 and 1+3 draw configs (default: 1+3 only)")
    p.add_argument("--json-out",
                   help="Output JSON path [default: flush_draw_P<N>.json in script dir]")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        high_card_ranks = [_parse_rank(s) for s in args.high_cards]
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    invalid = [r for r in high_card_ranks if r < 3]
    if invalid:
        print(f"All --high-cards must be 3 or higher (rank 2 is used as the second hole card). "
              f"Invalid: {[RANK_NAMES[r] for r in invalid]}")
        sys.exit(1)
    if not (1 <= args.num_players <= 6):
        print("--num-players must be between 1 and 6")
        sys.exit(1)

    # Deduplicate and sort descending (A first)
    high_card_ranks = sorted(set(high_card_ranks), reverse=True)

    script_dir = str(Path(__file__).parent)
    json_out = args.json_out or os.path.join(script_dir, f"flush_draw_P{args.num_players}.json")

    draw_types = ["2+2", "1+3"] if args.both else ["1+3"]

    hc_names = [RANK_NAMES[r] for r in high_card_ranks]
    print(f"UTH flush draw edge: high_cards={hc_names}, {args.num_players} players, "
          f"draw_types={draw_types}, max_outs={args.max_outs}")
    print(f"  n_config={args.n_config}, n_sims={args.n_sims}, seed={args.seed}")

    results, outs_range = compute_edges(
        high_card_ranks=high_card_ranks,
        num_players=args.num_players,
        max_outs=args.max_outs,
        n_config=args.n_config,
        n_sims=args.n_sims,
        seed=args.seed,
        draw_types=draw_types,
        verbose=True,
    )

    if not results:
        print("No results computed.")
        sys.exit(1)

    json_data = {
        "metadata": {
            "num_players": args.num_players,
            "max_outs": args.max_outs,
            "n_config": args.n_config,
            "n_sims": args.n_sims,
            "seed": args.seed,
            "draw_types": draw_types,
            "high_cards": hc_names,
        },
        "results": {
            RANK_NAMES[hc]: {
                dt: {str(n): results[hc][dt][n] for n in outs_range if n in results.get(hc, {}).get(dt, {})}
                for dt in draw_types
            }
            for hc in high_card_ranks
        },
    }
    with open(json_out, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON written to {json_out}")

    print("\nSummary:")
    for hc in high_card_ranks:
        for dt in draw_types:
            for n_outs in outs_range:
                e = results.get(hc, {}).get(dt, {}).get(n_outs)
                if e is None:
                    continue
                dec = "RAISE 2x" if e["decision"] == "raise" else "check  "
                print(f"  {RANK_NAMES[hc]:2s} {dt} outs_seen={n_outs}: "
                      f"edge={e['mean']:+.3f} [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}] "
                      f"-> {dec}  std={e['std']:.3f}")


if __name__ == "__main__":
    main()
