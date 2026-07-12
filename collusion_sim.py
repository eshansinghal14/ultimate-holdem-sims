"""
Collusion simulation for Ultimate Texas Hold'em.

Usage:
    python collusion_sim.py <num_trials> <num_hands> <num_players>

Example:
    python collusion_sim.py 100 500 3

Simulates <num_trials> independent sessions of <num_hands> hands each,
with <num_players> total players sharing hole-card info (collusion).

Strategy: load precomputed edge data from collusion_edge_data.json;
for each hand, look up the edge given the player's cards and visible
colluder cards. Raise 4x if edge > 0, else follow optimal check path.

Outputs:
  - Plot: PnL trajectories across hands for every trial + bold mean
  - Print: mean final PnL, std dev, per-hand EV, variance
"""

import json
import sys
from collections import defaultdict
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from deck import (
    Card, canonical_hand_key, create_deck, deal, hand_label,
    remove_cards, representative_cards, RANK_NAMES,
)
from hand_evaluator import best_of_7, PAIR
from uth_rules import resolve_hand

SEED = 0


# ── Strategy lookup ──────────────────────────────────────────────────────────

def load_edge_data(path: str = "collusion_edge_data.json") -> dict:
    with open(path) as f:
        return json.load(f)


def _scenario_key(colluder_cards: List[Card], r1: int, r2: int) -> str:
    """
    Build the scenario key matching collusion_edge_data.json format.
    Counts how many of rank r1 / r2 appear in colluder cards.
    """
    counts = defaultdict(int)
    for c in colluder_cards:
        counts[c.rank] += 1

    is_pair = (r1 == r2)
    parts = []
    if is_pair:
        n = counts.get(r1, 0)
        if n > 0:
            parts.append(f"{RANK_NAMES[r1]}_seen={n}")
    else:
        for rank in (r1, r2):
            n = counts.get(rank, 0)
            if n > 0:
                parts.append(f"{RANK_NAMES[rank]}_seen={n}")

    return ",".join(parts) if parts else "none_seen"


def lookup_decision(
    edge_data: dict,
    player_cards: List[Card],
    colluder_cards: List[Card],
    num_players: int,
) -> float:
    """
    Return the precomputed mean edge for (player_cards, colluder_cards, num_players).
    JSON structure: {hand_label: {scenario_key: {num_players: {mean, ci_low, ...}}}}.
    Falls back to 'none_seen' scenario, then to -999 (check) if hand not found.
    """
    r1, r2, suited = canonical_hand_key(player_cards[0], player_cards[1])
    lbl = hand_label(r1, r2, suited)

    if lbl not in edge_data:
        return -999.0

    hand_info = edge_data[lbl]
    p_str = str(num_players)

    # Try exact visible-rank scenario
    sk = _scenario_key(colluder_cards, r1, r2) if colluder_cards else "none_seen"
    if sk in hand_info and p_str in hand_info[sk]:
        return hand_info[sk][p_str]["mean"]

    # Fall back to none_seen
    if "none_seen" in hand_info and p_str in hand_info["none_seen"]:
        return hand_info["none_seen"][p_str]["mean"]

    return -999.0


# ── Hand simulation ──────────────────────────────────────────────────────────

def simulate_one_hand(
    rng: np.random.Generator,
    player_cards: List[Card],
    colluder_cards: List[Card],
    edge: float,
) -> float:
    """
    Deal out community + dealer cards and return net PnL for this hand.
    Strategy: raise 4x if edge > 0, else optimal check path.
    """
    dead = player_cards + colluder_cards
    deck = remove_cards(create_deck(), dead)
    community_and_dealer, _ = deal(rng, deck, 7)
    community = community_and_dealer[:5]
    dealer_hole = community_and_dealer[5:7]

    p_score, p_type = best_of_7(player_cards + community)
    d_score, d_type = best_of_7(dealer_hole + community)

    if edge > 0:
        play_multiple = 4
        return resolve_hand(p_score, p_type, d_score, d_type, play_multiple)
    else:
        return _check_path_result(rng, player_cards, colluder_cards,
                                  community, dealer_hole, p_score, p_type,
                                  d_score, d_type)


def _check_path_result(
    rng, player_cards, colluder_cards, community, dealer_hole,
    p_score, p_type, d_score, d_type,
) -> float:
    """Optimal check-path: raise 2x on flop with pair+, else raise 1x on river with pair+, else fold."""
    from hand_evaluator import evaluate_5
    from itertools import combinations

    flop = community[:3]
    # Evaluate player hand with flop only
    flop_cards = player_cards + flop
    best_flop_score, best_flop_type = 0, 0
    for combo in combinations(flop_cards, min(5, len(flop_cards))):
        s, t = evaluate_5(list(combo))
        if s > best_flop_score:
            best_flop_score, best_flop_type = s, t

    if best_flop_type >= PAIR:
        play_multiple = 2
    else:
        if p_type >= PAIR:
            play_multiple = 1
        else:
            play_multiple = 0

    return resolve_hand(p_score, p_type, d_score, d_type, play_multiple)


# ── Trial simulation ──────────────────────────────────────────────────────────

def run_trial(
    rng: np.random.Generator,
    edge_data: dict,
    num_hands: int,
    num_players: int,
) -> np.ndarray:
    """
    Simulate one trial of num_hands hands.
    Returns array of shape (num_hands,) — cumulative PnL at each hand.
    """
    pnl = 0.0
    history = np.zeros(num_hands)
    base_deck = create_deck()
    n_colluders = num_players - 1

    for i in range(num_hands):
        # Deal player cards
        shuffled = list(base_deck)
        rng.shuffle(shuffled)

        player_cards = shuffled[:2]
        if n_colluders > 0:
            colluder_cards = shuffled[2: 2 + n_colluders * 2]
        else:
            colluder_cards = []

        edge = lookup_decision(edge_data, player_cards, colluder_cards, num_players)
        pnl += simulate_one_hand(rng, player_cards, colluder_cards, edge)
        history[i] = pnl

    return history


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 4:
        print("Usage: python collusion_sim.py <num_trials> <num_hands> <num_players>")
        sys.exit(1)

    num_trials  = int(sys.argv[1])
    num_hands   = int(sys.argv[2])
    num_players = int(sys.argv[3])

    if not (1 <= num_players <= 6):
        print("num_players must be 1–6")
        sys.exit(1)

    print(f"Loading edge data...")
    try:
        edge_data = load_edge_data()
    except FileNotFoundError:
        print("collusion_edge_data.json not found. Run preflop_edge.py first.")
        sys.exit(1)

    rng = np.random.default_rng(SEED)
    all_histories = np.zeros((num_trials, num_hands))

    print(f"Simulating {num_trials} trials × {num_hands} hands at {num_players}-player table...")
    for t in range(num_trials):
        all_histories[t] = run_trial(rng, edge_data, num_hands, num_players)
        if (t + 1) % max(1, num_trials // 10) == 0:
            print(f"  Trial {t+1}/{num_trials} done", flush=True)

    final_pnls = all_histories[:, -1]
    per_hand_evs = final_pnls / num_hands

    print("\n" + "=" * 50)
    print(f"  RESULTS  ({num_trials} trials, {num_hands} hands, {num_players} players)")
    print("=" * 50)
    print(f"  Mean final PnL   : {np.mean(final_pnls):+.2f} ante units")
    print(f"  Std dev          : {np.std(final_pnls):.2f} ante units")
    print(f"  Variance         : {np.var(final_pnls):.2f}")
    print(f"  Per-hand EV      : {np.mean(per_hand_evs):+.4f} ante units")
    print(f"  Min final PnL    : {np.min(final_pnls):+.2f}")
    print(f"  Max final PnL    : {np.max(final_pnls):+.2f}")
    print("=" * 50)

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"UTH Collusion Simulation  |  {num_players} players, "
        f"{num_hands} hands/trial, {num_trials} trials",
        fontsize=13,
    )

    # Left: PnL trajectories
    ax = axes[0]
    x = np.arange(1, num_hands + 1)
    alpha = max(0.05, min(0.4, 20 / num_trials))
    for t in range(num_trials):
        ax.plot(x, all_histories[t], color="steelblue", alpha=alpha, linewidth=0.8)
    mean_curve = all_histories.mean(axis=0)
    ax.plot(x, mean_curve, color="crimson", linewidth=2, label="Mean PnL")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Hand number")
    ax.set_ylabel("Cumulative PnL (ante units)")
    ax.set_title("PnL Trajectories")
    ax.legend()

    # Right: Distribution of final PnL
    ax2 = axes[1]
    ax2.hist(final_pnls, bins=max(10, num_trials // 5), color="steelblue",
             edgecolor="white", alpha=0.85)
    ax2.axvline(np.mean(final_pnls), color="crimson", linewidth=2,
                label=f"Mean = {np.mean(final_pnls):+.1f}")
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Final PnL after all hands (ante units)")
    ax2.set_ylabel("Trial count")
    ax2.set_title("Final PnL Distribution")
    ax2.legend()

    plt.tight_layout()
    out_path = f"collusion_sim_P{num_players}_T{num_trials}_H{num_hands}.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
