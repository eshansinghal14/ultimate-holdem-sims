"""
Collusion Edge Calculator for Ultimate Texas Hold'em.

For each TARGET_HAND × visible-rank scenario × num_players, computes
the 99% confidence interval of EV(raise 4x) − EV(check path).

Outputs:
  - collusion_edge_chart.png  (only combos where decision flips across player counts)
  - collusion_edge_data.json
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deck import (
    Card, create_deck, deal, hand_label, remove_cards, representative_cards,
    RANK_NAMES,
)
from ev_calculator import calc_edge

# ── Tunable parameters ──────────────────────────────────────────────────────
N_CONFIG = 50      # colluder-hand samples per (hand, scenario, num_players)
N_SIMS   = 100      # MC trials per edge estimate inside calc_edge
SEED     = 42
Z_99     = 2.576   # z-score for 99% CI
# ────────────────────────────────────────────────────────────────────────────

MAX_PLAYERS = 6

# Hands to analyze — borderline raise/check decisions in standard UTH strategy.
TARGET_HANDS: List[Tuple[int, int, bool]] = [
    # A2–A4
    (14, 2, True), (14, 2, False),
    (14, 3, True), (14, 3, False),
    (14, 4, True), (14, 4, False),
    # K2–K6
    (13, 2, True), (13, 2, False),
    (13, 3, True), (13, 3, False),
    (13, 4, True), (13, 4, False),
    (13, 5, True), (13, 5, False),
    (13, 6, True), (13, 6, False),
    # Q5–Q8
    (12, 5, True), (12, 5, False),
    (12, 6, True), (12, 6, False),
    (12, 7, True), (12, 7, False),
    (12, 8, True), (12, 8, False),
    # J7–JT
    (11, 7, True), (11, 7, False),
    (11, 8, True), (11, 8, False),
    (11, 9, True), (11, 9, False),
    (11, 10, True), (11, 10, False),
    # Pairs 22–44
    (2, 2, False),
    (3, 3, False),
    (4, 4, False),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_scenarios(r1: int, r2: int, base_deck: List[Card]) -> Dict[str, Dict[int, int]]:
    """
    Build visible-rank scenarios for a hand.
    Returns {scenario_key: {rank: min_count_required_in_colluder_cards}}.

    Non-pairs (r1=high, r2=low):
        none_seen | 1 low seen | 1 high seen | 1 high + 1 low seen

    Pocket pairs:
        none_seen | 1 seen | both remaining seen (up to 2)
    """
    avail_r1 = sum(1 for c in base_deck if c.rank == r1)
    avail_r2 = sum(1 for c in base_deck if c.rank == r2) if r2 != r1 else avail_r1
    is_pair = (r1 == r2)

    scenarios: Dict[str, Dict[int, int]] = {"none_seen": {}}

    if is_pair:
        if avail_r1 >= 1:
            scenarios[f"{RANK_NAMES[r1]}_seen=1"] = {r1: 1}
        if avail_r1 >= 2:
            scenarios[f"{RANK_NAMES[r1]}_seen=2"] = {r1: 2}
    else:
        # r1 = high rank, r2 = low rank
        if avail_r2 >= 1:
            scenarios[f"{RANK_NAMES[r2]}_seen=1"] = {r2: 1}
        if avail_r1 >= 1:
            scenarios[f"{RANK_NAMES[r1]}_seen=1"] = {r1: 1}
        if avail_r1 >= 1 and avail_r2 >= 1:
            key = f"{RANK_NAMES[r1]}_seen=1,{RANK_NAMES[r2]}_seen=1"
            scenarios[key] = {r1: 1, r2: 1}

    return scenarios


def _sample_constrained(
    rng: np.random.Generator,
    base_deck: List[Card],
    n_colluders: int,
    required: Dict[int, int],
    max_attempts: int = 300,
) -> Optional[List[Card]]:
    n_cards = n_colluders * 2
    for _ in range(max_attempts):
        cards, _ = deal(rng, base_deck, n_cards)
        counts: Dict[int, int] = defaultdict(int)
        for c in cards:
            counts[c.rank] += 1
        if all(counts[r] >= cnt for r, cnt in required.items()):
            return cards
    return None


def _ci99(edges: List[float]) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) with 99% normal CI."""
    n = len(edges)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(edges))
    if n == 1:
        return mean, mean, mean
    se = float(np.std(edges, ddof=1)) / np.sqrt(n)
    return mean, mean - Z_99 * se, mean + Z_99 * se


# ── Core computation ──────────────────────────────────────────────────────────

def compute_all_scenario_edges(rng: np.random.Generator, verbose: bool = True) -> Dict:
    """
    Returns:
    {
      hand_label: {
        scenario_key: {
          num_players: {'mean': float, 'ci_low': float, 'ci_high': float, 'decision': str}
        }
      }
    }
    """
    n_hands = len(TARGET_HANDS)
    result: Dict = {}

    for hand_idx, (r1, r2, suited) in enumerate(TARGET_HANDS):
        lbl = hand_label(r1, r2, suited)
        p1, p2 = representative_cards(r1, r2, suited)
        player_cards = [p1, p2]
        base_deck = remove_cards(create_deck(), player_cards)

        scenarios = _build_scenarios(r1, r2, base_deck)
        hand_data: Dict = {}

        for sk, required in scenarios.items():
            total_req = sum(required.values())
            scenario_data: Dict = {}

            for num_players in range(1, MAX_PLAYERS + 1):
                n_col = num_players - 1

                if n_col == 0:
                    if sk != "none_seen":
                        continue  # can't see cards with no colluder
                    edges = [calc_edge(player_cards, [], rng, N_SIMS) for _ in range(N_CONFIG)]
                else:
                    if total_req > n_col * 2:
                        continue  # scenario impossible with this many colluders
                    edges = []
                    for _ in range(N_CONFIG):
                        cc = _sample_constrained(rng, base_deck, n_col, required)
                        if cc is not None:
                            edges.append(calc_edge(player_cards, cc, rng, N_SIMS))

                if edges:
                    mean, ci_low, ci_high = _ci99(edges)
                    scenario_data[num_players] = {
                        "mean":     round(mean,    4),
                        "ci_low":   round(ci_low,  4),
                        "ci_high":  round(ci_high, 4),
                        "decision": "raise" if mean > 0 else "check",
                    }

            if scenario_data:
                hand_data[sk] = scenario_data
                if verbose:
                    p_strs = "  ".join(
                        f"P{p}: {v['mean']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}] {'RAISE' if v['decision']=='raise' else 'check'}"
                        for p, v in sorted(scenario_data.items())
                    )
                    print(f"  [{hand_idx+1}/{n_hands}] {lbl:5s} | {sk:30s} | {p_strs}")

        result[lbl] = hand_data

    return result


# ── Filter ────────────────────────────────────────────────────────────────────

def filter_decision_flips(all_data: Dict) -> List[Tuple[str, str]]:
    """
    Return (hand_label, scenario_key) pairs where the mean edge changes sign
    across player counts — i.e., raise is better for some N, check for others.
    """
    qualifying = []
    for lbl, scenarios in all_data.items():
        for sk, players_data in scenarios.items():
            means = [v["mean"] for v in players_data.values() if not np.isnan(v["mean"])]
            if len(means) >= 2 and max(means) > 0 and min(means) < 0:
                qualifying.append((lbl, sk))
    return qualifying


# ── Table PNG ────────────────────────────────────────────────────────────────

# Cell background colors
_C_RAISE_SURE  = "#C8E6C9"   # green  — mean > 0 and CI entirely above 0
_C_RAISE_LEAN  = "#E8F5E9"   # pale green — mean > 0 but CI crosses 0
_C_CHECK_SURE  = "#FFCDD2"   # red   — mean < 0 and CI entirely below 0
_C_CHECK_LEAN  = "#FFEBEE"   # pale red  — mean < 0 but CI crosses 0
_C_NA          = "#F5F5F5"   # grey  — not applicable (too few colluders)
_C_HEADER      = "#37474F"   # dark slate for header row
_C_HAND_LABEL  = "#ECEFF1"   # light blue-grey for hand-label cells
_C_FLIP_ROW    = "#FFF9C4"   # yellow highlight for decision-flipping rows


def _cell_color(v: dict) -> str:
    if v["decision"] == "raise":
        return _C_RAISE_SURE if v["ci_low"] > 0 else _C_RAISE_LEAN
    else:
        return _C_CHECK_SURE if v["ci_high"] < 0 else _C_CHECK_LEAN


def save_table_png(
    all_data: Dict,
    qualifying_set: set,
    output_path: str = "collusion_edge_chart.png",
) -> None:
    """
    Render all hands × scenarios as a styled PNG table.

    Columns: Hand | Scenario | P1 | P2 | P3 | P4 | P5 | P6
    Cell format: mean ± half_CI  (two decimal places)
    Cell color:  green=raise, red=check; intensity shows CI confidence
    Yellow row background: decision flips across player counts
    """
    p_cols = list(range(1, MAX_PLAYERS + 1))
    col_headers = ["Hand", "Scenario"] + [f"P{p}" for p in p_cols]
    n_fixed = 2  # Hand + Scenario columns

    # Collect all rows in display order
    rows = []
    for lbl, scenarios in all_data.items():
        for sk, pd_ in scenarios.items():
            rows.append((lbl, sk, pd_))

    n_rows = len(rows)

    ROW_H   = 0.28   # inches per data row
    COL_W   = [0.55, 2.0] + [1.25] * len(p_cols)   # inches per column
    fig_w   = sum(COL_W) + 0.2
    fig_h   = n_rows * ROW_H + 0.8   # +0.8 for title + header

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # Cumulative x positions for columns
    col_x = []
    cx = 0.1
    for w in COL_W:
        col_x.append(cx + w / 2)
        cx += w

    HEADER_Y = fig_h - 0.55
    FONT_SZ  = 6.5

    # ── Draw header ──
    for j, (hdr, x) in enumerate(zip(col_headers, col_x)):
        ax.text(x, HEADER_Y, hdr, ha="center", va="center",
                fontsize=FONT_SZ + 0.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="square,pad=0.1", facecolor=_C_HEADER, linewidth=0))

    # ── Draw data rows ──
    prev_lbl = None
    for i, (lbl, sk, pd_) in enumerate(rows):
        y = HEADER_Y - (i + 1) * ROW_H
        is_flip = (lbl, sk) in qualifying_set
        row_bg  = _C_FLIP_ROW if is_flip else "white"

        # Hand label cell (only show when hand changes)
        hand_display = lbl if lbl != prev_lbl else ""
        hand_bg = _C_HAND_LABEL if lbl != prev_lbl else row_bg
        ax.text(col_x[0], y, hand_display, ha="center", va="center",
                fontsize=FONT_SZ, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.1", facecolor=hand_bg, linewidth=0.3,
                          edgecolor="#BDBDBD"))

        # Scenario cell
        ax.text(col_x[1], y, sk, ha="left", va="center",
                fontsize=FONT_SZ,
                bbox=dict(boxstyle="square,pad=0.1", facecolor=row_bg, linewidth=0.3,
                          edgecolor="#BDBDBD"))

        # Player-count cells
        for j, p in enumerate(p_cols):
            x = col_x[n_fixed + j]
            if p in pd_:
                v = pd_[p]
                half_ci = (v["ci_high"] - v["ci_low"]) / 2
                txt = f"{v['mean']:+.2f}\n±{half_ci:.2f}"
                bg  = _cell_color(v)
            else:
                txt = "N/A"
                bg  = _C_NA
            ax.text(x, y, txt, ha="center", va="center",
                    fontsize=FONT_SZ - 0.5, linespacing=1.2,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor=bg, linewidth=0.3,
                              edgecolor="#BDBDBD"))

        prev_lbl = lbl

    # ── Legend ──
    legend_items = [
        (_C_RAISE_SURE, "Raise (CI > 0)"),
        (_C_RAISE_LEAN, "Raise (CI crosses 0)"),
        (_C_CHECK_SURE, "Check (CI < 0)"),
        (_C_CHECK_LEAN, "Check (CI crosses 0)"),
        (_C_FLIP_ROW,   "Decision flips across # players"),
    ]
    lx = 0.1
    for color, desc in legend_items:
        ax.add_patch(plt.Rectangle((lx, 0.05), 0.18, 0.12, color=color,
                                   transform=ax.transData, clip_on=False))
        ax.text(lx + 0.22, 0.11, desc, va="center", fontsize=5.5)
        lx += 2.0

    fig.suptitle(
        "UTH Collusion Edge — All Hands × Visible-Rank Scenarios (99% CI, ante units)\n"
        "Edge = EV(raise 4x) − EV(optimal check path)",
        fontsize=9, fontweight="bold", y=0.995,
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Table PNG saved to {output_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    print(f"Computing edges for {len(TARGET_HANDS)} hands  (N_CONFIG={N_CONFIG}, N_SIMS={N_SIMS})...")
    all_data = compute_all_scenario_edges(rng, verbose=True)

    qualifying = filter_decision_flips(all_data)
    qualifying_set = set(qualifying)
    print(f"\n{len(qualifying)} hand/scenario combos with decision flipping across player counts.")

    save_table_png(all_data, qualifying_set)

    json_out = {
        lbl: {
            sk: {str(p): v for p, v in pd_.items()}
            for sk, pd_ in scenarios.items()
        }
        for lbl, scenarios in all_data.items()
    }
    with open("collusion_edge_data.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print("JSON written to collusion_edge_data.json")


if __name__ == "__main__":
    main()
