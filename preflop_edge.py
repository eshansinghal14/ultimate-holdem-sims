"""
Collusion Edge Calculator for Ultimate Texas Hold'em.

For each TARGET_HAND × visible-rank scenario × num_players, computes
the 99% confidence interval of EV(raise 4x) − EV(check path).

Outputs:
  - collusion_edge_chart.png  (only combos where decision flips across player counts)
  - collusion_edge_data.json
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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

Z_99 = 2.576  # z-score for 99% CI (fixed)

MAX_PLAYERS = 6

# Hands to analyze — borderline raise/check decisions in standard UTH strategy.
TARGET_HANDS: List[Tuple[int, int, bool]] = [
    # A2–A4
    (14, 2, True), (14, 2, False),
    (14, 3, True), (14, 3, False),
    (14, 4, False),
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


# ── Hand string parsing ───────────────────────────────────────────────────────

_NAME_TO_RANK = {v: k for k, v in RANK_NAMES.items()}  # "A"->14, "T"->10, etc.


def _parse_hand_str(s: str) -> Tuple[int, int, bool]:
    """
    Parse a hand string like 'A2s', 'k3o', 'JTo', '22' into (r1, r2, suited).
    r1 >= r2. Pairs may omit the trailing 'o'.
    Raises ValueError on bad input.
    """
    s = s.upper().strip()
    if len(s) == 2:
        # Pair with no suffix, e.g. "22", "KK"
        r = _NAME_TO_RANK.get(s[0])
        if r is None or s[0] != s[1]:
            raise ValueError(f"Invalid hand: '{s}' — 2-char form must be a pocket pair (e.g. '22')")
        return r, r, False
    if len(s) == 3:
        r1 = _NAME_TO_RANK.get(s[0])
        r2 = _NAME_TO_RANK.get(s[1])
        suit_char = s[2]
        if r1 is None or r2 is None:
            raise ValueError(f"Invalid rank in '{s}'")
        if suit_char not in ("S", "O"):
            raise ValueError(f"Invalid suit suffix in '{s}' — must be 's' or 'o'")
        suited = suit_char == "S"
        hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
        return hi, lo, suited
    raise ValueError(f"Cannot parse hand '{s}' — expected 2 or 3 characters (e.g. 'A2s', '22')")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_scenarios(r1: int, r2: int, base_deck: List[Card],
                     max_outs: int = 1) -> Dict[str, Dict[int, int]]:
    """
    Build visible-rank scenarios for a hand.
    Returns {scenario_key: {rank: min_count_required_in_colluder_cards}}.

    Non-pairs: enumerate all (n_high, n_low) where each ranges 0..min(max_outs, avail).
    Pocket pairs: enumerate n=1..min(max_outs, 2) — at most 2 remain after player holds 2.
    """
    avail_r1 = sum(1 for c in base_deck if c.rank == r1)
    avail_r2 = sum(1 for c in base_deck if c.rank == r2) if r2 != r1 else avail_r1
    is_pair = (r1 == r2)

    scenarios: Dict[str, Dict[int, int]] = {"none_seen": {}}

    if is_pair:
        cap = min(max_outs, avail_r1)
        for n in range(1, cap + 1):
            scenarios[f"{RANK_NAMES[r1]}_seen={n}"] = {r1: n}
    else:
        cap_r1 = min(max_outs, avail_r1)
        cap_r2 = min(max_outs, avail_r2)
        for n1 in range(0, cap_r1 + 1):
            for n2 in range(0, cap_r2 + 1):
                if n1 == 0 and n2 == 0:
                    continue  # already "none_seen"
                parts = []
                required: Dict[int, int] = {}
                if n1 > 0:
                    parts.append(f"{RANK_NAMES[r1]}_seen={n1}")
                    required[r1] = n1
                if n2 > 0:
                    parts.append(f"{RANK_NAMES[r2]}_seen={n2}")
                    required[r2] = n2
                scenarios[",".join(parts)] = required

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

def _compute_task(args_tuple) -> tuple:
    """
    Worker: compute edge for one (hand, scenario, num_players) cell.
    Returns (lbl, sk, num_players, entry_or_None, log_line_or_None).
    """
    task_idx, n_tasks, r1, r2, suited, sk, required, num_players, n_config, n_sims, seed_offset = args_tuple
    rng = np.random.default_rng(seed_offset)

    lbl = hand_label(r1, r2, suited)
    p1, p2 = representative_cards(r1, r2, suited)
    player_cards = [p1, p2]
    base_deck = remove_cards(create_deck(), player_cards)

    n_col = num_players - 1
    total_req = sum(required.values())

    if n_col == 0:
        edges = [calc_edge(player_cards, [], rng, n_sims) for _ in range(n_config)]
    else:
        edges = []
        for _ in range(n_config):
            cc = _sample_constrained(rng, base_deck, n_col, required)
            if cc is not None:
                edges.append(calc_edge(player_cards, cc, rng, n_sims))

    if not edges:
        return lbl, sk, num_players, None, None

    mean, ci_low, ci_high = _ci99(edges)
    entry = {
        "mean":     round(mean,    4),
        "ci_low":   round(ci_low,  4),
        "ci_high":  round(ci_high, 4),
        "decision": "raise" if mean > 0 else "check",
    }
    dec = "RAISE" if entry["decision"] == "raise" else "check"
    noise = float(np.std(edges, ddof=1)) if len(edges) > 1 else 0.0
    log_line = (
        f"  [{task_idx+1}/{n_tasks}] {lbl:5s} | {sk:30s} | "
        f"P{num_players}: {mean:+.3f} [{ci_low:+.3f}, {ci_high:+.3f}] {dec}  std={noise:.3f}"
    )
    return lbl, sk, num_players, entry, log_line


def _build_work_items(seed: int, n_config: int, n_sims: int,
                      player_counts: List[int],
                      hands: List[Tuple[int, int, bool]],
                      max_outs: int = 1) -> list:
    """Enumerate all valid (hand, scenario, num_players) tasks."""
    raw = []
    for r1, r2, suited in hands:
        p1, p2 = representative_cards(r1, r2, suited)
        base_deck = remove_cards(create_deck(), [p1, p2])
        scenarios = _build_scenarios(r1, r2, base_deck, max_outs)
        for sk, required in scenarios.items():
            total_req = sum(required.values())
            for num_players in player_counts:
                n_col = num_players - 1
                if n_col == 0 and sk != "none_seen":
                    continue
                if n_col > 0 and total_req > n_col * 2:
                    continue
                raw.append((r1, r2, suited, sk, required, num_players))
    n_tasks = len(raw)
    return [
        (i, n_tasks, r1, r2, suited, sk, required, num_players,
         n_config, n_sims, seed * 100_000 + i)
        for i, (r1, r2, suited, sk, required, num_players) in enumerate(raw)
    ]


def compute_all_scenario_edges(
    seed: int,
    verbose: bool = True,
    n_config: int = 50,
    n_sims: int = 100,
    num_players: int = MAX_PLAYERS,
    hands: Optional[List[Tuple[int, int, bool]]] = None,
    max_outs: int = 1,
) -> Dict:
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
    active_hands = hands if hands is not None else TARGET_HANDS
    work_items = _build_work_items(seed, n_config, n_sims, [num_players], active_hands, max_outs)
    n_tasks = len(work_items)
    if verbose:
        print(f"  Total tasks: {n_tasks}  (hands={len(active_hands)}, players={num_players}, cores={os.cpu_count()})")

    result: Dict = {}

    def _apply(lbl, sk, num_players, entry, log_line):
        if entry is None:
            return
        result.setdefault(lbl, {}).setdefault(sk, {})[num_players] = entry
        if verbose and log_line:
            print(log_line)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(_compute_task, item) for item in work_items]
        for fut in as_completed(futures):
            _apply(*fut.result())

    # Return in active_hands order, scenarios in insertion order
    ordered: Dict = {}
    for r1, r2, suited in active_hands:
        lbl = hand_label(r1, r2, suited)
        if lbl not in result:
            continue
        p1, p2 = representative_cards(r1, r2, suited)
        base_deck = remove_cards(create_deck(), [p1, p2])
        scenario_keys = list(_build_scenarios(r1, r2, base_deck, max_outs).keys())
        ordered[lbl] = {sk: result[lbl][sk]
                        for sk in scenario_keys if sk in result[lbl]}
    return ordered


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


def _normalize_sk(r1: int, r2: int, sk: str) -> str:
    """Normalize scenario key to hi/lo form so all hands share column headers."""
    if sk == "none_seen" or r1 == r2:
        if r1 == r2 and sk != "none_seen":
            return sk.replace(f"{RANK_NAMES[r1]}_seen", "seen")
        return sk
    hi, lo = RANK_NAMES[r1], RANK_NAMES[r2]
    return sk.replace(f"{hi}_seen", "high_seen").replace(f"{lo}_seen", "low_seen")


def save_table_png(
    all_data: Dict,
    output_path: str = "collusion_edge_chart.png",
    num_players: int = 1,
) -> None:
    """
    Render hands as rows, visible-rank scenarios as columns.

    Scenario keys are normalized to hi/lo so all non-pair hands share columns.
    Cell format: mean ± half_CI   Cell color: green=raise, red=check.
    """
    # Build ordered column list (normalized scenario keys, in first-seen order)
    col_order: Dict[str, int] = {}
    for lbl, scenarios in all_data.items():
        r1, r2, _ = _parse_hand_str(lbl)
        for sk in scenarios:
            nsk = _normalize_sk(r1, r2, sk)
            if nsk not in col_order:
                col_order[nsk] = len(col_order)
    scenario_cols = sorted(col_order, key=lambda k: col_order[k])

    n_rows = len(all_data)
    n_cols = len(scenario_cols)

    ROW_H  = 0.32
    COL_W  = [0.65] + [1.3] * n_cols
    fig_w  = sum(COL_W) + 0.2
    fig_h  = n_rows * ROW_H + 0.9

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    col_x = []
    cx = 0.1
    for w in COL_W:
        col_x.append(cx + w / 2)
        cx += w

    HEADER_Y = fig_h - 0.55
    FONT_SZ  = 6.5

    # ── Header row ──
    headers = ["Hand"] + scenario_cols
    for hdr, x in zip(headers, col_x):
        ax.text(x, HEADER_Y, hdr, ha="center", va="center",
                fontsize=FONT_SZ + 0.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="square,pad=0.1", facecolor=_C_HEADER, linewidth=0))

    # ── Data rows (one per hand) ──
    for i, (lbl, scenarios) in enumerate(all_data.items()):
        r1, r2, _ = _parse_hand_str(lbl)
        y = HEADER_Y - (i + 1) * ROW_H

        ax.text(col_x[0], y, lbl, ha="center", va="center",
                fontsize=FONT_SZ, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.1", facecolor=_C_HAND_LABEL,
                          linewidth=0.3, edgecolor="#BDBDBD"))

        # Build normalized-key → entry map for this hand
        nsk_to_entry: Dict[str, dict] = {}
        for raw_sk, pd_ in scenarios.items():
            entry = pd_.get(num_players)
            if entry is not None:
                nsk_to_entry[_normalize_sk(r1, r2, raw_sk)] = entry

        for j, nsk in enumerate(scenario_cols):
            x = col_x[1 + j]
            if nsk in nsk_to_entry:
                v = nsk_to_entry[nsk]
                half_ci = (v["ci_high"] - v["ci_low"]) / 2
                txt = f"{v['mean']:+.2f}\n±{half_ci:.2f}"
                bg  = _cell_color(v)
            else:
                txt = "N/A"
                bg  = _C_NA
            ax.text(x, y, txt, ha="center", va="center",
                    fontsize=FONT_SZ - 0.5, linespacing=1.2,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor=bg,
                              linewidth=0.3, edgecolor="#BDBDBD"))

    # ── Legend ──
    legend_items = [
        (_C_RAISE_SURE, "Raise (CI > 0)"),
        (_C_RAISE_LEAN, "Raise (CI crosses 0)"),
        (_C_CHECK_SURE, "Check (CI < 0)"),
        (_C_CHECK_LEAN, "Check (CI crosses 0)"),
    ]
    lx = 0.1
    for color, desc in legend_items:
        ax.add_patch(plt.Rectangle((lx, 0.05), 0.18, 0.12, color=color,
                                   transform=ax.transData, clip_on=False))
        ax.text(lx + 0.22, 0.11, desc, va="center", fontsize=5.5)
        lx += 2.2

    fig.suptitle(
        f"UTH Collusion Edge — {num_players}-player table (99% CI, ante units)\n"
        "Edge = EV(raise 4x) − EV(optimal check path)",
        fontsize=9, fontweight="bold", y=0.995,
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Table PNG saved to {output_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="UTH Collusion Edge Calculator — compute EV(raise 4x) − EV(check) "
                    "for borderline hands across player counts and visible-rank scenarios."
    )
    p.add_argument("--n-config", type=int, default=50,
                   help="Colluder-hand samples per (hand, scenario, num_players) [default: 50]")
    p.add_argument("--n-sims", type=int, default=100,
                   help="Monte Carlo board runouts per edge estimate [default: 100]")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed [default: 42]")
    p.add_argument("--table-out", default="collusion_edge_chart.png",
                   help="Output path for the PNG table [default: collusion_edge_chart.png]")
    p.add_argument("--json-out", default="collusion_edge_data.json",
                   help="Output path for the JSON data [default: collusion_edge_data.json]")
    p.add_argument("--num-players", type=int, default=MAX_PLAYERS,
                   help=f"Number of players at the table to simulate [default: {MAX_PLAYERS}]")
    p.add_argument("--target-hands", nargs="+", metavar="HAND",
                   help="Hands to analyze, e.g. A2s A2o K3o 22. Overrides built-in list.")
    p.add_argument("--max-outs", type=int, default=1,
                   help="Max copies of each hole-card rank to check as visible [default: 1]")
    return p.parse_args()


def main():
    args = _parse_args()
    n_config = args.n_config
    n_sims   = args.n_sims
    seed     = args.seed

    num_players = args.num_players
    if not (1 <= num_players <= MAX_PLAYERS):
        print(f"--num-players must be between 1 and {MAX_PLAYERS}")
        import sys; sys.exit(1)

    hands: Optional[List[Tuple[int, int, bool]]] = None
    if args.target_hands:
        try:
            hands = [_parse_hand_str(h) for h in args.target_hands]
        except ValueError as e:
            print(f"Error in --target-hands: {e}")
            import sys; sys.exit(1)

    n_hands = len(hands) if hands is not None else len(TARGET_HANDS)
    print(f"Computing edges for {n_hands} hands  "
          f"(--n-config={n_config}, --n-sims={n_sims}, --seed={seed}, "
          f"--num-players={num_players})...")
    all_data = compute_all_scenario_edges(seed=seed, verbose=True,
                                          n_config=n_config, n_sims=n_sims,
                                          num_players=num_players,
                                          hands=hands,
                                          max_outs=args.max_outs)

    save_table_png(all_data, output_path=args.table_out, num_players=num_players)

    json_out = {
        lbl: {
            sk: {str(p): v for p, v in pd_.items()}
            for sk, pd_ in scenarios.items()
        }
        for lbl, scenarios in all_data.items()
    }
    with open(args.json_out, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"JSON written to {args.json_out}")


if __name__ == "__main__":
    main()
