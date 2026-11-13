"""Territory design and rules-based account routing.

Territory design is a balanced, geography-aware partition of states:
1. Start from weighted k-means on state centroids (compact, drivable territories).
2. Move border states from the heaviest territory to a neighbour while it reduces imbalance.
Result: territories a rep can drive, with similar $ potential - so quota can be fair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from castiq.geo import ADJACENCY, STATES, abbr, centroid
from castiq.stats import coefficient_of_variation


@dataclass
class TerritoryPlan:
    assignment: pd.DataFrame  # state, territory, potential_usd, accounts
    summary: pd.DataFrame  # territory, name, states, potential_usd, accounts
    cv: float  # coefficient of variation of potential across territories
    max_min_ratio: float


def _summarize(assign: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    rows = []
    for _, g in assign.groupby("territory"):
        g = g.sort_values("potential_usd", ascending=False)
        tid = int(g["territory"].iloc[0])
        top = [abbr(s) for s in g["state"].head(3)]
        rows.append({
            "territory": tid,
            "name": f"T{tid + 1} · " + "/".join(top),
            "states": ", ".join(sorted(abbr(s) for s in g["state"])),
            "n_states": len(g),
            "potential_usd": float(g["potential_usd"].sum()),
            "accounts": int(g["accounts"].sum()),
        })
    summary = pd.DataFrame(rows).sort_values("potential_usd", ascending=False, kind="mergesort").reset_index(drop=True)
    pot = summary["potential_usd"].to_numpy()
    ratio = float(pot.max() / pot.min()) if pot.min() > 0 else float("inf")
    return summary, coefficient_of_variation(pot), ratio


def _xy(states: list[str]) -> np.ndarray:
    # Equirectangular projection, scaled by cos(latitude), so distances look like miles.
    pts = np.array([centroid(s) for s in states])
    lat, lon = pts[:, 0], pts[:, 1]
    return np.column_stack([lon * np.cos(np.radians(38.0)), lat])


def all_states_potential(state_potential: pd.DataFrame) -> pd.DataFrame:
    """Every lower-48 state + DC, with zero potential where no accounts exist (territories must cover the map)."""
    lower48 = [s for s in STATES if s not in ("Alaska", "Hawaii")]
    base = pd.DataFrame({"state": lower48})
    out = base.merge(state_potential, on="state", how="left").fillna({"potential_usd": 0.0, "accounts": 0})
    # Alaska / Hawaii join the nearest territory only when they have accounts.
    extra = state_potential[state_potential["state"].isin(["Alaska", "Hawaii"])]
    return pd.concat([out, extra], ignore_index=True)


def design_territories(state_potential: pd.DataFrame, k: int, seed: int = 42,
                       balance: bool = True, max_moves: int = 500,
                       compactness_weight: float = 0.02) -> TerritoryPlan:
    """Split states into k contiguous territories with balanced potential.

    balance=False returns the plain geographic k-means start point - the "before" for the report.
    """
    df = all_states_potential(state_potential).reset_index(drop=True)
    states = df["state"].tolist()
    xy = _xy(states)
    w = df["potential_usd"].to_numpy(dtype=float)
    nbrs = _neighbour_index(states)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = km.fit_predict(xy, sample_weight=np.maximum(w, w.max() * 1e-3))
    labels = _repair_contiguity(labels, nbrs, k)
    if balance:
        labels = _rebalance(xy, w, labels, nbrs, k, max_moves, compactness_weight)
    df["territory"] = labels
    summary, cv, ratio = _summarize(df)
    name_map = dict(zip(summary["territory"], summary["name"], strict=True))
    df["territory_name"] = df["territory"].map(name_map)
    return TerritoryPlan(df, summary, cv, ratio)


def _neighbour_index(states: list[str]) -> list[set[int]]:
    pos = {s: i for i, s in enumerate(states)}
    return [{pos[n] for n in ADJACENCY.get(s, set()) if n in pos} for s in states]


def _components(members: set[int], nbrs: list[set[int]]) -> list[set[int]]:
    left, comps = set(members), []
    while left:
        start = left.pop()
        comp, stack = {start}, [start]
        while stack:
            for n in nbrs[stack.pop()]:
                if n in left:
                    left.remove(n)
                    comp.add(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def is_contiguous(labels: np.ndarray, nbrs: list[set[int]], territory: int) -> bool:
    members = set(np.where(labels == territory)[0])
    return len(_components(members, nbrs)) <= 1


def _repair_contiguity(labels: np.ndarray, nbrs: list[set[int]], k: int) -> np.ndarray:
    """k-means works on points, not borders. Keep each territory's largest connected piece and hand
    every stray piece to the bordering territory it touches most."""
    labels = labels.copy()
    for _ in range(100):
        changed = False
        for t in range(k):
            comps = sorted(_components(set(np.where(labels == t)[0]), nbrs), key=len, reverse=True)
            for comp in comps[1:]:
                votes: dict[int, int] = {}
                for i in comp:
                    for n in nbrs[i]:
                        if labels[n] != t:
                            votes[int(labels[n])] = votes.get(int(labels[n]), 0) + 1
                if votes:
                    new = max(votes, key=lambda x: votes[x])
                    for i in comp:
                        labels[i] = new
                    changed = True
        if not changed:
            break
    return labels


def _rebalance(xy: np.ndarray, w: np.ndarray, labels: np.ndarray, nbrs: list[set[int]], k: int,
               max_moves: int, compactness_weight: float) -> np.ndarray:
    """Greedy local search: move one border state at a time to a bordering territory, keeping both
    territories connected, while that lowers (imbalance + compactness penalty)."""
    labels = labels.copy()
    target = w.sum() / k
    spread = float(((xy - xy.mean(axis=0)) ** 2).sum(axis=1).mean())

    def cost(lbl: np.ndarray) -> float:
        loads = np.bincount(lbl, weights=w, minlength=k)
        imbalance = float(((loads / target - 1) ** 2).sum())
        compact = 0.0
        for t in range(k):
            pts = xy[lbl == t]
            if len(pts):
                compact += float(((pts - pts.mean(axis=0)) ** 2).sum()) / spread
        return imbalance + compactness_weight * compact

    current = cost(labels)
    for _ in range(max_moves):
        best: tuple[float, int, int] | None = None
        for i in range(len(xy)):
            own = int(labels[i])
            if (labels == own).sum() == 1:
                continue  # never empty a territory
            for t in {int(labels[n]) for n in nbrs[i]} - {own}:
                trial = labels.copy()
                trial[i] = t
                if not is_contiguous(trial, nbrs, own):
                    continue
                c = cost(trial)
                if c < current - 1e-9 and (best is None or c < best[0]):
                    best = (c, i, t)
        if best is None:
            break
        current, labels[best[1]] = best[0], best[2]
    return labels


def route_accounts(scored: pd.DataFrame, plan: TerritoryPlan, cfg: dict[str, Any]) -> pd.DataFrame:
    """Assign each A/B account to an owner with explicit, auditable rules.

    Rule order (first match wins):
      1. Strategic carve-out: tier and size band in the strategic list -> Strategic team.
      2. No usable state -> Data-review queue (RevOps fixes the record, then it routes).
      3. Territory by state -> round-robin across that territory's reps, highest fit first,
         until each rep reaches capacity; the rest go to the territory's Inside-sales pool.
    """
    tcfg = cfg["territories"]
    strat = tcfg["strategic"]
    reps_per = int(tcfg["reps_per_territory"])
    cap = int(tcfg["rep_capacity_accounts"])
    work = scored[scored["tier"].isin(tcfg["potential_tiers"])].copy()
    work = work.sort_values(["fit_score", "est_annual_spend_usd"], ascending=False, kind="mergesort")

    is_strat = work["tier"].isin(strat["tiers"]) & work["size_band"].isin(strat["size_bands"])
    state_to_t = dict(zip(plan.assignment["state"], plan.assignment["territory_name"], strict=True))
    work["territory"] = work["state"].map(state_to_t)
    work["route_rule"] = np.where(is_strat, "1-strategic",
                                  np.where(work["territory"].isna(), "2-data-review", "3-territory"))
    # Rules 1 and 2 set the owner here; rule 3 (territory round-robin) fills the rest below.
    owners = np.full(len(work), None, dtype=object)
    owners[work["territory"].isna().to_numpy()] = "RevOps data review"
    owners[is_strat.to_numpy()] = "Strategic team"
    terr = work["territory"].to_numpy(dtype=object)
    counters: dict[str, int] = {}
    for idx in np.where(work["route_rule"].to_numpy() == "3-territory")[0]:
        t = terr[idx]
        n = counters.get(t, 0)
        counters[t] = n + 1
        if n < reps_per * cap:
            owners[idx] = f"{t.split(' · ')[0]} Rep {n % reps_per + 1}"
        else:
            owners[idx] = f"{t.split(' · ')[0]} Inside sales"
    work["owner"] = owners
    return work


def owner_load(routed: pd.DataFrame) -> pd.DataFrame:
    return (
        routed.groupby(["route_rule", "owner"])
        .agg(accounts=("handle", "count"), potential_usd=("est_annual_spend_usd", "sum"),
             avg_fit=("fit_score", "mean"))
        .reset_index()
        .sort_values(["route_rule", "potential_usd"], ascending=[True, False])
    )
