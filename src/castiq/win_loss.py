"""Win/loss drivers on 448 REAL B2B deals, translated into an industrial RFQ qualification checklist.

The deals come from an IT solutions seller, but the buying motion is the same as die casting:
RFI -> RFP/RFQ -> (sometimes) formal tender, a purchasing department, competing quotes, a funded budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from castiq.stats import rate_table

# column -> (plain-English question, industrial die-cast equivalent)
FIELDS: dict[str, tuple[str, str]] = {
    "Authority": ("Authority level of our contact", "Do we talk to the person who signs the PO?"),
    "Comp_size": ("Size of the buying company", "OEM / Tier-1 / small job shop"),
    "Competitors": ("Competitors in the deal", "Are other casters quoting?"),
    "Purch_dept": ("Purchasing department involved", "Is Purchasing running the sourcing event?"),
    "Partnership": ("Sold with a partner", "Sold with a machining / finishing partner"),
    "Budgt_alloc": ("Budget allocated", "Is the program funded (tooling PO budget approved)?"),
    "Forml_tend": ("Formal tender", "Formal sourcing event / e-auction"),
    "RFI": ("RFI received", "Early-stage supplier survey / RFI"),
    "RFP": ("RFP received", "Full RFQ package (drawings, volumes, PPAP level)"),
    "Growth": ("Buyer's business is growing", "Is the buyer's program volume growing?"),
    "Posit_statm": ("Buyer made a positive statement", "Engineer said 'you are our preferred source'"),
    "Source": ("Where the deal came from", "Existing relationship / referral / trade show"),
    "Client": ("New or current client", "Already an approved supplier at this plant?"),
    "Scope": ("Scope clarity", "Are drawings, alloy, and volumes clear?"),
    "Strat_deal": ("Strategic importance to us", "Is this a strategic program for us?"),
    "Cross_sale": ("Cross-sale", "Adds a new part family at an existing customer"),
    "Up_sale": ("Up-sale", "More volume / more operations on an existing part"),
    "Deal_type": ("Deal type", "New program / re-source / maintenance"),
    "Needs_def": ("Needs are defined", "Are requirements (specs, tolerances) defined?"),
    "Att_t_client": ("Our attitude to the client", "Strategic account vs. normal vs. first deal"),
}
EXCLUDE = {"Product", "Seller", "Status", "is_won", "deal_id"}


@dataclass
class WinLossResult:
    base_win_rate: float
    n_deals: int
    drivers: pd.DataFrame
    cv_auc_mean: float
    cv_auc_std: float
    score_quartiles: pd.DataFrame
    checklist: pd.DataFrame


def drivers(deals: pd.DataFrame, min_n: int) -> pd.DataFrame:
    df = deals.copy()
    df["won"] = df["is_won"].astype(int)
    frames = []
    for col, (question, diecast) in FIELDS.items():
        if col not in df.columns:
            continue
        t = rate_table(df, col, "won", min_n=min_n).rename(columns={col: "value"})
        t.insert(0, "field", col)
        t.insert(1, "question", question)
        t.insert(2, "die_cast_equivalent", diecast)
        frames.append(t)
    out = pd.concat(frames, ignore_index=True)
    out["abs_lift"] = out["lift_pts"].abs()
    return out.sort_values("abs_lift", ascending=False, kind="mergesort").reset_index(drop=True)


def _features(deals: pd.DataFrame) -> list[str]:
    return [c for c in deals.columns if c not in EXCLUDE and c in FIELDS]


def _pipe(seed: int) -> Pipeline:
    return Pipeline([
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ("lr", LogisticRegression(C=0.3, max_iter=5000, random_state=seed)),
    ])


def run(deals: pd.DataFrame, cfg: dict[str, Any]) -> WinLossResult:
    seed = int(cfg["random_seed"])
    feats = _features(deals)
    x = deals[feats].astype(str)
    y = deals["is_won"].astype(int).to_numpy()

    rcv = RepeatedStratifiedKFold(n_splits=cfg["cv_folds"], n_repeats=cfg["cv_repeats"], random_state=seed)
    aucs = []
    for tr, te in rcv.split(x, y):
        m = _pipe(seed).fit(x.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(x.iloc[te])[:, 1]))

    # Out-of-fold "deal score" -> does a high score really win more often?
    oof = cross_val_predict(_pipe(seed), x, y, method="predict_proba",
                            cv=StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=seed))[:, 1]
    q = pd.qcut(oof, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    quart = (pd.DataFrame({"quartile": q, "won": y}).groupby("quartile", observed=True)["won"]
             .agg(["mean", "count"]).reset_index().rename(columns={"mean": "win_rate", "count": "deals"}))

    drv = drivers(deals, int(cfg["min_group_n"]))
    checklist = build_checklist(drv)
    return WinLossResult(float(y.mean()), len(y), drv, float(np.mean(aucs)), float(np.std(aucs)), quart,
                         checklist)


def build_checklist(drv: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """The strongest *significant* signal per field, phrased as a qualification question for a rep."""
    sig = drv[drv["significant"]].sort_values("abs_lift", ascending=False, kind="mergesort")
    best = sig.drop_duplicates("field").head(top)
    rows = []
    for _, r in best.iterrows():
        direction = "raises" if r["lift_pts"] > 0 else "lowers"
        rows.append({
            "ask_the_buyer": r["die_cast_equivalent"],
            "signal": f"{r['question']} = {r['value']}",
            "effect": f"{direction} win rate by {abs(r['lift_pts']):.0f} pts",
            "win_rate": r["rate"],
            "n": int(r["n"]),
            "lift_pts": r["lift_pts"],
        })
    return pd.DataFrame(rows)
