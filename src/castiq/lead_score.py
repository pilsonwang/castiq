"""Predictive lead scoring with a leakage audit, honest uncertainty, and a rep-readable points scorecard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

POINTS_TO_DOUBLE_ODDS = 20  # scorecard convention: +20 points = 2x the odds of converting


@dataclass
class LeadScoreResult:
    model_name: str
    metrics: dict[str, float]
    leakage_audit: dict[str, Any]
    tiers: pd.DataFrame
    scorecard: pd.DataFrame
    model_comparison: pd.DataFrame
    test_scores: pd.DataFrame = field(repr=False)


def split_features(df: pd.DataFrame, cfg: dict[str, Any], include_leaky: bool = False) -> tuple[list, list]:
    drop = set(cfg["id_columns"]) | {cfg["target"]}
    if not include_leaky:
        drop |= set(cfg["leakage_columns"])
    cols = [c for c in df.columns if c not in drop]
    # Numeric = real numbers only. Booleans and every string flavour (object, "string", pandas-3 "str",
    # category) are categorical.
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    return num, cat


def _prep(num: list[str], cat: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                               drop="if_binary"))]), cat),
    ])


def make_models(num: list[str], cat: list[str], seed: int) -> dict[str, Pipeline]:
    return {
        "Logistic regression (scorecard)": Pipeline([
            ("prep", _prep(num, cat)),
            ("clf", LogisticRegression(C=0.1, max_iter=5000)),
        ]),
        "Gradient boosting": Pipeline([
            ("prep", _prep(num, cat)),
            ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=250,
                                                   l2_regularization=1.0, random_state=seed)),
        ]),
    }


def _clean(df: pd.DataFrame, num: list[str], cat: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cat:
        out[c] = out[c].astype("object").where(out[c].notna(), "missing").astype(str)
    for c in num:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


def bootstrap_auc(y: np.ndarray, p: np.ndarray, runs: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(runs):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        stats.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def tier_table(y: np.ndarray, p: np.ndarray, shares: dict[str, float]) -> pd.DataFrame:
    """Rank leads by score and cut into tiers by share. Shows conversion rate and share of all conversions."""
    order = np.argsort(-p, kind="stable")
    n = len(p)
    rows, start = [], 0
    total_conv = max(int(y.sum()), 1)
    names = list(shares)
    for i, (tier, share) in enumerate(shares.items()):
        end = n if i == len(names) - 1 else start + round(share * n)
        idx = order[start:end]
        conv = int(y[idx].sum())
        rows.append({"tier": tier, "leads": len(idx), "conversions": conv,
                     "conversion_rate": conv / len(idx) if len(idx) else 0.0,
                     "share_of_conversions": conv / total_conv,
                     "min_score": float(p[idx].min()) if len(idx) else float("nan")})
        start = end
    out = pd.DataFrame(rows)
    base = y.mean()
    out["lift_vs_avg"] = out["conversion_rate"] / base if base > 0 else np.nan
    return out


def scorecard(model: Pipeline, top: int = 14) -> pd.DataFrame:
    """Turn logistic-regression coefficients into points a rep can read."""
    prep: ColumnTransformer = model.named_steps["prep"]
    clf: LogisticRegression = model.named_steps["clf"]
    names = prep.get_feature_names_out()
    coefs = clf.coef_[0]
    df = pd.DataFrame({"feature": names, "coef": coefs})
    df["points"] = (df["coef"] * POINTS_TO_DOUBLE_ODDS / np.log(2)).round(1)
    df["odds_ratio"] = np.exp(df["coef"])
    cat_cols = list(prep.transformers_[1][2])
    df["feature"] = df["feature"].map(lambda f: readable_feature(f, cat_cols))
    df = df.reindex(df["coef"].abs().sort_values(ascending=False).index).head(top)
    return df.reset_index(drop=True)


def readable_feature(name: str, cat_cols: list[str]) -> str:
    """'cat__industry_manufacturing' -> 'industry = manufacturing'; 'cat__has_open_opportunity_True' ->
    'has open opportunity = yes'; 'num__touch_count' -> 'touch count'."""
    if name.startswith("num__"):
        body = name[5:]
        if body.startswith("missingindicator_"):
            return f"{body[17:].replace('_', ' ')} is missing"
        return body.replace("_", " ")
    body = name[5:] if name.startswith("cat__") else name
    for col in sorted(cat_cols, key=len, reverse=True):
        if body.startswith(col + "_"):
            value = body[len(col) + 1:]
            value = {"True": "yes", "False": "no"}.get(value, value.replace("_", " "))
            return f"{col.replace('_', ' ')} = {value}"
    return body.replace("_", " ")


def run(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, cfg: dict[str, Any]) -> LeadScoreResult:
    seed = int(cfg["random_seed"])
    target = cfg["target"]
    num, cat = split_features(train, cfg)
    tr, va, te = (_clean(d, num, cat) for d in (train, valid, test))
    y_tr, y_va, y_te = (d[target].astype(int).to_numpy() for d in (train, valid, test))

    # 1. Choose the model on the validation split only.
    rows = []
    fitted = {}
    for name, model in make_models(num, cat, seed).items():
        model.fit(tr[num + cat], y_tr)
        p_va = model.predict_proba(va[num + cat])[:, 1]
        rows.append({"model": name, "valid_auc": roc_auc_score(y_va, p_va),
                     "valid_ap": average_precision_score(y_va, p_va)})
        fitted[name] = model
    comparison = pd.DataFrame(rows).sort_values("valid_auc", ascending=False).reset_index(drop=True)
    best_name = str(comparison.iloc[0]["model"])

    # 2. Refit the chosen model on train + valid, score the untouched test split once.
    trva = pd.concat([tr, va], ignore_index=True)
    y_trva = np.concatenate([y_tr, y_va])
    final = make_models(num, cat, seed)[best_name].fit(trva[num + cat], y_trva)
    p_te = final.predict_proba(te[num + cat])[:, 1]
    lo, hi = bootstrap_auc(y_te, p_te, int(cfg["bootstrap_runs"]), seed)
    top_n = min(100, len(p_te))
    top_idx = np.argsort(-p_te)[:top_n]
    metrics = {
        "test_auc": float(roc_auc_score(y_te, p_te)),
        "test_auc_ci_low": lo,
        "test_auc_ci_high": hi,
        "test_average_precision": float(average_precision_score(y_te, p_te)),
        "test_brier": float(brier_score_loss(y_te, p_te)),
        "test_base_rate": float(y_te.mean()),
        "test_rows": len(y_te),
        "test_conversions": int(y_te.sum()),
        "precision_at_100": float(y_te[top_idx].mean()),
        "n_features_numeric": len(num),
        "n_features_categorical": len(cat),
    }
    comparison["test_auc"] = np.nan
    comparison.loc[comparison["model"] == best_name, "test_auc"] = metrics["test_auc"]

    # 3. Leakage audit: what would the "too good to be true" model have reported?
    num_l, cat_l = split_features(train, cfg, include_leaky=True)
    trva_l = _clean(pd.concat([train, valid], ignore_index=True), num_l, cat_l)
    te_l = _clean(test, num_l, cat_l)
    leaky = make_models(num_l, cat_l, seed)["Gradient boosting"].fit(trva_l[num_l + cat_l], y_trva)
    clean_gbm = make_models(num, cat, seed)["Gradient boosting"].fit(trva[num + cat], y_trva)
    auc_leaky = roc_auc_score(y_te, leaky.predict_proba(te_l[num_l + cat_l])[:, 1])
    auc_clean = roc_auc_score(y_te, clean_gbm.predict_proba(te[num + cat])[:, 1])
    leakage: dict[str, Any] = {"columns_removed": ", ".join(cfg["leakage_columns"]),
               "gbm_auc_with_leak": float(auc_leaky), "gbm_auc_without_leak": float(auc_clean),
               "auc_inflation": float(auc_leaky - auc_clean)}

    tiers = tier_table(y_te, p_te, cfg["tier_shares"])
    lr_full = make_models(num, cat, seed)["Logistic regression (scorecard)"].fit(trva[num + cat], y_trva)
    test_scores = test[["lead_id"]].copy()
    test_scores["score"] = p_te
    test_scores["converted"] = y_te
    return LeadScoreResult(best_name, metrics, leakage, tiers, scorecard(lr_full), comparison, test_scores)
