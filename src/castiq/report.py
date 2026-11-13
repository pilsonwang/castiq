"""Render the static dashboard (docs/index.html).

Charts are Plotly figures serialised to JSON. Colours are written as tokens (T["series1"], T["ink"], ...)
and resolved in the browser from CSS custom properties, so light and dark mode each use their own
validated colour steps (see the `dataviz` palette in the template).
"""

from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment

from castiq.analyze import Results
from castiq.geo import STATES, abbr, tile


class _Tokens(dict):
    """Placeholder colours. Plotly validates colours in Python, so each design token gets a unique,
    unlikely hex value; after serialising, `detokenize` swaps it for "@token", which the page resolves
    from CSS custom properties (light / dark aware)."""

    def __missing__(self, key: str) -> str:
        value = f"#c0{len(self):02x}{len(self):02x}"[:7]
        self[key] = value
        return value


T = _Tokens()


def detokenize(js: str) -> str:
    for name, hexval in T.items():
        js = js.replace(f'"{hexval}"', f'"@{name}"')
    return js


ORD = [T["ord4"], T["ord3"], T["ord2"], T["ord1"]]  # tier A strongest ... tier D lightest


# ---------------------------------------------------------------- formatting helpers
def money(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    a = abs(x)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"${x / div:,.{digits}f}{suf}"
    return f"${x:,.0f}"


def pct(x: float | None, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x * 100:.{digits}f}%"


def num(x: float | None, digits: int = 0) -> str:
    if x is None:
        return "–"
    return f"{x:,.{digits}f}"


# ---------------------------------------------------------------- figure helpers
def _base(fig: go.Figure, height: int = 340, **kw: Any) -> go.Figure:
    kw.setdefault("bargap", 0.35)
    kw.setdefault("margin", {"l": 8, "r": 16, "t": 12, "b": 8})
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "system-ui, -apple-system, 'Segoe UI', sans-serif", "size": 13, "color": T["ink2"]},
        hoverlabel={"bgcolor": T["surface"], "bordercolor": T["grid"], "font": {"color": T["ink"]}},
        showlegend=False, **kw,
    )
    axis = {"gridcolor": T["grid"], "linecolor": T["axis"], "zerolinecolor": T["axis"],
            "tickfont": {"color": T["muted"]}, "title": {"font": {"color": T["ink2"]}}, "automargin": True}
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def fig_market_industries(frame: pd.DataFrame) -> go.Figure:
    d = frame.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["spend_usd"], y=d["industry"].str.slice(0, 42), orientation="h", marker={"color": T["series1"],
                                                                                         "cornerradius": 4},
        customdata=np.stack([d["accounts"], d["tier_ab"], d["industry_fit"], d["spend_usd"].map(money)], axis=1),
        hovertemplate="<b>%{y}</b><br>Est. spend %{customdata[3]}<br>%{customdata[0]:,} companies"
                      " · %{customdata[1]:,} tier A/B<br>Industry fit %{customdata[2]:.2f}<extra></extra>"))
    return _base(fig, 380, xaxis={"title": "Estimated annual die-cast spend (USD)", "tickprefix": "$",
                                  "exponentformat": "B"})


def fig_tiers_market(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=frame["tier"], y=frame["accounts"], marker={"color": ORD, "cornerradius": 4},
        customdata=np.stack([frame["spend_usd"].map(money), frame["avg_fit"]], axis=1),
        text=[f"{v:,.0f}" for v in frame["accounts"]], textposition="outside",
        textfont={"color": T["ink2"]}, cliponaxis=False,
        hovertemplate="Tier %{x}<br>%{y:,} companies<br>Spend %{customdata[0]}"
                      "<br>Avg fit %{customdata[1]:.0f}<extra></extra>"))
    return _base(fig, 300, xaxis={"title": "ICP tier"}, yaxis={"title": "Companies", "tickformat": ",.0f"})


def fig_tile_map(states: pd.DataFrame) -> go.Figure:
    """Tile-grid map. Tile colour = A/B potential in 5 classes of one hue; thick lines = territory borders.

    Tiles are shapes in data units (not fixed-pixel markers), so the map scales down to phone width.
    """
    d = states.set_index("state")
    pot = d["potential_usd"]
    vmax = float(pot.max()) or 1.0
    fig = go.Figure()
    xs, ys, txt, hov, ink = [], [], [], [], []
    for s in STATES:
        if s not in d.index:
            continue
        r, c = tile(s)
        share = float(pot[s]) / vmax
        k = min(4, int(share * 5))  # 5 equal-width classes
        fig.add_shape(type="rect", x0=c - 0.46, x1=c + 0.46, y0=-r - 0.46, y1=-r + 0.46, layer="below",
                      line={"width": 0}, fillcolor=T[f"seq{k}"])
        xs.append(c)
        ys.append(-r)
        terr = str(d.loc[s, "territory_name"])
        txt.append(f"<b>{abbr(s)}</b><br>{terr.split(' · ')[0]}")
        hov.append(f"<b>{s}</b><br>{terr}<br>A/B potential {money(float(pot[s]))}"
                   f"<br>{int(float(str(d.loc[s, 'accounts']))):,} accounts")
        ink.append(T[f"tileink{k}"])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=txt, hovertext=hov, hoverinfo="text",
        marker={"symbol": "square", "size": 30, "opacity": 0},  # invisible hit target for hover
        textfont={"color": ink, "size": 11}))
    # Territory borders between adjacent tiles that belong to different territories.
    pos = {tile(s): s for s in d.index if s in STATES}
    for (r, c), s in pos.items():
        for dr, dc in ((0, 1), (1, 0)):
            n = pos.get((r + dr, c + dc))
            if n and d.loc[n, "territory"] != d.loc[s, "territory"]:
                if dc:
                    x0 = x1 = c + 0.5
                    y0, y1 = -r - 0.5, -r + 0.5
                else:
                    y0 = y1 = -r - 0.5
                    x0, x1 = c - 0.5, c + 0.5
                fig.add_shape(type="line", x0=x0, x1=x1, y0=y0, y1=y1, line={"color": T["ink"], "width": 3})
    fig = _base(fig, 430, meta="tilemap")
    fig.update_xaxes(visible=False, range=[-0.6, 10.6], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-7.6, 0.6], scaleanchor="x", fixedrange=True)
    return fig


def fig_territory_balance(before: pd.DataFrame, after: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=("Before", "After"))
    for i, (frame, color) in enumerate(((before, T["neutral"]), (after, T["series1"])), start=1):
        f = frame.sort_values("potential_usd", ascending=False)
        fig.add_trace(go.Bar(x=[n.split(" · ")[0] for n in f["name"]], y=f["potential_usd"],
                             marker={"color": color, "cornerradius": 4},
                             customdata=np.stack([f["name"], f["states"], f["potential_usd"].map(money)], axis=1),
                             hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[2]}<br>%{customdata[1]}"
                                           "<extra></extra>"), row=1, col=i)
    target = float(after["potential_usd"].mean())
    for i in (1, 2):
        fig.add_hline(y=target, line={"color": T["muted"], "width": 1}, row=1, col=i)
    fig.add_annotation(x=1, xref="paper", y=target, text="even split", showarrow=False, yshift=10,
                       xanchor="right", font={"color": T["muted"], "size": 11})
    fig = _base(fig, 330, yaxis={"title": "A/B potential (USD)", "tickprefix": "$", "exponentformat": "B"},
                margin={"l": 8, "r": 16, "t": 36, "b": 8})
    fig.update_annotations(font={"color": T["ink2"]})
    return fig


def fig_lead_tiers(tiers: pd.DataFrame, base: float) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=tiers["tier"], y=tiers["conversion_rate"], marker={"color": ORD, "cornerradius": 4},
        customdata=np.stack([tiers["leads"], tiers["conversions"], tiers["share_of_conversions"],
                             tiers["lift_vs_avg"]], axis=1),
        text=[pct(v, 0) for v in tiers["conversion_rate"]], textposition="outside", cliponaxis=False,
        textfont={"color": T["ink2"]},
        hovertemplate="Tier %{x}<br>Converts %{y:.1%}<br>%{customdata[1]} of %{customdata[0]} leads"
                      "<br>%{customdata[2]:.0%} of all conversions · %{customdata[3]:.2f}x avg<extra></extra>"))
    fig.add_hline(y=base, line={"color": T["muted"], "width": 1})
    fig.add_annotation(x=1, xref="paper", y=base, text=f"average {pct(base)}", showarrow=False,
                       yshift=10, xanchor="right", font={"color": T["muted"], "size": 11})
    return _base(fig, 300, xaxis={"title": "Score tier (A = top 10% of leads)"},
                 yaxis={"title": "90-day conversion rate", "tickformat": ".0%"})


def fig_ci_dots(frame: pd.DataFrame, label_col: str, overall: float, title: str, height: int = 320) -> go.Figure:
    f = frame.sort_values("rate")
    colors = [T["series1"] if s else T["neutral"] for s in f["significant"]]
    fig = go.Figure(go.Scatter(
        x=f["rate"], y=f[label_col].astype(str), mode="markers",
        marker={"size": 11, "color": colors, "line": {"width": 2, "color": T["surface"]}},
        error_x={"type": "data", "symmetric": False, "array": f["ci_high"] - f["rate"],
                 "arrayminus": f["rate"] - f["ci_low"], "color": T["axis"], "thickness": 2, "width": 0},
        customdata=np.stack([f["n"], f["ci_low"], f["ci_high"]], axis=1),
        hovertemplate="<b>%{y}</b><br>Win rate %{x:.1%}<br>95% CI %{customdata[1]:.1%}–%{customdata[2]:.1%}"
                      "<br>n = %{customdata[0]:,} closed deals<extra></extra>"))
    fig.add_vline(x=overall, line={"color": T["muted"], "width": 1})
    return _base(fig, height, xaxis={"title": title, "tickformat": ".0%"})


def fig_funnel(funnel: pd.DataFrame) -> go.Figure:
    f = funnel.sort_values("step_order")
    conv = [None] + [f"{b / a:.0%} of previous" if a else "" for a, b in zip(f["n"][:-1], f["n"][1:],
                                                                             strict=True)]
    fig = go.Figure(go.Bar(
        x=f["n"], y=f["step"], orientation="h", marker={"color": T["series1"], "cornerradius": 4},
        text=[f"{v:,}" for v in f["n"]], textposition="outside", cliponaxis=False, textfont={"color": T["ink2"]},
        customdata=[c or "" for c in conv],
        hovertemplate="<b>%{y}</b><br>%{x:,}<br>%{customdata}<extra></extra>"))
    fig = _base(fig, 280, xaxis={"title": "Records", "tickformat": ",.0f", "range": [0, float(f["n"].max()) * 1.15]})
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_forecast(sims: np.ndarray, m: dict[str, Any]) -> go.Figure:
    hist, edges = np.histogram(sims, bins=60)
    mids = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=mids, y=hist / hist.sum(), marker={"color": T["series1"]},
                           customdata=[f"{money(a)} – {money(b)}" for a, b in itertools.pairwise(edges)],
                           hovertemplate="Bookings %{customdata}<br>%{y:.1%} of simulations<extra></extra>"))
    marks = [("commit_p10", "Commit (P10)", "right", 0.0), ("most_likely_p50", "Most likely", "center", 1.0),
             ("best_case_p90", "Best case (P90)", "left", 0.0)]
    for key, label, anchor, row in marks:
        fig.add_vline(x=m[key], line={"color": T["ink2"], "width": 1})
        fig.add_annotation(x=m[key], y=1.0 + 0.11 * row, yref="paper", text=f"{label} {money(m[key])}",
                           showarrow=False, yanchor="bottom", xanchor=anchor, font={"size": 11, "color": T["ink2"]})
    fig.add_vline(x=m["forecast_stage_default"], line={"color": T["critical"], "width": 2})
    fig.add_annotation(x=m["forecast_stage_default"], y=0.55, yref="paper", xanchor="right",
                       text=f"CRM stage defaults<br><b>{money(m['forecast_stage_default'])}</b><br>"
                            f"P(reach) = {pct(m['prob_reaching_stage_default'], 1)}",
                       showarrow=False, font={"size": 11, "color": T["ink"]}, align="right", xshift=-6)
    lo = min(float(sims.min()), m["forecast_stage_default"]) * 0.95
    hi = max(float(sims.max()), m["forecast_stage_default"]) * 1.03
    return _base(fig, 330, bargap=0.05, margin={"l": 8, "r": 16, "t": 64, "b": 8},
                 xaxis={"title": "Bookings from today's open pipeline (USD)", "tickprefix": "$", "exponentformat": "B",
                        "range": [lo, hi]},
                 yaxis={"title": "Share of simulations", "tickformat": ".0%"})


def fig_calibration(calib: pd.DataFrame) -> go.Figure:
    lo = float(min(calib.min().min(), 0.2))
    hi = float(max(calib.max().max(), 0.8))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"color": T["axis"], "width": 1},
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=calib["predicted"], y=calib["actual"], mode="lines+markers",
                             line={"color": T["series1"], "width": 2},
                             marker={"size": 9, "line": {"width": 2, "color": T["surface"]}},
                             hovertemplate="Predicted %{x:.0%}<br>Actually won %{y:.0%}<extra></extra>"))
    return _base(fig, 300, xaxis={"title": "Predicted win probability", "tickformat": ".0%"},
                 yaxis={"title": "Actual win rate", "tickformat": ".0%"})


def fig_drivers(drv: pd.DataFrame, top: int = 14) -> go.Figure:
    d = drv[drv["significant"]].head(top).copy()
    d["label"] = d["question"] + " = " + d["value"].astype(str)
    d = d.sort_values("lift_pts")
    colors = [T["pos"] if v > 0 else T["neg"] for v in d["lift_pts"]]
    fig = go.Figure(go.Bar(
        x=d["lift_pts"], y=d["label"], orientation="h", marker={"color": colors, "cornerradius": 4},
        customdata=np.stack([d["rate"], d["n"], d["die_cast_equivalent"]], axis=1),
        hovertemplate="<b>%{y}</b><br>Win rate %{customdata[0]:.0%} (n=%{customdata[1]})"
                      "<br>%{x:+.0f} pts vs average<br><i>Die-cast: %{customdata[2]}</i><extra></extra>"))
    fig.add_vline(x=0, line={"color": T["axis"], "width": 1})
    return _base(fig, 470, xaxis={"title": "Win-rate difference vs. average (percentage points)",
                                  "ticksuffix": " pts"})


def fig_quartiles(q: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=q["quartile"].astype(str), y=q["win_rate"], marker={"color": ORD[::-1], "cornerradius": 4},
        text=[pct(v) for v in q["win_rate"]], textposition="outside", cliponaxis=False,
        textfont={"color": T["ink2"]}, customdata=q["deals"],
        hovertemplate="%{x}<br>Won %{y:.0%} of %{customdata} deals<extra></extra>"))
    return _base(fig, 300, xaxis={"title": "Deal score quartile (out-of-fold)"},
                 yaxis={"title": "Win rate", "tickformat": ".0%", "range": [0, 1.05]})


def fig_forecast_reps(by_rep: pd.DataFrame) -> go.Figure:
    f = by_rep.sort_values("forecast_model")
    fig = go.Figure(go.Bar(
        x=f["forecast_model"], y=f["owner_rep_id"], orientation="h", marker={"color": T["series1"],
                                                                             "cornerradius": 4},
        customdata=np.stack([f["forecast_stage_default"].map(money), f["open_deals"], f["pipeline_acv"].map(money),
                             f["forecast_model"].map(money)], axis=1),
        hovertemplate="<b>%{y}</b><br>Model forecast %{customdata[3]}<br>CRM-default forecast %{customdata[0]}"
                      "<br>%{customdata[1]} open deals · pipeline %{customdata[2]}<extra></extra>"))
    return _base(fig, 330, xaxis={"title": "Model-based forecast (USD)", "tickprefix": "$", "exponentformat": "B"})


# ---------------------------------------------------------------- page
def _table(df: pd.DataFrame, formats: dict[str, Any], rename: dict[str, str] | None = None,
           limit: int | None = None) -> str:
    d = df.head(limit) if limit else df
    cols = list(formats)
    head = "".join(f"<th>{(rename or {}).get(c, c)}</th>" for c in cols)
    rows = []
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            fmt = formats[c]
            text = fmt(v) if callable(fmt) else ("" if pd.isna(v) else str(v))
            klass = ' class="num"' if callable(fmt) and fmt in (money, pct, num) else ""
            cells.append(f"<td{klass}>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def build(results: Results, out_path: Path) -> Path:
    m = results.metrics
    f = results.frames
    figs: dict[str, go.Figure] = {
        "market_industries": fig_market_industries(f["market_by_industry"]),
        "market_tiers": fig_tiers_market(f["market_by_tier"]),
        "tile_map": fig_tile_map(f["territory_states"]),
        "territory_balance": fig_territory_balance(f["territories_before"], f["territories"]),
        "lead_tiers": fig_lead_tiers(f["lead_tiers"], m["lead_scoring"]["test_base_rate"]),
        "funnel": fig_funnel(f["funnel"]),
        "seg_source": fig_ci_dots(f["winrate::Lead source"], "lead_source", m["pipeline"]["win_rate"],
                                  "Win rate by lead source (95% CI)", 240),
        "seg_industry": fig_ci_dots(f["winrate::Industry"], "industry", m["pipeline"]["win_rate"],
                                    "Win rate by industry (95% CI)", 260),
        "reps": fig_ci_dots(f["reps"], "owner_rep_id", m["pipeline"]["win_rate"],
                            "Win rate by rep (95% CI)", 380),
        "forecast": fig_forecast(results.arrays["forecast_sims"], m["forecast"]),
        "calibration": fig_calibration(f["calibration"]),
        "forecast_reps": fig_forecast_reps(f["forecast_by_rep"]),
        "drivers": fig_drivers(f["win_drivers"]),
        "quartiles": fig_quartiles(f["win_quartiles"]),
    }
    fig_json = {k: json.loads(detokenize(v.to_json())) for k, v in figs.items()}

    sev_icon = {"high": "▲ high", "medium": "● medium", "low": "○ low"}
    status = {"PASS": '<span class="ok">✓ pass</span>', "FAIL": '<span class="bad">✕ fail</span>'}
    tables = {
        "territories": _table(f["territories"], {"name": str, "states": str, "potential_usd": money,
                                                   "accounts": num},
                              {"name": "Territory", "states": "States", "potential_usd": "A/B potential",
                               "accounts": "Accounts"}),
        "top_accounts": _table(f["top_accounts"], {"name": str, "industry": str, "size_band": str,
                                                    "city": str, "state": str, "fit_score": lambda v: f"{v:.0f}",
                                                    "tier": str},
                               {"name": "Company", "industry": "Industry", "size_band": "Employees",
                                "city": "City", "state": "State", "fit_score": "Fit", "tier": "Tier"}, 12),
        "scorecard": _table(f["lead_scorecard"], {"feature": str, "points": lambda v: f"{v:+.1f}",
                                                   "odds_ratio": lambda v: f"{v:.2f}×"},
                            {"feature": "Signal", "points": "Points", "odds_ratio": "Odds ratio"}, 10),
        "checklist": _table(f["checklist"], {"ask_the_buyer": str, "signal": str, "effect": str,
                                              "win_rate": pct, "n": num},
                            {"ask_the_buyer": "Ask in the RFQ qualification call", "signal": "Signal in the data",
                             "effect": "Effect", "win_rate": "Win rate", "n": "Deals"}),
        "quality": _table(f["quality"], {"status": lambda v: status[v], "check": str, "table": str,
                                          "severity": lambda v: sev_icon[v], "failed": num,
                                          "failed_pct": lambda v: pct(v, 1), "owner": str, "why": str},
                          {"status": "Status", "check": "Check", "table": "Table", "severity": "Severity",
                           "failed": "Failed rows", "failed_pct": "Failed %", "owner": "Fix owner",
                           "why": "Why it matters"}),
        "state_cleanup": _table(f["state_cleanup"], {"state_raw": str,
                                                      "cleaned_to": lambda v: v if isinstance(v, str)
                                                      else "— not a US state (sent to review)",
                                                      "companies": num},
                                {"state_raw": "Raw value", "cleaned_to": "Cleaned to",
                                 "companies": "Companies"}, 10),
    }
    if "benchmark" in f:
        b = f["benchmark"]
        tables["benchmark"] = _table(
            b.assign(our=b.apply(lambda r: f"{r.our_test_auc:.3f} ({r.our_auc_ci_low:.2f}–"
                                           f"{r.our_auc_ci_high:.2f})", axis=1),
                     pub=b.apply(lambda r: f"{max(r.published_lr_auc_median, r.published_gbm_auc_median):.3f}"
                                           f" ± {r.published_auc_spread / 2:.2f}", axis=1)),
            {"tier": str, "missing_rate": lambda v: pct(v), "base_rate": lambda v: pct(v, 1), "our": str,
             "pub": str, "top_tier_lift": lambda v: f"{v:.2f}×", "leak_auc_inflation": lambda v: f"+{v:.3f}"},
            {"tier": "Data tier", "missing_rate": "Missing values", "base_rate": "Conversion rate",
             "our": "Our test AUC (95% CI)", "pub": "Published best baseline (median ± half-spread)",
             "top_tier_lift": "Tier-A lift", "leak_auc_inflation": "AUC from leaked column"})

    env = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
    env.filters.update(money=money, pct=pct, num=num)
    template = env.from_string((resources.files("castiq") / "templates" / "dashboard.html.j2").read_text())
    import plotly.offline

    html = template.render(m=m, figs=json.dumps(fig_json), tables=tables, plotlyjs=plotly.offline.get_plotlyjs(),
                           generated=datetime.now(UTC).strftime("%Y-%m-%d"),
                           sig_segments=m["pipeline"].get("significant_segments", []))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
