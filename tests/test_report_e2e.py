from __future__ import annotations

import json
import re

import plotly.graph_objects as go

from castiq import cli, report


def test_money_and_pct_formatting() -> None:
    assert report.money(21_743_548_750) == "$21.7B"
    assert report.money(64_976_858) == "$65.0M"
    assert report.money(7_500) == "$7.5K"
    assert report.money(12) == "$12"
    assert report.money(None) == "–"
    assert report.pct(0.8099) == "81%"
    assert report.pct(0.48022, 1) == "48.0%"


def test_detokenize_replaces_every_placeholder() -> None:
    fig = go.Figure(go.Bar(x=[1], y=[1], marker={"color": report.T["series1"]}))
    js = report.detokenize(fig.to_json())
    assert '"@series1"' in js
    assert report.T["series1"] not in js


def test_end_to_end_analyze_writes_dashboard_and_docs(project) -> None:  # type: ignore[no-untyped-def]
    assert cli.main(["analyze", "--root", str(project.root)]) == 0

    html = (project.docs / "index.html").read_text()
    figs = json.loads(re.search(r"const FIGS = (\{.*?\});\n", html, re.S).group(1))  # type: ignore[union-attr]
    for fid in figs:
        assert f'id="fig-{fid}"' in html, f"chart {fid} has no container"
    assert not re.search(r'"#c0[0-9a-f]{4}"', html), "an untranslated colour placeholder leaked into the page"
    for sec in ("market", "territories", "winloss", "leads", "pipeline", "forecast", "quality", "method"):
        assert f'<section id="{sec}">' in html
    assert "fig-forecast" in html and 'id="forecast"' in html  # section and chart ids no longer collide

    metrics = json.loads((project.docs / "metrics.json").read_text())
    for key in ("market", "territories", "pipeline", "forecast", "lead_scoring", "win_loss", "quality", "reps"):
        assert key in metrics
    assert metrics["territories"]["cv_after"] <= metrics["territories"]["cv_before"]

    readme = (project.root / "README.md").read_text()
    assert report.num(metrics["market"]["in_market_accounts"]) in readme
    assert report.money(metrics["forecast"]["forecast_stage_default"]) in readme
    bullets = (project.docs / "career" / "RESUME_BULLETS.md").read_text()
    assert report.pct(metrics["forecast"]["stage_default_overstatement"]) in bullets
    for f in ("CASE_STUDY.md", "DATA_SOURCES.md", "career/LINKEDIN_POST.md", "career/INTERVIEW_TALK_TRACK.md",
              "career/JOB_SEARCH.md", "data/top_accounts.csv", "data/quality.csv", "data/benchmark.csv"):
        assert (project.docs / f).exists(), f
