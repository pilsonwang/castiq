"""Download the pinned Hugging Face datasets into data/raw.

Each file is fetched at an exact commit, hashed, and listed in data/raw/manifest.json, so any number in
the report can be traced back to a byte-identical source file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from castiq.config import Paths, load_datasets_config, load_icp_config
from castiq.geo import normalize_state

log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(repo_id: str, filename: str, revision: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download  # imported here so unit tests never need the network

    return Path(
        hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision, repo_type="dataset", cache_dir=cache_dir
        )
    )


def _entry(repo_id: str, revision: str, filename: str, local: Path, rows: int) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "revision": revision,
        "file": filename,
        "url": f"https://huggingface.co/datasets/{repo_id}/blob/{revision}/{filename}",
        "local_path": str(local),
        "rows": int(rows),
        "sha256": _sha256(local),
    }


def fetch_crm(paths: Paths, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out_dir = paths.raw / "crm"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    tier, task = cfg["tier"], cfg["task"]
    files = {t: f"{tier}/tables/{t}.parquet" for t in cfg["tables"]}
    files |= {f"task_{s}": f"{tier}/tasks/{task}/{f}.parquet" for s, f in cfg["splits"].items()}
    for name, remote in files.items():
        src = _download(cfg["repo_id"], remote, cfg["revision"], paths.hf_cache)
        dst = out_dir / f"{name}.parquet"
        shutil.copyfile(src, dst)
        rows = len(pd.read_parquet(dst))
        entries.append(_entry(cfg["repo_id"], cfg["revision"], remote, dst, rows))
        log.info("crm %-18s %6d rows", name, rows)
    # Task splits + published baseline metrics for every tier, for the benchmark table.
    bench_dir = out_dir / "benchmark"
    bench_dir.mkdir(exist_ok=True)
    for bt in cfg.get("benchmark_tiers", []):
        for s, f in cfg["splits"].items():
            remote = f"{bt}/tasks/{task}/{f}.parquet"
            src = _download(cfg["repo_id"], remote, cfg["revision"], paths.hf_cache)
            dst = bench_dir / f"{bt}_{s}.parquet"
            shutil.copyfile(src, dst)
            entries.append(_entry(cfg["repo_id"], cfg["revision"], remote, dst, len(pd.read_parquet(dst))))
        src = _download(cfg["repo_id"], f"{bt}/metrics.json", cfg["revision"], paths.hf_cache)
        shutil.copyfile(src, bench_dir / f"{bt}_published_metrics.json")
    return entries


def fetch_deals(paths: Paths, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out_dir = paths.raw / "deals"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = _download(cfg["repo_id"], cfg["file"], cfg["revision"], paths.hf_cache)
    df = pd.read_csv(src, sep=cfg.get("sep", ","))
    dst = out_dir / "deals.parquet"
    df.to_parquet(dst, index=False)
    log.info("deals %d rows", len(df))
    return [_entry(cfg["repo_id"], cfg["revision"], cfg["file"], dst, len(df))]


def filter_companies(csv_gz: Path, country_code: str, industries: list[str]) -> tuple[pd.DataFrame, dict]:
    """Stream the 17M-row file with DuckDB and keep in-market companies for one country.

    Returns the filtered frame plus country-level counts used for context and data-quality checks.
    """
    con = duckdb.connect()
    con.execute("set enable_progress_bar = false")
    con.execute(
        f"""
        create table country as
        select handle, name, website, lower(trim(industry)) as industry, size, type,
               try_cast(founded as integer) as founded, city, state, country_code
        from read_csv('{csv_gz.as_posix()}', header=true, all_varchar=true,
                      ignore_errors=true, parallel=false)
        where country_code = ?
        """,
        [country_code],
    )
    total, with_industry, with_size, with_state, with_web = con.execute(
        "select count(*), count(industry), count(size), count(state), count(website) from country"
    ).fetchall()[0]
    con.register("wanted", pd.DataFrame({"industry": [i.lower() for i in industries]}))
    df = con.execute("select c.* from country c join wanted w using (industry)").df()
    con.close()
    df["state_raw"] = df["state"]
    df["state"] = df["state_raw"].map(normalize_state)
    stats = {
        "country_companies": int(total),
        "country_with_industry": int(with_industry),
        "country_with_size": int(with_size),
        "country_with_state": int(with_state),
        "country_with_website": int(with_web),
        "in_market_companies": len(df),
    }
    return df, stats


def fetch_companies(paths: Paths, cfg: dict[str, Any], industries: list[str]) -> list[dict[str, Any]]:
    out_dir = paths.raw / "companies"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = _download(cfg["repo_id"], cfg["file"], cfg["revision"], paths.hf_cache)
    df, stats = filter_companies(src, cfg["country_code"], industries)
    dst = out_dir / "companies.parquet"
    df.to_parquet(dst, index=False)
    (out_dir / "country_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("companies: %d in-market of %d %s companies", len(df), stats["country_companies"],
             cfg["country_code"])
    entry = _entry(cfg["repo_id"], cfg["revision"], cfg["file"], src, stats["country_companies"])
    entry["rows_note"] = f"rows = {cfg['country_code']} companies in source; {len(df)} kept as in-market"
    return [entry]


def ingest_all(paths: Paths) -> Path:
    paths.ensure()
    ds = load_datasets_config(paths)
    icp = load_icp_config(paths)
    manifest = {
        "crm": fetch_crm(paths, ds["crm"]),
        "deals": fetch_deals(paths, ds["deals"]),
        "companies": fetch_companies(paths, ds["companies"], list(icp["industry_fit"].keys())),
    }
    # Store paths relative to the project, so the manifest is the same on every machine.
    for entries in manifest.values():
        for e in entries:
            local = Path(e["local_path"])
            if local.is_relative_to(paths.root):
                e["local_path"] = local.relative_to(paths.root).as_posix()
    out = paths.raw / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out
