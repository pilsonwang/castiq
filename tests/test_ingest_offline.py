"""ingest_all() end to end with the Hugging Face download replaced by local fixture files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from castiq import ingest
from castiq.config import Paths
from tests.conftest import _crm_tables, build_fixture_project, make_deals, make_task_split


@pytest.fixture()
def fake_hub(tmp_path: Path, companies_csv_gz: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Write fixture files where a Hub download would land, and route _download to them."""
    rng = np.random.default_rng(3)
    hub = tmp_path / "hub"
    files: dict[str, Path] = {}
    for tier in ("intro", "intermediate", "advanced"):
        for split in ("train", "valid", "test"):
            p = hub / tier / "tasks" / "converted_within_90_days" / f"{split}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            make_task_split(rng, 50).to_parquet(p, index=False)
            files[f"{tier}/tasks/converted_within_90_days/{split}.parquet"] = p
        m = hub / tier / "metrics.json"
        m.write_text(json.dumps({"medians": {}}))
        files[f"{tier}/metrics.json"] = m
    for name, df in _crm_tables(rng).items():
        p = hub / "intermediate" / "tables" / f"{name}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        files[f"intermediate/tables/{name}.parquet"] = p
    deals = hub / "deals.csv"
    make_deals(rng, 60).to_csv(deals, sep=";", index=False)
    files["Salvirt_B2B_ML_dataset_HF.csv"] = deals
    files["companies-2023-q4-sm.csv.gz"] = companies_csv_gz

    def fake_download(repo_id: str, filename: str, revision: str, cache_dir: Path) -> Path:
        assert len(revision) == 40, "every download must be pinned to a full commit hash"
        return files[filename]

    monkeypatch.setattr(ingest, "_download", fake_download)
    return files


def test_ingest_all_writes_hashed_relative_manifest(tmp_path: Path, fake_hub: dict[str, Path]) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    paths: Paths = build_fixture_project(root)
    manifest = json.loads(ingest.ingest_all(paths).read_text())

    assert set(manifest) == {"crm", "deals", "companies"}
    crm = manifest["crm"]
    assert len(crm) == 7 + 3 + 3  # 7 tables + 3 task splits + 1 benchmark tier (fixture config) x 3 splits
    for e in crm + manifest["deals"]:
        assert not Path(e["local_path"]).is_absolute(), "manifest must be machine-independent"
        local = root / e["local_path"]
        assert hashlib.sha256(local.read_bytes()).hexdigest() == e["sha256"]
        assert e["url"].startswith("https://huggingface.co/datasets/")
    assert manifest["deals"][0]["rows"] == 60
    assert manifest["companies"][0]["rows"] == 401
    assert (paths.raw / "crm" / "benchmark" / "intro_published_metrics.json").exists()
    stats = json.loads((paths.raw / "companies" / "country_stats.json").read_text())
    assert stats["in_market_companies"] == 400
