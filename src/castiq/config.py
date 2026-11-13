"""Load project configuration and resolve file-system paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """All locations the pipeline reads from or writes to.

    Everything hangs off one root, so tests can point the whole pipeline at a temporary folder.
    """

    root: Path = PROJECT_ROOT
    config_dir: Path = field(default=PROJECT_ROOT / "config")

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def hf_cache(self) -> Path:
        return self.data / "hf_cache"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def warehouse(self) -> Path:
        return self.data / "castiq.duckdb"

    @property
    def outputs(self) -> Path:
        return self.data / "outputs"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    def ensure(self) -> Paths:
        for p in (self.data, self.raw, self.outputs, self.docs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_datasets_config(paths: Paths) -> dict[str, Any]:
    return load_yaml(paths.config_dir / "datasets.yaml")


def load_icp_config(paths: Paths) -> dict[str, Any]:
    cfg = load_yaml(paths.config_dir / "icp.yaml")
    validate_icp_config(cfg)
    return cfg


def load_revops_config(paths: Paths) -> dict[str, Any]:
    cfg = load_yaml(paths.config_dir / "revops.yaml")
    shares = cfg["lead_scoring"]["tier_shares"]
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        raise ValueError("revops.yaml: lead_scoring.tier_shares must add up to 1.0")
    for stage, p in cfg["forecast"]["default_stage_probability"].items():
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"revops.yaml: probability for {stage} must be between 0 and 1")
    return cfg


def validate_icp_config(cfg: dict[str, Any]) -> None:
    """Fail fast on a bad edit by a sales leader, with a message they can act on."""
    weights = cfg["weights"]
    total = sum(float(v) for v in weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"icp.yaml: weights must add up to 1.0 (now {total:.3f})")
    for section in ("industry_fit", "size_fit"):
        for key, value in cfg[section].items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"icp.yaml: {section}.{key} must be between 0 and 1 (now {value})")
    tiers = cfg["tiers"]
    if not tiers["A"] > tiers["B"] > tiers["C"]:
        raise ValueError("icp.yaml: tier thresholds must be A > B > C")
