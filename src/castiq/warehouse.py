"""Build the DuckDB warehouse by running the SQL models in order (a small, dependency-free dbt)."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


def sql_models() -> list[tuple[str, str]]:
    """Return (name, sql) for every model file, in run order (file names start with 01_, 02_, ...)."""
    pkg = resources.files("castiq") / "sql"
    files = sorted((f for f in pkg.iterdir() if f.name.endswith(".sql")), key=lambda f: f.name)
    return [(f.name, f.read_text()) for f in files]


def render(sql: str, params: dict[str, str]) -> str:
    for key, value in params.items():
        sql = sql.replace("{" + key + "}", value)
    return sql


def build(raw_dir: Path, db_path: Path, snapshot_day: int = 30) -> list[str]:
    """Run every model against raw_dir and write the warehouse to db_path. Returns the tables built."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    params = {"raw": raw_dir.as_posix(), "snapshot_day": str(int(snapshot_day))}
    con = duckdb.connect(str(db_path))
    try:
        for name, sql in sql_models():
            log.info("running %s", name)
            con.execute(render(sql, params))
        tables = [r[0] for r in con.execute("select table_name from information_schema.tables").fetchall()]
    finally:
        con.close()
    return sorted(tables)


def read(db_path: Path, table_or_sql: str) -> pd.DataFrame:
    """Read a table (by name) or the result of a SELECT statement into pandas."""
    q = table_or_sql if " " in table_or_sql.strip() else f"select * from {table_or_sql}"
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(q).df()
    finally:
        con.close()
