"""Command line: `castiq ingest | build | analyze | run-all | report`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from castiq.config import Paths, load_datasets_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def cmd_ingest(paths: Paths) -> None:
    from castiq import ingest

    print(f"manifest: {ingest.ingest_all(paths)}")


def cmd_build(paths: Paths) -> None:
    from castiq import warehouse

    snapshot = int(load_datasets_config(paths)["crm"].get("snapshot_day", 30))
    tables = warehouse.build(paths.raw, paths.warehouse, snapshot)
    print(f"warehouse: {paths.warehouse} ({len(tables)} tables)")


def cmd_analyze(paths: Paths, benchmark: bool) -> None:
    from castiq import analyze, docs, report

    results = analyze.run(paths, run_benchmark=benchmark)
    analyze.save(results, paths)
    out = report.build(results, paths.docs / "index.html")
    written = docs.write_all(results.metrics, results.frames, paths)
    print(f"dashboard: {out}")
    for p in written:
        print(f"doc:       {p}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="castiq", description=__doc__)
    parser.add_argument("command", choices=["ingest", "build", "analyze", "report", "run-all"])
    parser.add_argument("--root", type=Path, default=None, help="project root (default: this repo)")
    parser.add_argument("--no-benchmark", action="store_true", help="skip the 3-tier lead-scoring benchmark")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    paths = (Paths(root=args.root, config_dir=args.root / "config") if args.root else Paths()).ensure()

    start = time.time()
    if args.command in ("ingest", "run-all"):
        cmd_ingest(paths)
    if args.command in ("build", "run-all"):
        cmd_build(paths)
    if args.command in ("analyze", "report", "run-all"):
        cmd_analyze(paths, benchmark=not args.no_benchmark)
    print(f"done in {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
