.PHONY: install data build analyze all test test-network lint typecheck check screenshot clean

install:        ## create the virtualenv and install CastIQ with dev tools
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

data:           ## download the pinned Hugging Face datasets (~630 MB, cached)
	castiq ingest

build:          ## build the DuckDB warehouse from data/raw
	castiq build

analyze:        ## run every analysis, write docs/index.html and the generated docs
	castiq analyze

all:            ## ingest + build + analyze
	castiq run-all

test:           ## offline test suite (fixtures only)
	pytest --cov=castiq --cov-report=term-missing

test-network:   ## also run the test that downloads from Hugging Face
	CASTIQ_NETWORK_TESTS=1 pytest -m network

lint:
	ruff check src tests

typecheck:
	mypy

check: lint typecheck test

screenshot:     ## refresh docs/assets/dashboard.png (needs Node + Playwright)
	node scripts/screenshot.js docs/index.html docs/assets/dashboard.png

clean:
	rm -rf data/castiq.duckdb data/outputs .pytest_cache .mypy_cache .ruff_cache
