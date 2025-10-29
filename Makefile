.PHONY: test smoke ci

test:
	uv run pytest -q

smoke:
	bash scripts/smoke.sh

ci: test smoke
