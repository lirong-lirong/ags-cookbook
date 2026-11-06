SHELL := /bin/bash
.PHONY: setup build build2 push push2 run run2 verify verify2 update update2 clean check

setup:
	uv sync
	@test -f .env || cp .env.example .env

build:
	uv run python scripts/demo.py build --flavor osworld1
build2:
	uv run python scripts/demo.py build --flavor osworld2
push:
	uv run python scripts/demo.py push --flavor osworld1
push2:
	uv run python scripts/demo.py push --flavor osworld2
run:
	uv run python scripts/demo.py run --flavor osworld1
run2:
	uv run python scripts/demo.py run --flavor osworld2
verify:
	uv run python scripts/demo.py verify --flavor osworld1
verify2:
	uv run python scripts/demo.py verify --flavor osworld2
update:
	uv run python scripts/demo.py update --flavor osworld1
update2:
	uv run python scripts/demo.py update --flavor osworld2
clean:
	uv run python scripts/demo.py clean
check:
	uv run pytest -q tests
