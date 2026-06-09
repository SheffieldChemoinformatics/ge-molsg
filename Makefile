.PHONY: test linting

test:
	pytest tests/

linting:
	ruff check src/ tests/
