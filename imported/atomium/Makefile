.PHONY: install test lint lint-fix clean

install:
	poetry install

test:
	poetry run pytest --cov=. --cov-report=term-missing

lint:
	poetry run ruff check .

lint-fix:
	poetry run ruff check . --fix

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f .coverage
