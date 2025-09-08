.PHONY: poetry-install dev-install build run-tests run-linter run-linter-fix run-pylint run-mypy run-deptry clean

poetry-install: dev-install
	@echo "Installation complete. Please activate your poetry shell with 'poetry shell'"

# Install Python dependencies without building the C++ code
dev-install:
	poetry install

build:
	@mkdir -p c_legacy/bin
	@echo "Compiling C code..."
	(cd c_legacy/src && cc -O3 -o ../bin/pulchra pulchra.c pulchra_data.c -lm)

run-tests: build
	poetry run pytest -s tests/

run-linter:
	poetry run ruff check .

run-linter-fix:
	poetry run ruff check . --fix

run-pylint:
	poetry run pylint .

run-mypy:
	poetry run mypy --explicit-package-bases .

run-deptry:
	poetry run deptry .

clean:
	@echo "Cleaning up..."
	@rm -rf c_legacy/bin
	@rm -f c_legacy/examples/model.rebuilt.pdb
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
