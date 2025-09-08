# Agent Instructions for Pulchra (Python Port)

This file provides instructions for AI agents working with this repository.

## Project Goal

The primary goal of this project is to create a Python-based version of the Pulchra tool. Pulchra is used for reconstructing and refining protein structures from reduced models. The Python version aims to have very limited dependencies.

## Repository Structure

- `c_legacy/`: Contains the original C source code of Pulchra. This is kept for reference and for comparison during development of the Python version.
- `tests/`: Contains the Python test suite. The tests are designed to run the compiled C executable and verify its output against known "golden" results.
- `pyproject.toml`: Defines the Python project, its dependencies, and build system (`poetry`).
- `Makefile`: Provides convenient commands for common tasks like building the C code, running tests, and linting.

## Development Workflow

### 1. Building the C code

The C code must be compiled before running the tests. The `Makefile` provides a `build` target for this.

```bash
make build
```

This command compiles the C source files in `c_legacy/src` and places the executable at `c_legacy/bin/pulchra`.

### 2. Running Tests

The test suite is located in the `tests/` directory and uses `pytest`. The tests execute the compiled `pulchra` binary with various flags and compare its output to pre-computed "golden" output files located in `tests/golden_output/`.

To run the tests, use the `Makefile`:

```bash
make run-tests
```

This command will first ensure the C code is built (as `run-tests` depends on the `build` target) and then execute `pytest`.

### 3. Linting

The project uses `ruff`, `pylint`, and `mypy` for linting. You can run the linters using the `Makefile`:

```bash
make run-linter
make run-pylint
make run-mypy
```

### 4. Dependencies

Python dependencies are managed with `poetry`. If you need to add a new dependency, add it to `pyproject.toml` and run `poetry install`.
