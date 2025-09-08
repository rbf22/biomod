# Pulchra (Python Port)

This project is a Python-based version of the Pulchra tool for reconstructing and refining protein structures from reduced models. The goal is to provide a version with very limited dependencies for fixing and adding missing atoms to protein structures.

This repository contains the original C code for reference and a Python test suite to verify the behavior of the compiled C code.

## Building and Testing

This project uses `poetry` for dependency management.

To build the C code and run the Python tests, you can use the provided `Makefile`:

1.  **Install dependencies:**
    ```bash
    poetry install
    ```

2.  **Build the C code:**
    ```bash
    make build
    ```

3.  **Run the tests:**
    ```bash
    make run-tests
    ```
