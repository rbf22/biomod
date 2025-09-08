# Contributing Guidelines

Thank you for your interest in contributing to **Biomod**!  
This guide applies to both human developers and AI coding agents.  

---

## Principles

- **Clarity over cleverness**: write code that is easy to understand.  
- **Composable design**: keep functions small, reusable, and testable.  
- **Minimal dependencies**: only NumPy (and optional PySMILES).  
- **Stable interfaces**: never break core data contracts.  

---

## Code Style

- Follow **PEP8** conventions.  
- Use **type hints** for all function signatures.  
- Add **docstrings** with usage examples.  
- Prefer **NumPy vectorization** over Python loops.  

---

## Data Contracts

- `Structure.coords` → `(N,3)` NumPy array of float32  
- `EnergyTerm.energy(coords)` → `float`  
- `EnergyTerm.gradient(coords)` → `(N,3)` NumPy array  
- `Optimizer.run(coords, steps)` → `(N,3)` updated coords  
- `Sampler.sample(n)` → generator of `(N,3)` coords  

**Do not break these contracts.**

---

## Tests

- Every new function/class must include a **unit test**.  
- Tests should be small and run quickly.  
- Use Python’s built-in `unittest` or `pytest`.  

---

## AI Agent Guidance

- ✅ Do: refactor legacy code into clean, modular components.  
- ✅ Do: preserve functionality from Atomium, Pulchra, DSSPy, Vitra, PySMILES.  
- ✅ Do: enforce consistent input/output signatures.  
- ❌ Don’t: introduce large new dependencies.  
- ❌ Don’t: create hidden global state.  
- ❌ Don’t: bypass tests or remove documentation.  

---

## Submitting Changes

1. Fork the repository.  
2. Create a feature branch.  
3. Add code + tests.  
4. Run test suite.  
5. Open a pull request.  

---

Thank you for helping build **Biomod**!
