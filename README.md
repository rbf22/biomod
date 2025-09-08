# Biomod: A Unified Toolkit for Molecular Modeling in Python

**Biomod** is a lightweight, composable Python library for **proteins, nucleic acids, and small molecules**.  
It merges and refactors functionality from several specialized packages into a single cohesive framework:

- Atomium: parsing and managing PDB/mmCIF structures  
- Pulchra: reconstruction of missing atoms  
- DSSPy: secondary structure assignment  
- Vitra: energy evaluation and minimization  
- PySMILES: parsing SMILES strings for small molecules  

The result is a **CHARMM-like experience in pure Python** — with both high-level commands and low-level access to coordinates, energies, and topology.  

---

## Features

- Parse **PDB**, **mmCIF**, **SMILES**, and **FASTA**  
- Unified `Structure` class for proteins, DNA/RNA, and ligands  
- Add missing atoms and hydrogens (Pulchra-based)  
- Assign secondary structure (DSSP-like)  
- Define modular energy functions (bonded + nonbonded)  
- Optimize with steepest descent or conjugate gradient  
- Sample conformational space (torsions, rotamers, etc.)  
- Minimal dependencies (`numpy`, `pysmiles` optional)  

---

## Quickstart

```python
from biomod import Structure, Topology, Energy, Optimizer

# Build from protein sequence
seq = "ACDEFGHIK"
struct = Structure.from_topology(Topology.protein_from_sequence(seq))

# Repair structure (add missing heavy atoms)
struct = struct.add_missing_atoms(mode="heavy")

# Define energy function
E = Energy.bonds() + Energy.angles() + Energy.torsions() + Energy.lj() + Energy.coulomb()

# Minimize heavy atoms
coords = Optimizer.steepest_descent(E, mask="heavy").run(struct.coords, steps=500)

# Add hydrogens and finalize
struct = struct.add_hydrogens()
coords = Optimizer.conjugate_gradient(E).run(coords, steps=200)

struct.update_coords(coords)
struct.write("final.pdb")
```

---

## Project Layout

- `core/` → structure, topology, atoms, residues  
- `energy/` → energy terms (bonds, angles, LJ, Coulomb)  
- `optimize/` → minimization algorithms  
- `sample/` → conformational sampling tools  
- `io/` → parsers and writers for PDB, mmCIF, SMILES, FASTA  
- `utilities/` → reconstruction, secondary structure, misc  

---

## Dependencies

- Python 3.9+  
- NumPy  
- PySMILES (optional, for SMILES parsing)  

No heavy external dependencies (e.g. OpenMM, RDKit, Rosetta).  

---

## License

MIT License (see `LICENSE`).  

---

## Contributing

Please see `CONTRIBUTING.md` for guidelines.  
Our goal is **clarity, composability, and minimalism** — keep functions small, well-documented, and NumPy-friendly.


*This document should be updated as the code evolves.  
Keep the interfaces stable, so that new features remain composable.*
