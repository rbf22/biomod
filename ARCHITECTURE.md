# Biomod Architecture

This document describes the internal design of Biomod.  
It serves as a **map for AI agents and human contributors** working on the codebase.

---

## Core Principles

1. **Unified `Structure` class**  
   All biomolecules (proteins, DNA, RNA, small molecules) share the same data model.  
2. **Composable energy terms**  
   Energies are sums of modular `EnergyTerm` objects (`bonds + angles + LJ`).  
3. **Expressive dual API**  
   - High-level: `struct.minimize(steps=500)`  
   - Low-level: directly manipulate `coords` NumPy arrays.  
4. **Minimal dependencies**  
   Only NumPy (and optional PySMILES).  
5. **Staged workflows**  
   Optimize heavy atoms → add hydrogens → refine full structure.  

---

## Module Layout

```
biomod/
│
├── core/
│   ├── Structure      # atoms, bonds, residues, chains, coordinates
│   ├── Atom           # element, coords_ref, charge, parameters
│   ├── Bond           # atom1, atom2, order, parameters
│   ├── Residue        # residue type, atoms
│   ├── Chain          # collection of residues
│   └── Topology       # builders for protein, DNA, RNA, SMILES
│
├── energy/
│   ├── EnergyTerm     # abstract base class
│   ├── Bonds          # bond stretching
│   ├── Angles         # bond angles
│   ├── Torsions       # dihedral terms
│   ├── LennardJones   # van der Waals
│   ├── Coulomb        # electrostatics
│   └── Energy         # composition of terms
│
├── optimize/
│   ├── Optimizer      # abstract base
│   ├── SteepestDescent
│   └── ConjugateGradient
│
├── sample/
│   ├── Sampler        # abstract base
│   ├── BackboneSampler
│   ├── RotamerSampler
│   └── RandomDisplacement
│
├── io/
│   ├── pdb.py         # PDB parser/writer (Atomium-based)
│   ├── mmcif.py       # mmCIF parser/writer (Atomium-based)
│   ├── smiles.py      # SMILES parser (PySMILES-based)
│   └── fasta.py       # FASTA parser
│
└── utilities/
    ├── reconstruction.py  # Pulchra: add_missing_atoms(), add_hydrogens()
    ├── secondary.py       # DSSPy: assign_secondary_structure()
    └── misc.py
```

---

## Mapping Legacy Repos

- **Atomium** → `core.Structure`, `io.pdb`, `io.mmcif`  
- **PySMILES** → `io.smiles`, feeding into `core.Topology`  
- **Pulchra** → `utilities.reconstruction`  
- **DSSPy** → `utilities.secondary`  
- **Vitra** → `energy.*` + `optimize.*`  

---

## Data Contracts

To ensure consistency, all modules follow strict contracts:

- `Structure.coords` → `(N,3)` NumPy array of float32  
- `EnergyTerm.energy(coords)` → `float`  
- `EnergyTerm.gradient(coords)` → `(N,3)` NumPy array  
- `Optimizer.run(coords, steps)` → `(N,3)` updated coords  
- `Sampler.sample(n)` → generator of `(N,3)` coords  

These contracts **must never be broken**, even if internals change.

---

## Extensibility Hooks

- Add a new **EnergyTerm** by subclassing `EnergyTerm` and implementing `energy()` + `gradient()`.  
- Add a new **Sampler** by subclassing `Sampler` and implementing `sample()`.  
- Add a new **Topology builder** in `core.Topology`.  
- Add a new **file parser/writer** in `io/`.  

---

## Pipeline Overview

1. **Parse input** → Atomium / PySMILES / FASTA  
2. **Build topology** → `Topology`  
3. **Reconstruct missing atoms** → Pulchra-based utilities  
4. **Assign secondary structure** → DSSPy-based  
5. **Build energy function** → Vitra-based terms  
6. **Optimize** → heavy → hydrogens → full system  
7. **Sample** → torsions / rotamers / random  
8. **Export** → PDB / mmCIF  

---
