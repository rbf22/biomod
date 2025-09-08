# Biomod Roadmap

Planned development stages for the project.  

---

## Phase 1: Core Integration (MVP)
- Merge Atomium, Pulchra, DSSPy, Vitra, PySMILES into one repo.  
- Establish unified `Structure` class.  
- Implement core energy terms + minimizers.  
- Parse PDB, mmCIF, SMILES, FASTA.  

## Phase 2: Reconstruction & Secondary Structure
- Full support for missing atom reconstruction.  
- Add hydrogen placement.  
- DSSP-like secondary structure assignments.  

## Phase 3: Sampling
- Backbone torsion sampling.  
- Sidechain rotamer sampling.  
- Monte Carlo random displacement.  

## Phase 4: Protein/RNA/DNA Design
- Mutations and sequence redesign.  
- Evaluate stability/energy.  
- Simple optimization for target features.  

## Phase 5: Extensibility
- Hooks for new energy terms.  
- Support for statistical potentials.  
- Optional ML-based energy functions.  

---

*This roadmap is iterative and will evolve as the project matures.*
