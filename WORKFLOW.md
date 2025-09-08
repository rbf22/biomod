# Biomod Workflow

This document describes the **end-to-end modeling process** in Biomod.  

---

## Stages

### 1. Input Parsing
- PDB/mmCIF → Atomium-based parser  
- SMILES → PySMILES-based parser  
- FASTA → lightweight sequence parser  

### 2. Topology Building
- Construct atoms, bonds, residues, chains.  
- Supports proteins, DNA, RNA, small molecules.  

### 3. Reconstruction
- Add missing heavy atoms (Pulchra-based).  
- Add hydrogens (Pulchra templates).  

### 4. Secondary Structure
- Assign α-helix, β-sheet, turns (DSSPy).  

### 5. Energy Definition
- Build modular force field:  
  - Bonds  
  - Angles  
  - Torsions  
  - Lennard-Jones  
  - Coulombic  

### 6. Optimization
- Stage 1: heavy atoms only.  
- Stage 2: add hydrogens.  
- Stage 3: full system refinement.  
- Algorithms: steepest descent, conjugate gradient.  

### 7. Sampling
- Backbone torsions  
- Sidechain rotamers  
- Random displacements  

### 8. Output
- Write to PDB or mmCIF.  

---

## Example Pipeline

```
parse → build topology → add missing atoms → assign secondary structure →
define energy → optimize (heavy → hydrogens → full) → sample → export
```
