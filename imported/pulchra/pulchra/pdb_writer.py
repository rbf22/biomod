def write_pdb(molecule, filepath):
    """
    Writes a Molecule object to a PDB file.
    """
    with open(filepath, 'w') as f:
        # Placeholder for writing the PDB file
        f.write("REMARK 999 REBUILT BY PULCHRA V.3.04\n")
        anum = 1
        atom_order = {"N": 0, "CA": 1, "C": 2, "O": 3}
        for res in molecule.residues:
            # Sort atoms based on a predefined order (N, CA, C, O, then others)
            print(f"Residue {res.num}: {[atom.name for atom in res.atoms]}")
            sorted_atoms = sorted(res.atoms, key=lambda atom: atom_order.get(atom.name, 4))
            print(f"Residue {res.num} sorted: {[atom.name for atom in sorted_atoms]}")
            for atom in sorted_atoms:
                f.write(
                    f"ATOM  {anum:5d} {atom.name:<4s} {res.name:<3s}  {res.num:4d}    "
                    f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}\n"
                )
                anum += 1
        f.write("TER\nEND\n")
