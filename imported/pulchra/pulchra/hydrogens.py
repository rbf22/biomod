import numpy as np

def place_atom(p1, p2, p3, bond, angle, dihedral):
    """
    Places an atom based on three reference atoms and ideal geometry.

    Args:
        p1, p2, p3: Coordinates of the three reference atoms.
        bond: Bond length to the new atom.
        angle: Bond angle in radians.
        dihedral: Dihedral angle in radians.

    Returns:
        The coordinates of the new atom.
    """
    a = p2 - p1
    b = p3 - p2

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    n = np.cross(a, b)
    n /= np.linalg.norm(n)

    c = np.cross(n, b)

    x = bond * np.cos(np.pi - angle)
    y = bond * np.sin(np.pi - angle) * np.cos(dihedral)
    z = bond * np.sin(np.pi - angle) * np.sin(dihedral)

    d = -b * x + c * y + n * z

    return p3 + d

def get_hydrogen_positions(residue_name, residue_heavy_atoms, prev_c=None):
    """
    Calculates the positions of hydrogen atoms for a given residue.

    Args:
        residue_name: The name of the residue (e.g., "SER").
        residue_heavy_atoms: A dictionary of heavy atom names and their coordinates.
        prev_c: The coordinates of the C atom of the previous residue.

    Returns:
        A dictionary of hydrogen atom names and their coordinates.
    """
    hydrogens = {}

    # Amide hydrogen (H)
    if 'N' in residue_heavy_atoms and 'CA' in residue_heavy_atoms and prev_c is not None:
        n_coord = residue_heavy_atoms['N']
        ca_coord = residue_heavy_atoms['CA']

        # Ideal geometry for amide hydrogen
        bond_length = 1.01
        bond_angle = np.deg2rad(120.0)
        dihedral_angle = np.deg2rad(180.0)  # For a trans peptide bond

        h_coord = place_atom(ca_coord, prev_c, n_coord, bond_length, bond_angle, dihedral_angle)
        hydrogens['H'] = h_coord

    # WARNING: The following are approximate values and should be replaced with more accurate ones.

    # Serine (SER) hydroxyl hydrogen (HG)
    if residue_name == "SER" and 'CA' in residue_heavy_atoms and 'CB' in residue_heavy_atoms and 'OG' in residue_heavy_atoms:
        ca_coord = residue_heavy_atoms['CA']
        cb_coord = residue_heavy_atoms['CB']
        og_coord = residue_heavy_atoms['OG']

        bond_length = 0.96
        bond_angle = np.deg2rad(109.5)
        dihedral_angle = np.deg2rad(180.0)

        hg_coord = place_atom(ca_coord, cb_coord, og_coord, bond_length, bond_angle, dihedral_angle)
        hydrogens['HG'] = hg_coord

    # Threonine (THR) hydroxyl hydrogen (HG1)
    if residue_name == "THR" and 'CA' in residue_heavy_atoms and 'CB' in residue_heavy_atoms and 'OG1' in residue_heavy_atoms:
        ca_coord = residue_heavy_atoms['CA']
        cb_coord = residue_heavy_atoms['CB']
        og1_coord = residue_heavy_atoms['OG1']

        bond_length = 0.96
        bond_angle = np.deg2rad(109.5)
        dihedral_angle = np.deg2rad(180.0)

        hg1_coord = place_atom(ca_coord, cb_coord, og1_coord, bond_length, bond_angle, dihedral_angle)
        hydrogens['HG1'] = hg1_coord

    # Tyrosine (TYR) hydroxyl hydrogen (HH)
    if residue_name == "TYR" and 'CE1' in residue_heavy_atoms and 'CZ' in residue_heavy_atoms and 'OH' in residue_heavy_atoms:
        ce1_coord = residue_heavy_atoms['CE1']
        cz_coord = residue_heavy_atoms['CZ']
        oh_coord = residue_heavy_atoms['OH']

        bond_length = 0.96
        bond_angle = np.deg2rad(120.0)
        dihedral_angle = np.deg2rad(180.0)

        hh_coord = place_atom(ce1_coord, cz_coord, oh_coord, bond_length, bond_angle, dihedral_angle)
        hydrogens['HH'] = hh_coord

    return hydrogens
