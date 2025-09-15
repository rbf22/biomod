"""
This module contains functions for calculating hydrogen bond energies between residues.
"""
import numpy as np

from ..core.residues import Residue
from ..utilities.secondary_structure.data_structures import HBond
from ..config import MINIMAL_DISTANCE, MIN_HBOND_ENERGY, COUPLING_CONSTANT


def calculate_h_bond_energy(donor: Residue, acceptor: Residue):
    """Calculate the H-bond energy between two residues based on their coordinates.

    This function implements the H-bond energy calculation from the DSSP paper,
    which is a simplified electrostatic model. The energy is calculated based
    on the positions of the donor's amide group (N-H) and the acceptor's
    carbonyl group (C=O).

    The calculated energy is then used to update the h-bond lists on both the
    donor and acceptor Residue objects.

    Args:
        donor (Residue): The residue donating the hydrogen.
        acceptor (Residue): The residue accepting the hydrogen.

    Returns:
        float: The calculated H-bond energy in kcal/mol.
    """
    energy = 0.0
    # Critical fix: Only calculate energy if donor is NOT proline
    # Proline cannot donate hydrogen bonds because its nitrogen is part of a ring
    if donor.name != "PRO":
        o_atom = acceptor.atom(name="O")  # type: ignore
        c_atom = acceptor.atom(name="C")  # type: ignore
        n_atom = donor.atom(name="N")  # type: ignore
        if o_atom is None or c_atom is None or n_atom is None:
            return 0.0
        dist_ho = np.linalg.norm(donor.h_coord - np.array(o_atom.location))
        dist_hc = np.linalg.norm(donor.h_coord - np.array(c_atom.location))
        dist_nc = np.linalg.norm(np.array(n_atom.location) - np.array(c_atom.location))
        dist_no = np.linalg.norm(np.array(n_atom.location) - np.array(o_atom.location))

        if (
            dist_ho < MINIMAL_DISTANCE
            or dist_hc < MINIMAL_DISTANCE
            or dist_nc < MINIMAL_DISTANCE
            or dist_no < MINIMAL_DISTANCE
        ):
            energy = MIN_HBOND_ENERGY
        else:
            energy = float(
                COUPLING_CONSTANT / dist_ho
                - COUPLING_CONSTANT / dist_hc
                + COUPLING_CONSTANT / dist_nc
                - COUPLING_CONSTANT / dist_no
            )

        # Round to match DSSP compatibility mode
        energy = round(energy * 1000) / 1000
        energy = max(energy, MIN_HBOND_ENERGY)

    # Only update hydrogen bond arrays if energy is significant
    # This prevents weak/zero bonds from overwriting stronger ones
    # Update donor's acceptor bonds (bonds where this residue donates)
    if energy < donor.hbond_acceptor[0].energy:
        donor.hbond_acceptor[1] = donor.hbond_acceptor[0]
        donor.hbond_acceptor[0] = HBond(acceptor, energy)
    elif energy < donor.hbond_acceptor[1].energy:
        donor.hbond_acceptor[1] = HBond(acceptor, energy)

    # Update acceptor's donor bonds (bonds where this residue accepts)
    if energy < acceptor.hbond_donor[0].energy:
        acceptor.hbond_donor[1] = acceptor.hbond_donor[0]
        acceptor.hbond_donor[0] = HBond(donor, energy)
    elif energy < acceptor.hbond_donor[1].energy:
        acceptor.hbond_donor[1] = HBond(donor, energy)

    return energy


def assign_hydrogen_to_residues(residues: list[Residue]):
    """Assign amide hydrogen positions for all residues in a list.

    The position of the amide hydrogen is not present in most PDB/CIF files.
    This function estimates its position based on the geometry of the
    preceding residue's carbonyl group, which is essential for accurate
    H-bond energy calculations.

    Args:
        residues (list[Residue]): A list of Residue objects to process.
    """
    for i, residue in enumerate(residues):
        # Start with nitrogen position
        n_atom = residue.atom(name="N")  # type: ignore
        if n_atom is None:
            continue
        residue.h_coord = np.array(n_atom.location, dtype=float)
        # For non-proline residues with a previous residue
        if residue.name != "PRO":
            if i > 0:
                prev_residue = residues[i - 1]
                prev_c = prev_residue.atom(name="C")  # type: ignore
                prev_o = prev_residue.atom(name="O")  # type: ignore
                if prev_c is None or prev_o is None:
                    continue
                # Calculate CO vector and normalize it
                co_vector = np.array(prev_c.location) - np.array(prev_o.location)
                co_distance = np.linalg.norm(co_vector)
                if co_distance > 0:
                    co_unit = co_vector / co_distance
                    # Place hydrogen along the CO vector direction from nitrogen
                    residue.h_coord += co_unit
            else:  # First residue
                ca_atom = residue.atom(name="CA")  # type: ignore
                if ca_atom is not None:
                    n_ca_vector = np.array(ca_atom.location) - np.array(n_atom.location)
                    norm = np.linalg.norm(n_ca_vector)
                    if norm > 0:
                        residue.h_coord -= n_ca_vector / norm


def calculate_h_bonds(residues: list[Residue]) -> None:
    """Calculate and assign H-bonds for all residues in the structure.

    This is a main function in the DSSP pipeline. It iterates through all
    pairs of residues, calculates the H-bond energy between them, and updates
    the `hbond_acceptor` and `hbond_donor` lists on each Residue object.

    Note: This function has side effects, as it modifies the Residue objects
    in the input list.

    Args:
        residues (list[Residue]): A list of all Residue objects in the structure.
    """

    # Note: H-bond arrays are already initialized in Residue.__init__()
    # But we need to reset them to ensure clean state
    for res in residues:
        res.hbond_acceptor = [HBond(None, 0.0), HBond(None, 0.0)]
        res.hbond_donor = [HBond(None, 0.0), HBond(None, 0.0)]
        res.h_coord = None

    assign_hydrogen_to_residues(residues)

    for i, res_i in enumerate(residues):
        for j in range(i + 1, len(residues)):
            res_j = residues[j]
            ca_i = res_i.atom(name="CA")  # type: ignore
            ca_j = res_j.atom(name="CA")  # type: ignore
            if ca_i is None or ca_j is None:
                continue
            if np.linalg.norm(np.array(ca_i.location) - np.array(ca_j.location)) > 9.0:
                continue

            calculate_h_bond_energy(res_i, res_j)
            if j != i + 1:
                calculate_h_bond_energy(res_j, res_i)
