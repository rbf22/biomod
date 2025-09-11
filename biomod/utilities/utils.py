import sys
import numpy as np

from biomod.config import PDB_ATOM_LINE, PDB_TER_LINE, PDB_END_LINE
from biomod.core.residues import Residue
from biomod.core.atoms import Atom
from biomod.core.base import Chain, Model, Structure
from biomod.io.pdb_parser import PDBParser


def get_model_pdb_lines(model, model_id=0):
    """
    """
    lines = []
    for chain in model.chains:
        for residue in chain.residues:
            for atom in residue.atoms:
                lines.append(get_atom_pdb_line(atom))
        lines.append(get_ter_pdb_line(residue, chain))
    lines.append(PDB_END_LINE)
    return lines


def get_atom_pdb_line(atom):
    """
    """
    return PDB_ATOM_LINE % (
        atom.atom_id,
        atom.atom_name,
        atom.res_name,
        atom.chain_id,
        atom.res_id,
        atom.coords[0],
        atom.coords[1],
        atom.coords[2],
        atom.occupancy,
        atom.b_factor,
        atom.element
    )


def get_ter_pdb_line(residue, chain):
    """
    """
    return PDB_TER_LINE % (
        residue.atoms[-1].atom_id + 1,
        residue.res_name,
        chain.chain_id,
        residue.res_id
    )


def get_chain_from_res_list(residues, chain_id='A'):
    """
    """
    chain = Chain(chain_id)
    chain.residues = residues
    return chain


def get_model_from_chain_list(chains, model_id=0):
    """
    """
    model = Model(model_id)
    model.chains = chains
    return model


def get_structure_from_model(model):
    """
    """
    structure = Structure('str')
    structure.models.append(model)
    return structure


def get_structure_from_pdb(path):
    """
    """
    parser = PDBParser(path)
    return parser.get_structure()


def get_res_list_from_pdb(path, chain_id='A'):
    """
    """
    parser = PDBParser(path)
    return parser.get_residues(chain_id=chain_id)


def get_res_from_pdb(path, res_id, chain_id='A'):
    """
    """
    parser = PDBParser(path)
    return parser.get_residue(res_id, chain_id)


def get_ideal_bond_lengths(atom1, atom2):
    """
    Returns ideal bond length between two atoms.

    Args:
        atom1 (Atom): first atom.
        atom2 (Atom): second atom.

    Returns:
        A float, the ideal bond length.
    """
    bond_lengths = {
        frozenset(['C', 'C']): 1.54,
        frozenset(['C', 'N']): 1.47,
        frozenset(['C', 'O']): 1.43,
        frozenset(['C', 'H']): 1.09,
        frozenset(['N', 'H']): 1.01,
        frozenset(['O', 'H']): 0.96
    }
    return bond_lengths[frozenset([atom1.element, atom2.element])]

def get_ideal_bond_angle(atom1, atom2, atom3):
    """
    Returns ideal bond angle between three atoms (atom2 is central).

    Args:
        atom1 (Atom): first atom.
        atom2 (Atom): second (central) atom.
        atom3 (Atom): third atom.

    Returns:
        A float, the ideal bond angle in degrees.
    """
    bond_angles = {
        ('C', 'C', 'C'): 109.5,
        ('N', 'C', 'C'): 110.0,
        ('C', 'N', 'H'): 120.0,
        ('C', 'C', 'O'): 109.5,
        ('H', 'C', 'H'): 109.5
    }
    key = (atom1.element, atom2.element, atom3.element)
    return bond_angles.get(key, 109.5)  # Default to tetrahedral angle

def calculate_rmsd(coords1, coords2):
    """
    Calculates the Root Mean Square Deviation (RMSD) between two sets of coordinates.

    Args:
        coords1 (np.array): First set of coordinates (N, 3).
        coords2 (np.array): Second set of coordinates (N, 3).

    Returns:
        float: The RMSD value.
    """
    diff = coords1 - coords2
    return np.sqrt(np.sum(diff * diff) / coords1.shape[0])

def superimpose(mobile_coords, reference_coords):
    """
    Superimposes mobile_coords onto reference_coords using the Kabsch algorithm.

    Args:
        mobile_coords (np.array): Coordinates to be moved (N, 3).
        reference_coords (np.array): Static coordinates (N, 3).

    Returns:
        tuple: A tuple containing:
            - np.array: The rotated and translated mobile coordinates.
            - float: The RMSD after superimposition.
    """
    # Center the coordinates
    mobile_centroid = mobile_coords.mean(axis=0)
    reference_centroid = reference_coords.mean(axis=0)
    mobile_centered = mobile_coords - mobile_centroid
    reference_centered = reference_coords - reference_centroid

    # Calculate the covariance matrix
    covariance_matrix = np.dot(mobile_centered.T, reference_centered)

    # Singular Value Decomposition (SVD)
    u, s, vh = np.linalg.svd(covariance_matrix)

    # Calculate the rotation matrix
    # Handle reflection case
    d = np.linalg.det(np.dot(vh.T, u.T))
    if d < 0:
        vh[2, :] *= -1

    rotation_matrix = np.dot(vh.T, u.T)

    # Apply the rotation and translation
    rotated_coords = np.dot(mobile_centered, rotation_matrix) + reference_centroid

    # Calculate RMSD
    rmsd = calculate_rmsd(rotated_coords, reference_coords)

    return rotated_coords, rmsd

def get_atoms_from_structure(structure):
    """
    Extracts all atoms from a Structure object.

    Args:
        structure (Structure): The structure to extract atoms from.

    Returns:
        list: A list of all Atom objects in the structure.
    """
    atoms = []
    for model in structure.models:
        for chain in model.chains:
            for residue in chain.residues:
                atoms.extend(residue.atoms)
    return atoms

def get_residues_from_structure(structure):
    """
    Extracts all residues from a Structure object.

    Args:
        structure (Structure): The structure to extract residues from.

    Returns:
        list: A list of all Residue objects in the structure.
    """
    residues = []
    for model in structure.models:
        for chain in model.chains:
            residues.extend(chain.residues)
    return residues
