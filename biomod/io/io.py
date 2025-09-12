"""
This module handles reading of PDB and mmCIF files.
"""
import builtins
import gzip
from . import pdb
from . import mmcif
from . import mmtf
from ..core.base import Model, Chain
from ..core.residues import Residue, Ligand
from ..core.atoms import Atom
from .builder import data_dict_to_file

class PdbFile:
    def __init__(self, model):
        self.model = model

def open(path, data_dict=False, file_dict=False):
    """
    Opens a PDB, mmCIF or MMTF file and returns a file object.
    This function is a generic entry point that will delegate to the
    appropriate parser based on the file extension.
    """
    path_str = str(path)
    extension = path_str.split(".")[-1]
    if extension == "gz":
        extension = path_str.split(".")[-2]
    if extension == "ent":
        extension = "pdb"

    content = None
    if path_str.endswith(".gz"):
        with gzip.open(path_str, "rt") as f:
            content = f.read()
    elif extension == "mmtf":
        with builtins.open(path_str, "rb") as f:
            content = f.read()
    else:
        with builtins.open(path_str) as f:
            content = f.read()

    file_dict_data = {}
    if extension == "pdb":
        file_dict_data = pdb.pdb_string_to_pdb_dict(content)
    elif extension == "cif":
        file_dict_data = mmcif.mmcif_string_to_mmcif_dict(content)
    elif extension == "mmtf":
        file_dict_data = mmtf.mmtf_bytes_to_mmtf_dict(content)

    if file_dict:
        return file_dict_data

    data_dict_data = {}
    if extension == "pdb":
        data_dict_data = pdb.pdb_dict_to_data_dict(file_dict_data)
    elif extension == "cif":
        data_dict_data = mmcif.mmcif_dict_to_data_dict(file_dict_data)
    elif extension == "mmtf":
        data_dict_data = mmtf.mmtf_dict_to_data_dict(file_dict_data)

    if data_dict:
        return data_dict_data

    return data_dict_to_file(data_dict_data, extension)


def data_dict_to_model(data_dict):
    """
    Converts a data_dict to a Model object.
    """
    model_dict = data_dict["models"][0]
    chains = []
    for chain_id, chain_dict in model_dict["polymer"].items():
        residues = []
        for res_id, res_dict in chain_dict["residues"].items():
            atoms = []
            for atom_id, atom_dict in res_dict["atoms"].items():
                atom = Atom(
                    atom_dict["element"],
                    atom_dict["x"], atom_dict["y"], atom_dict["z"],
                    id=atom_id, name=atom_dict["name"], charge=atom_dict["charge"],
                    bvalue=atom_dict["bvalue"]
                )
                atoms.append(atom)
            residue = Residue(
                *atoms, id=res_id, name=res_dict["name"]
            )
            residues.append(residue)
        chain = Chain(*residues, id=chain_id)
        chains.append(chain)

    return Model(*chains)
