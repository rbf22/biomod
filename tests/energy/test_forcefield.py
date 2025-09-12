import os
import torch
import argparse
from biomod.energy import vitra_tensors as data_structures
from biomod.io import vitra_utils as utils
from biomod.energy.force_field import ForceField

def test_forcefield_regression():
    """
    Tests that the ForceField output matches the reference data.
    """
    # Load reference data
    reference_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_data.pt")
    reference_energies = torch.load(reference_data_path)

    # Load PDB and compute energies
    pdb_file = "tests/reference_data/alanine.pdb"
    device = "cpu"

    coordinates, atom_names, _ = utils.parse_pdb(pdb_file)
    info_tensors = data_structures.create_info_tensors(atom_names, device=device)

    container = ForceField(device=device)
    weights_path = "biomod/parameters/final_model.weights"
    container.load_state_dict(torch.load(weights_path, map_location=torch.device(device)), strict=False)

    energies = container(coordinates.to(device), info_tensors).data

    # Compare current output with reference data
    assert torch.allclose(energies, reference_energies), "ForceField output does not match reference data."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    if args.regenerate:
        # Load PDB and compute energies
        pdb_file = "tests/reference_data/alanine.pdb"
        device = "cpu"
        coordinates, atom_names, _ = utils.parse_pdb(pdb_file)
        info_tensors = data_structures.create_info_tensors(atom_names, device=device)
        container = ForceField(device=device)
        weights_path = "biomod/parameters/final_model.weights"
        container.load_state_dict(torch.load(weights_path, map_location=torch.device(device)), strict=False)
        energies = container(coordinates.to(device), info_tensors).data
        # Save reference data
        reference_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_data.pt")
        torch.save(energies, reference_data_path)
        print("Reference data regenerated.")
