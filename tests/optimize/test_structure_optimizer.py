import unittest
import torch

from biomod.optimize import structure_optimizer
from biomod.energy import hashings

class MockForceField(torch.nn.Module):
    def __init__(self):
        super(MockForceField, self).__init__()
        self.device = torch.device("cpu")

    def forward(self, coords, info_tensors):
        # Return a dummy energy tensor that depends on the input coords
        # to ensure the computation graph is connected.
        return torch.randn(1, 1, 1, 1, 11) + coords.sum() * 0

class TestStructureOptimizer(unittest.TestCase):
    def test_optimizer_runs(self):
        model = MockForceField()
        n_atoms = 11
        coords = torch.randn(1, n_atoms, 3)

        # Create dummy info_tensors
        atom_number = torch.tensor([n_atoms])
        atom_description = torch.zeros(n_atoms, 10, dtype=torch.long)
        atom_description[:, 0] = 0 # batch
        atom_description[:, 1] = 0 # chain
        atom_description[:, 2] = 0 # resnum
        atom_description[:, 3] = hashings.resi_hash['ARG'] # resname

        arg_atoms = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2']
        for i, atom_name in enumerate(arg_atoms):
            atom_description[i, 4] = hashings.atom_hash['ARG'][atom_name] # aname

        coordsIndexingAtom = torch.arange(n_atoms)
        partnersIndexingAtom = torch.zeros(n_atoms, 10, dtype=torch.long)
        angle_indices = torch.zeros(n_atoms, 10, dtype=torch.long)
        alternativeMask = torch.ones(n_atoms, 1, dtype=torch.bool)

        info_tensors = (
            atom_number,
            atom_description,
            coordsIndexingAtom,
            partnersIndexingAtom,
            angle_indices,
            alternativeMask
        )

        # In optimize(), maxseq is calculated from resnum.max() + 1.
        # In this test, resnum is all 0, so maxseq is 1.
        # The output yp should have shape (1, 1, 1, 1, 11)
        yp, coords_local = structure_optimizer.optimize(
            model, coords, info_tensors, epochs=1
        )

        self.assertEqual(yp.shape, (1, 1, 1, 1, 11))
        self.assertEqual(coords_local.shape, coords.shape)

    def test_optimizer_verbose(self):
        model = MockForceField()
        n_atoms = 11
        coords = torch.randn(1, n_atoms, 3)

        # Create dummy info_tensors
        atom_number = torch.tensor([n_atoms])
        atom_description = torch.zeros(n_atoms, 10, dtype=torch.long)
        atom_description[:, 0] = 0 # batch
        atom_description[:, 1] = 0 # chain
        atom_description[:, 2] = 0 # resnum
        atom_description[:, 3] = hashings.resi_hash['ARG'] # resname

        arg_atoms = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2']
        for i, atom_name in enumerate(arg_atoms):
            atom_description[i, 4] = hashings.atom_hash['ARG'][atom_name] # aname

        coordsIndexingAtom = torch.arange(n_atoms)
        partnersIndexingAtom = torch.zeros(n_atoms, 10, dtype=torch.long)
        angle_indices = torch.zeros(n_atoms, 10, dtype=torch.long)
        alternativeMask = torch.ones(n_atoms, 1, dtype=torch.bool)

        info_tensors = (
            atom_number,
            atom_description,
            coordsIndexingAtom,
            partnersIndexingAtom,
            angle_indices,
            alternativeMask
        )

        yp, coords_local = structure_optimizer.optimize(
            model, coords, info_tensors, epochs=1, verbose=True
        )

        self.assertEqual(yp.shape, (1, 1, 1, 1, 11))
        self.assertEqual(coords_local.shape, coords.shape)
