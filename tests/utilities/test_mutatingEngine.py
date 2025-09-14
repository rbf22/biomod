import unittest
import torch

from biomod.utilities import mutatingEngine

class TestMutatingEngine(unittest.TestCase):
    def test_mutate_ala_to_gly(self):
        # Create a single Alanine residue for mutation
        # The 'mut' part of the name is 1, to indicate it's a mutable residue
        coords = torch.randn(1, 5, 3)
        atName = [
            ['ALA_1_N_A_1_0', 'ALA_1_CA_A_1_0', 'ALA_1_C_A_1_0', 'ALA_1_O_A_1_0', 'ALA_1_CB_A_1_0']
        ]

        # The mutation list refers to residue 1 of chain A, mutating it to GLY
        mutationList = [
            [[ "1_A_GLY" ]]
        ]

        new_coords, new_atName = mutatingEngine.mutate(coords, atName, mutationList)

        # The original atName has 5 atoms. The new mutant (GLY) has 4 atoms.
        # The code appends the new mutant atoms to the original list of atom names.
        # But it seems the logic is to create a new set of atoms for the mutant, not to modify the original.
        # atNameNew += [atNameProt + mutAname]
        # Let's check the number of atoms in the new mutant part.

        mutant_atoms = [name for name in new_atName[0] if name.startswith('GLY')]
        self.assertEqual(len(mutant_atoms), 4)

        # Check that the CB atom is not in the new glycine residue
        for atom_name in mutant_atoms:
            self.assertNotIn('CB', atom_name)

    def test_no_mutation(self):
        coords = torch.randn(1, 5, 3)
        atName = [
            ['ALA_1_N_A_0_0', 'ALA_1_CA_A_0_0', 'ALA_1_C_A_0_0', 'ALA_1_O_A_0_0', 'ALA_1_CB_A_0_0']
        ]
        mutationList = [
            []
        ]
        new_coords, new_atName = mutatingEngine.mutate(coords, atName, mutationList)
        self.assertEqual(len(new_atName[0]), 5)
        self.assertEqual(len(new_coords[0]), 5)

    def test_invalid_mutation_list(self):
        coords = torch.randn(1, 5, 3)
        atName = [
            ['ALA_1_N_A_0_0', 'ALA_1_CA_A_0_0', 'ALA_1_C_A_0_0', 'ALA_1_O_A_0_0', 'ALA_1_CB_A_0_0']
        ]
        mutationList = [] # a list with length different from coords
        with self.assertRaises(ValueError):
            mutatingEngine.mutate(coords, atName, mutationList)

    def test_mutate_ala_to_phe(self):
        # Create a single Alanine residue for mutation
        coords = torch.randn(1, 5, 3)
        atName = [
            ['ALA_1_N_A_1_0', 'ALA_1_CA_A_1_0', 'ALA_1_C_A_1_0', 'ALA_1_O_A_1_0', 'ALA_1_CB_A_1_0']
        ]

        # The mutation list refers to residue 1 of chain A, mutating it to PHE
        mutationList = [
            [[ "1_A_PHE" ]]
        ]

        # We need to provide dummy coordinates for the atoms that are used to build the new ones.
        mutatingEngine.protdiz = {
            'A': {
                '1': {
                    'N': torch.randn(3), 'CA': torch.randn(3), 'C': torch.randn(3),
                    'CB': torch.randn(3), 'CG': torch.randn(3), 'CD1': torch.randn(3),
                    'CD2': torch.randn(3), 'CE1': torch.randn(3), 'CE2': torch.randn(3),
                }
            }
        }

        new_coords, new_atName = mutatingEngine.mutate(coords, atName, mutationList)

        mutant_atoms = [name for name in new_atName[0] if name.startswith('PHE')]
        self.assertEqual(len(mutant_atoms), 12)

    def test_mutate_gly_to_ala(self):
        # Create a single Glycine residue for mutation
        coords = torch.randn(1, 4, 3)
        atName = [
            ['GLY_1_N_A_1_0', 'GLY_1_CA_A_1_0', 'GLY_1_C_A_1_0', 'GLY_1_O_A_1_0']
        ]

        # The mutation list refers to residue 1 of chain A, mutating it to ALA
        mutationList = [
            [[ "1_A_ALA" ]]
        ]

        mutatingEngine.protdiz = {
            'A': {
                '1': {
                    'N': torch.randn(3), 'CA': torch.randn(3), 'C': torch.randn(3),
                }
            }
        }

        new_coords, new_atName = mutatingEngine.mutate(coords, atName, mutationList)

        mutant_atoms = [name for name in new_atName[0] if name.startswith('ALA')]
        self.assertEqual(len(mutant_atoms), 5)
