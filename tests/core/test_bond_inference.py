from unittest import TestCase
from unittest.mock import patch, PropertyMock
from biomod.core.atoms import Atom
from biomod.core.residues import Residue

class BondInferenceTest(TestCase):

    def setUp(self):
        self.atom1 = Atom("C", 0, 0, 0, 1, "C1", 0, 0, [0, 0, 0, 0, 0, 0])
        self.atom2 = Atom("N", 1, 0, 0, 2, "N1", 0, 0, [0, 0, 0, 0, 0, 0])
        self.atom3 = Atom("O", 10, 10, 10, 3, "O1", 0, 0, [0, 0, 0, 0, 0, 0])
        self.atom4 = Atom("S", 2, 0, 0, 4, "S1", 0, 0, [0, 0, 0, 0, 0, 0])


    @patch("biomod.core.atoms.Atom.covalent_radius", new_callable=PropertyMock)
    def test_can_infer_bonds(self, mock_radius):
        mock_radius.side_effect = [0.7, 0.6, 0.6, 0.7, 0.6, 0.6, 0.7, 0.6, 0.6]
        residue = Residue(self.atom1, self.atom2, self.atom3)
        residue.infer_bonds()
        self.assertEqual(self.atom1.bonded_atoms, {self.atom2})
        self.assertEqual(self.atom2.bonded_atoms, {self.atom1})
        self.assertEqual(self.atom3.bonded_atoms, set())


    @patch("biomod.core.atoms.Atom.covalent_radius", new_callable=PropertyMock)
    def test_can_infer_bonds_with_tolerance(self, mock_radius):
        mock_radius.side_effect = [0.7, 1.0, 0.7, 1.0]
        residue = Residue(self.atom1, self.atom4)
        residue.infer_bonds(tolerance=0.4)
        self.assertEqual(self.atom1.bonded_atoms, {self.atom4})
        self.assertEqual(self.atom4.bonded_atoms, {self.atom1})


    @patch("biomod.core.atoms.Atom.covalent_radius", new_callable=PropertyMock)
    def test_infer_bonds_needs_tolerance(self, mock_radius):
        mock_radius.side_effect = [0.7, 1.0, 0.7, 1.0]
        residue = Residue(self.atom1, self.atom4)
        residue.infer_bonds(tolerance=0.2)
        self.assertEqual(self.atom1.bonded_atoms, set())
        self.assertEqual(self.atom4.bonded_atoms, set())
