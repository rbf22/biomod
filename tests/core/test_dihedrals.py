from unittest import TestCase
from unittest.mock import patch, PropertyMock
from biomod.core.atoms import Atom
from biomod.core.residues import Residue
from biomod.core.base import Chain

class DihedralAngleTests(TestCase):

    def setUp(self):
        self.atom1 = Atom("C", 0, 0, 0, 1, "C1", 0, 0, [0,0,0,0,0,0])
        self.atom2 = Atom("N", 1, 0, 0, 2, "N1", 0, 0, [0,0,0,0,0,0])
        self.atom3 = Atom("C", 1, 1, 0, 3, "C2", 0, 0, [0,0,0,0,0,0])
        self.atom4 = Atom("N", 2, 1, 0, 4, "N2", 0, 0, [0,0,0,0,0,0])

    def test_can_calculate_dihedral_angle(self):
        angle = Atom.dihedral(self.atom1, self.atom2, self.atom3, self.atom4)
        self.assertAlmostEqual(angle, 180, delta=0.001)

    def test_can_calculate_phi_psi_omega(self):
        # Create a mini-chain of 3 residues
        res1_atoms = [
            Atom("C", 0, 0, 0, 1, "C", 0, 0, []),
        ]
        res2_atoms = [
            Atom("N", 1, 0, 0, 2, "N", 0, 0, []),
            Atom("CA", 1, 1, 0, 3, "CA", 0, 0, []),
            Atom("C", 2, 1, 0, 4, "C", 0, 0, []),
        ]
        res3_atoms = [
            Atom("N", 2, 1, 1, 5, "N", 0, 0, []),
            Atom("CA", 3, 1, 1, 6, "CA", 0, 0, []),
        ]
        res1 = Residue(*res1_atoms, name="GLY", id="A1")
        res2 = Residue(*res2_atoms, name="GLY", id="A2")
        res3 = Residue(*res3_atoms, name="GLY", id="A3")
        res1.next = res2
        res2.next = res3

        # Test middle residue
        self.assertIsNotNone(res2.phi)
        self.assertAlmostEqual(res2.phi, 180, delta=0.001)
        self.assertAlmostEqual(res2.psi, -90, delta=0.001)
        self.assertAlmostEqual(res2.omega, 180, delta=0.001)

        # Test termini
        self.assertIsNone(res1.phi)
        self.assertIsNone(res3.psi)
        self.assertIsNone(res3.omega)

    def test_chi_angles(self):
        # Test ARG
        arg_atoms = [
            Atom("N", 0,0,0, 1, "N", 0,0,[]), Atom("CA", 0,1,0, 2, "CA", 0,0,[]),
            Atom("CB", 1,1,0, 3, "CB", 0,0,[]), Atom("CG", 1,2,0, 4, "CG", 0,0,[]),
            Atom("CD", 2,2,0, 5, "CD", 0,0,[]), Atom("NE", 2,3,0, 6, "NE", 0,0,[]),
            Atom("CZ", 3,3,0, 7, "CZ", 0,0,[]), Atom("NH1", 3,4,0, 8, "NH1", 0,0,[])
        ]
        arg = Residue(*arg_atoms, name="ARG", id="A1")
        self.assertAlmostEqual(arg.chi(1), 180, delta=0.001)
        self.assertAlmostEqual(arg.chi(2), 180, delta=0.001)
        self.assertAlmostEqual(arg.chi(3), 180, delta=0.001)
        self.assertAlmostEqual(arg.chi(4), 180, delta=0.001)
        self.assertAlmostEqual(arg.chi(5), 180, delta=0.001)
        self.assertIsNone(arg.chi(6))

        # Test GLY (no chi)
        gly = Residue(name="GLY", id="A2")
        self.assertIsNone(gly.chi(1))

        # Test missing atom
        arg_missing = Residue(*arg_atoms[:-1], name="ARG", id="A3")
        self.assertIsNone(arg_missing.chi(5))
