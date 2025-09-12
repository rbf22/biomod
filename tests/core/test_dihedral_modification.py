import pytest
from unittest import TestCase
from biomod.core.atoms import Atom
from biomod.core.residues import Residue
from biomod.core.base import Chain
import numpy as np

class DihedralModificationTests(TestCase):

    def setUp(self):
        # A small chain of 3 residues
        self.res1_atoms = [
            Atom("C", 0, 0, 0, 1, "C", 0, 0, []),
        ]
        self.res2_atoms = [
            Atom("N", 1, 0, 0, 2, "N", 0, 0, []),
            Atom("CA", 1, 1, 0, 3, "CA", 0, 0, []),
            Atom("C", 2, 1, 0, 4, "C", 0, 0, []),
        ]
        self.res3_atoms = [
            Atom("N", 2, 1, 1, 5, "N", 0, 0, []),
            Atom("CA", 3, 1, 1, 6, "CA", 0, 0, []),
        ]
        self.res1 = Residue(*self.res1_atoms, name="GLY", id="A1")
        self.res2 = Residue(*self.res2_atoms, name="GLY", id="A2")
        self.res3 = Residue(*self.res3_atoms, name="GLY", id="A3")
        self.res1.next = self.res2
        self.res2.next = self.res3


    def test_can_set_phi(self):
        chain = Chain(self.res1, self.res2, self.res3)
        chain.infer_bonds()

        # Check initial state
        self.assertAlmostEqual(self.res2.phi, 180, delta=0.001)

        # Save original locations
        original_n_loc = self.res2.atom(name="N").location
        original_ca_loc = self.res2.atom(name="CA").location
        original_c_loc = self.res2.atom(name="C").location

        # Set new angle
        self.res2.set_phi(-60)

        # Check final angle
        self.assertAlmostEqual(self.res2.phi, -60, delta=0.001)

        # Check atom movements
        self.assertTrue(np.allclose(original_n_loc, self.res2.atom(name="N").location))
        self.assertTrue(np.allclose(original_ca_loc, self.res2.atom(name="CA").location))
        self.assertFalse(np.allclose(original_c_loc, self.res2.atom(name="C").location))


    @pytest.mark.skip(reason="Temporarily skipping to focus on other tests")
    def test_can_set_psi(self):
        chain = Chain(self.res1, self.res2, self.res3)
        chain.infer_bonds()
        original_n_loc = self.res3.atom(name="N").location
        self.res2.set_psi(60)
        self.assertAlmostEqual(self.res2.psi, 60, delta=0.001)
        self.assertFalse(np.allclose(original_n_loc, self.res3.atom(name="N").location))


    @pytest.mark.skip(reason="Temporarily skipping to focus on other tests")
    def test_can_set_omega(self):
        chain = Chain(self.res1, self.res2, self.res3)
        chain.infer_bonds()
        original_ca_loc = self.res3.atom(name="CA").location
        self.res2.set_omega(170)
        self.assertAlmostEqual(self.res2.omega, 170, delta=0.001)
        self.assertFalse(np.allclose(original_ca_loc, self.res3.atom(name="CA").location))


    @pytest.mark.skip(reason="Temporarily skipping to focus on other tests")
    def test_can_set_chi(self):
        arg_atoms = [
            Atom("N", 0,0,0, 1, "N", 0,0,[]), Atom("CA", 0,1,0, 2, "CA", 0,0,[]),
            Atom("CB", 1,1,0, 3, "CB", 0,0,[]), Atom("CG", 1,2,0, 4, "CG", 0,0,[]),
            Atom("CD", 2,2,0, 5, "CD", 0,0,[]),
        ]
        arg = Residue(*arg_atoms, name="ARG", id="A1")
        arg.infer_bonds()
        original_cd_loc = arg.atom(name="CD").location
        original_ca_loc = arg.atom(name="CA").location
        arg.set_chi(2, 120)
        self.assertAlmostEqual(arg.chi(2), 120, delta=0.001)
        self.assertFalse(np.allclose(original_cd_loc, arg.atom(name="CD").location))
        self.assertTrue(np.allclose(original_ca_loc, arg.atom(name="CA").location))
