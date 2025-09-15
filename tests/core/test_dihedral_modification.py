import pytest
from biomod.core.atoms import Atom
from biomod.core.residues import Residue
from biomod.core.base import Chain
import numpy as np

class TestDihedralModification:

    def setup_method(self, method):
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
        assert self.res2.phi == pytest.approx(180, abs=0.001)

        # Save original locations
        original_n_loc = self.res2.atom(name="N").location
        original_ca_loc = self.res2.atom(name="CA").location
        original_c_loc = self.res2.atom(name="C").location

        # Set new angle
        self.res2.set_phi(-60)

        # Check final angle
        assert self.res2.phi == pytest.approx(-60, abs=0.001)

        # Check atom movements
        assert np.allclose(original_n_loc, self.res2.atom(name="N").location)
        assert np.allclose(original_ca_loc, self.res2.atom(name="CA").location)
        assert not np.allclose(original_c_loc, self.res2.atom(name="C").location)


    def test_can_set_psi(self):
        chain = Chain(self.res1, self.res2, self.res3)
        chain.infer_bonds()
        original_n_loc = self.res3.atom(name="N").location
        self.res2.set_psi(60)
        assert self.res2.psi == pytest.approx(60, abs=0.001)
        assert not np.allclose(original_n_loc, self.res3.atom(name="N").location)


    def test_can_set_omega(self):
        chain = Chain(self.res1, self.res2, self.res3)
        chain.infer_bonds()
        original_ca_loc = self.res3.atom(name="CA").location
        self.res2.set_omega(170)
        assert self.res2.omega == pytest.approx(170, abs=0.001)
        assert not np.allclose(original_ca_loc, self.res3.atom(name="CA").location)


    def test_can_set_chi(self):
        arg_atoms = [
            Atom("N", 0,0,0, 1, "N", 0,0,[]), Atom("CA", 0,1,0, 2, "CA", 0,0,[]),
            Atom("CB", 1,1,0, 3, "CB", 0,0,[]), Atom("CG", 2,2,0, 4, "CG", 0,0,[]),
            Atom("CD", 3,2,1, 5, "CD", 0,0,[]),
        ]
        arg = Residue(*arg_atoms, name="ARG", id="A1")
        arg.infer_bonds()
        original_cd_loc = arg.atom(name="CD").location
        original_ca_loc = arg.atom(name="CA").location
        arg.set_chi(2, 120)
        assert arg.chi(2) == pytest.approx(120, abs=0.001)
        assert not np.allclose(original_cd_loc, arg.atom(name="CD").location)
        assert np.allclose(original_ca_loc, arg.atom(name="CA").location)
