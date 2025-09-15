import pytest
from biomod.core.atoms import Atom
from biomod.core.residues import Residue

class TestDihedralAngles:

    def setup_method(self, method):
        self.atom1 = Atom("C", 0, 0, 0, 1, "C1", 0, 0, [0,0,0,0,0,0])
        self.atom2 = Atom("N", 1, 0, 0, 2, "N1", 0, 0, [0,0,0,0,0,0])
        self.atom3 = Atom("C", 1, 1, 0, 3, "C2", 0, 0, [0,0,0,0,0,0])
        self.atom4 = Atom("N", 2, 1, 0, 4, "N2", 0, 0, [0,0,0,0,0,0])

    def test_can_calculate_dihedral_angle(self):
        angle = Atom.dihedral(self.atom1, self.atom2, self.atom3, self.atom4)
        assert angle == pytest.approx(180, abs=0.001)

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
        assert res2.phi is not None
        assert res2.phi == pytest.approx(180, abs=0.001)
        assert res2.psi == pytest.approx(-90, abs=0.001)
        assert res2.omega == pytest.approx(180, abs=0.001)

        # Test termini
        assert res1.phi is None
        assert res3.psi is None
        assert res3.omega is None

    def test_chi_angles(self):
        # Test ARG
        arg_atoms = [
            Atom("N", 0,0,0, 1, "N", 0,0,[]), Atom("CA", 0,1,0, 2, "CA", 0,0,[]),
            Atom("CB", 1,1,0, 3, "CB", 0,0,[]), Atom("CG", 1,2,0, 4, "CG", 0,0,[]),
            Atom("CD", 2,2,0, 5, "CD", 0,0,[]), Atom("NE", 2,3,0, 6, "NE", 0,0,[]),
            Atom("CZ", 3,3,0, 7, "CZ", 0,0,[]), Atom("NH1", 3,4,0, 8, "NH1", 0,0,[])
        ]
        arg = Residue(*arg_atoms, name="ARG", id="A1")
        assert arg.chi(1) == pytest.approx(180, abs=0.001)
        assert arg.chi(2) == pytest.approx(180, abs=0.001)
        assert arg.chi(3) == pytest.approx(180, abs=0.001)
        assert arg.chi(4) == pytest.approx(180, abs=0.001)
        assert arg.chi(5) == pytest.approx(180, abs=0.001)
        assert arg.chi(6) is None

        # Test GLY (no chi)
        gly = Residue(name="GLY", id="A2")
        assert gly.chi(1) is None

        # Test missing atom
        arg_missing = Residue(*arg_atoms[:-1], name="ARG", id="A3")
        assert arg_missing.chi(5) is None
