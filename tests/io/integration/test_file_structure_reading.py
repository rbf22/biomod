from datetime import date
import math
import pytest
from biomod.core.atoms import Atom
from biomod.core.residues import Residue, Ligand
from biomod.core.base import Chain, Model
from biomod.io import io

class TestDeNovoStructure:

    def test_structure_processing(self):
        # Create five atoms of a residue
        atom1 = Atom("N", 0, 0, 0, 1, "N", 0.5, 0.5, [0] * 6)
        atom2 = Atom("C", 1.5, 0, 0, 2, "CA", 0, 0.4, [0] * 6)
        atom3 = Atom("C", 1.5, 1.5, 0, 3, "CB", 0, 0.3, [1] * 6)
        atom4 = Atom("C", 3, 0, 0, 4, "C", 0, 0.2, [0] * 6)
        atom5 = Atom("O", 3, -1.5, 0, 5, "O", 0, 0.1, [0] * 6)

        # Check basic atom properties
        assert atom2.element == "C"
        assert atom1.location == (0, 0, 0)
        assert tuple(atom1) == (0, 0, 0)
        assert atom2.name == "CA"
        assert atom1.charge == 0.5
        assert atom2.charge == 0
        assert atom2.bvalue == 0.4
        assert atom3.anisotropy == [1] * 6

        # Check can update some properties
        atom1.name = "HG"
        atom1.charge = 200
        atom1.bvalue = 20
        assert atom1.name == "HG"
        assert atom1.charge == 200
        assert atom1.bvalue == 20
        atom1.name = "N"
        atom1.charge = 0.5
        atom1.bvalue = 0.5

        # Check atoms are not part of any higher structures
        for atom in (atom1, atom2, atom3, atom4, atom5):
            assert atom.het is None
            assert atom.chain is None
            assert atom.model is None
            assert atom.bonded_atoms == set()

        # Check atoms' calculated properties
        assert atom5.mass == pytest.approx(16, abs=0.05)
        assert atom5.atomic_number == 8
        assert atom1.covalent_radius == 0.71
        for atom in (atom1, atom2, atom3, atom4, atom5):
            assert not atom.is_metal
            assert not atom.is_backbone # Not yet
            assert not atom.is_side_chain # Not yet

        # Check atom magic methods
        assert list(atom5) == [3, -1.5, 0]
        for a1 in (atom1, atom2, atom3, atom4, atom5):
            for a2 in (atom1, atom2, atom3, atom4, atom5):
                if a1 is a2:
                    assert a1 == a2
                else:
                    assert a1 != a2

        # Check atom safe methods
        assert atom1.distance_to(atom2) == 1.5
        assert atom1.distance_to(atom3) == 4.5 ** 0.5
        assert atom2.angle(atom3, atom4) == math.pi / 2
        for atom in (atom1, atom2, atom3, atom4, atom5):
            assert atom.nearby_atoms(5) == set() # Not without model
            assert atom.nearby_hets(5) == set() # Ditto

        # Check atom side effect methods
        atom2.translate(0, 0, 1)
        assert atom2.location == (1.5, 0, 1)
        atom2.translate(0, 0, -1)
        assert atom2.location == (1.5, 0, 0)
        atom2.transform([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        assert atom2.location == (-1.5, 0, 0)
        atom2.transform([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        assert atom2.location == (1.5, 0, 0)
        atom2.rotate(math.pi / 2, "y")
        assert atom2.location == (0, 0, -1.5)
        atom2.rotate(math.pi * 1.5, "y")
        assert atom2.location == (1.5, 0, 0)
        atom2.move_to(10, 10, 10)
        assert atom2.location == (10, 10, 10)
        atom2.move_to(1.5, 0, 0)

        # Bond atoms
        atom1.bond(atom2)
        atom2.bond(atom3)
        atom2.bond(atom4)
        atom4.bond(atom5)
        assert atom1.bonded_atoms == {atom2}
        assert atom2.bonded_atoms == {atom1, atom3, atom4}
        assert atom3.bonded_atoms == {atom2}
        assert atom4.bonded_atoms == {atom2, atom5}
        assert atom5.bonded_atoms == {atom4}

        # Check can copy atom
        copy = atom2.copy()
        assert copy.element == "C"
        assert copy.location == (1.5, 0, 0)
        assert copy.name == "CA"
        assert copy.id == 2
        assert copy.charge == 0
        assert copy.charge == 0
        assert copy.bvalue == 0.4
        assert copy.anisotropy == [0] * 6
        assert copy.het is None
        assert copy.chain is None
        assert copy.model is None
        assert copy.bonded_atoms == set()
        assert atom2 == copy

        # Can copy atom with new ID
        copy = atom2.copy(id=10000)
        assert copy.id == 10000

        # Create residue
        res1 = Residue(
         atom1, atom2, atom3, atom4, atom5, id="A5", name="AL"
        )

        # Check basic residue properties
        assert res1.id == "A5"
        assert res1.name == "AL"
        assert res1.code == "X"
        res1.name = "ALA"
        assert res1.name == "ALA"
        assert res1.code == "A"
        assert res1.next is None
        assert res1.previous is None
        assert res1.chain is None

        # Check residue and atoms
        for atom in (atom1, atom2, atom3, atom4, atom5):
            assert atom.het is res1
        assert atom1.is_backbone
        assert atom3.is_side_chain
        assert res1.atoms() == {atom1, atom2, atom3, atom4, atom5}
        assert res1.atoms(element="C") == {atom2, atom3, atom4}
        assert res1.atoms(name="O") == {atom5}
        assert res1.atoms(is_backbone=True) == {atom1, atom2, atom4, atom5}
        assert res1.atoms(mass__gt=13) == {atom1, atom5}
        assert res1.atoms(name__regex="N|O") == {atom1, atom5}

        # Check residue is container
        assert atom1 in res1
        assert copy in res1

        # Check residue calculated properties
        assert res1.code == "A"
        assert res1.full_name == "alanine"
        assert res1.model is None
        assert res1.mass == pytest.approx(66, abs=0.05)
        assert res1.charge == 0.5
        assert res1.formula == {"C": 3, "O": 1, "N": 1}
        assert res1.center_of_mass[0] == pytest.approx(1.818, abs=0.001)
        assert res1.center_of_mass[1] == pytest.approx(-0.091, abs=0.001)
        assert res1.center_of_mass[2] == 0
        assert res1.radius_of_gyration == pytest.approx(1.473, abs=0.001)

        # Check residue safe methods
        assert len(tuple(res1.pairwise_atoms())) == 10
        assert res1.nearby_hets(10) == set()
        assert res1.nearby_atoms(10) == set()
        assert res1.nearby_chains(10) == set()
        assert tuple(res1.create_grid(size=3)) == (
         (0, -3, 0), (0, 0, 0), (0, 3, 0), (3, -3, 0), (3, 0, 0), (3, 3, 0)
        )
        assert res1.atoms_in_sphere((1.5, 0, 0), 1.5) == {atom2, atom1, atom3, atom4}
        assert res1.atoms_in_sphere((1.5, 0, 0), 1.5, element="C") == {atom2, atom3, atom4}
        res1.check_ids()
        assert not res1.helix
        assert not res1.strand

        # Check residue side effect methods
        res1.translate(0, 0, 1)
        assert atom2.location == (1.5, 0, 1)
        res1.translate(0, 0, -1)
        assert atom2.location == (1.5, 0, 0)
        res1.transform([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        assert atom2.location == (-1.5, 0, 0)
        res1.transform([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        assert atom2.location == (1.5, 0, 0)
        res1.rotate(math.pi / 2, "y")
        assert atom2.location == (0, 0, -1.5)
        res1.rotate(math.pi * 1.5, "y")
        assert atom2.location == (1.5, 0, 0)

        # Can make copy of residue
        res_copy = res1.copy()
        assert res1 == res_copy
        assert res_copy.id == "A5"
        assert res_copy.name == "ALA"
        assert res1.pairing_with(res_copy) == {
         atom1: res_copy.atom(1), atom2: res_copy.atom(2),
         atom3: res_copy.atom(3), atom4: res_copy.atom(4),
         atom5: res_copy.atom(5)
        }
        assert len(res1.atoms() | res_copy.atoms()) == 10
        assert res1.rmsd_with(res_copy) == 0
        res_copy.atom(1).translate(1)
        assert res1.rmsd_with(res_copy) == pytest.approx(0.4, abs=0.001)

        # Can make copy of residue with new IDs
        res_copy = res1.copy(id="C5", atom_ids=lambda i: i * 100)
        assert res_copy.id == "C5"
        assert {a.id for a in res_copy.atoms()} == {100, 200, 300, 400, 500}

        # Make more residues
        atom6 = Atom("N", 4.5, 0, 0, 6, "N", 0, 0.5, [0] * 6)
        atom7 = Atom("C", 6, 0, 0, 7, "CA", 0, 0.5, [0] * 6)
        atom8 = Atom("C", 6, -1.5, 0, 8, "CB", 0, 0.5, [0] * 6)
        atom9 = Atom("S", 6, -3, 0, 9, "S", 0, 0.5, [0] * 6)
        atom10 = Atom("C", 7.5, 0, 0, 10, "C", 0, 0.5, [0] * 6)
        atom11 = Atom("O", 7.5, 1.5, 0, 11, "O", 0, 0.5, [0] * 6)
        atom12 = Atom("N", 9, 0, 0, 12, "CA", 0, 0.5, [0] * 6)
        atom13 = Atom("C", 10.5, 0, 0, 13, "CB", 0, 0.5, [0] * 6)
        atom14 = Atom("C", 10.5, 1.5, 0, 14, "OG", 0, 0.5, [0] * 6)
        atom15 = Atom("O", 10.5, 3, 0, 15, "C", 0, 0.5, [0] * 6)
        atom16 = Atom("C", 12, 0, 0, 16, "C", 0, 0.5, [0] * 6)
        atom17 = Atom("O", 13.5, 1.25, 0, 17, "OX1", 0, 0.5, [0] * 6)
        atom18 = Atom("O", 13.5, -1.25, 0, 18, "OX2", 0, 0.5, [0] * 6)
        atom6.bond(atom7)
        atom6.bond(atom4)
        atom7.bond(atom8)
        atom7.bond(atom10)
        atom8.bond(atom9)
        atom10.bond(atom11)
        atom10.bond(atom12)
        atom12.bond(atom13)
        atom13.bond(atom14)
        atom13.bond(atom16)
        atom14.bond(atom15)
        atom16.bond(atom17)
        atom16.bond(atom18)
        res2 = Residue(
         atom6, atom7, atom8, atom9, atom10, atom11, id="A5A", name="CYS"
        )
        res3 = Residue(
         atom12, atom13, atom14, atom15, atom16, atom17, atom18, id="A6", name="SER"
        )

        # Connect residues
        res1.next = res2
        res3.previous = res2
        assert res1.next is res2
        assert res2.next is res3
        assert res3.previous is res2
        assert res2.previous is res1

        # Create chain
        chain1 = Chain(
         res1, res2, res3, id="A", sequence="MACSD", helices=((res1, res2),), strands=((res3,),)
        )
        assert chain1.id == "A"
        assert chain1.internal_id == "A"
        assert chain1.name is None
        assert chain1.model is None

        # Chain properties
        assert chain1.sequence == "MACSD"
        chain1.sequence = "MACSDA"
        assert chain1.sequence == "MACSDA"
        assert chain1.present_sequence == "ACS"
        assert chain1.helices[0] == (res1, res2)
        assert chain1.strands[0] == (res3,)

        # Check chain residues and atoms
        assert chain1.residues() == (res1, res2, res3)
        assert chain1.residues(mass__gt=80) == (res2, res3)
        assert chain1.residues(mass__lt=98.1) == (res1, res3)
        assert chain1.atoms() == {
         atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9, atom10,
         atom11, atom12, atom13, atom14, atom15, atom16, atom17, atom18
        }
        assert chain1.atoms(het__name="ALA") == {
         atom1, atom2, atom3, atom4, atom5
        }
        assert chain1.atoms(het__name="CYS") == {
         atom6, atom7, atom8, atom9, atom10, atom11
        }
        assert chain1.atoms(het__name__regex="CYS|ALA") == {
         atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9, atom10, atom11
        }
        assert res1.helix
        assert not res1.strand
        assert res2.helix
        assert not res2.strand
        assert not res3.helix
        assert res3.strand

        # Check chain magic methods
        assert chain1.length == 3
        assert chain1[0] is res1
        for res in chain1:
            assert res in (res1, res2, res3)
        assert res1 in chain1
        assert atom10 in chain1

        # Check chain ligands and atoms
        assert chain1.ligands() == set()
        assert (
         chain1.atoms(element__regex="O|S")
         == {atom5, atom9, atom11, atom15, atom17, atom18}
        )
        assert (
         chain1.atoms(het__name="CYS")
         == {atom6, atom7, atom8, atom9, atom10, atom11}
        )

        # Make copy of chain
        chain2 = chain1.copy(
         id="B",
         residue_ids=lambda i: i.replace("A", "B"),
         atom_ids=lambda i: i * 100
        )
        assert chain1 == chain2
        assert chain2.id == "B"
        assert chain2.internal_id == "B"
        assert [res.id for res in chain2] == ["B5", "B5B", "B6"]
        assert {a.id for a in chain2.atoms()} == set([x * 100 for x in range(1, 19)])
        assert chain2[0].next is chain2[1]
        assert chain2[2].previous is chain2[1]
        assert chain2.helices == ((chain2[0], chain2[1]),)
        assert chain2.strands == ((chain2[2],),)

        # Move chain into place
        chain2.rotate(math.pi, "x")
        chain2.rotate(math.pi, "y")
        chain2.translate(12, -10.5)
        assert chain1.rmsd_with(chain2) == 0

        # Make ligand
        copper_atom = Atom("Cu", 6, -5.25, 2, 100, "Cu", 2, 0, [0] * 6)
        copper = Ligand(
         copper_atom, id="A100", internal_id="M", name="CU", chain=chain1, full_name="copper"
        )

        # Check ligand properties
        assert copper.id == "A100"
        assert copper.name == "CU"
        assert copper.full_name == "copper"
        copper.full_name = None
        assert copper.full_name == "CU"
        assert copper.internal_id == "M"
        assert copper.chain is chain1
        assert copper.model is None
        assert copper.atom() == copper_atom
        assert copper_atom in copper
        assert not copper.is_water

        # Can make copy of ligand
        cu_copy = copper.copy()
        assert copper == cu_copy
        assert cu_copy.id == "A100"
        assert cu_copy.name == "CU"
        assert len(cu_copy.atoms() | cu_copy.atoms()) == 1
        assert not cu_copy.is_water

        # Can make copy of ligand with new IDs
        cu_copy = copper.copy(id="C100", atom_ids=lambda i: i * 100)
        assert cu_copy.id == "C100"
        assert cu_copy.atom().id == 10000

        # Create waters
        hoh1 = Ligand(
         Atom("O", 3, -3, 3, 500, "O", 0, 0, [0] * 6),
         id="A1000", name="HOH", water=True
        )
        assert hoh1.is_water
        hoh2 = Ligand(
         Atom("O", 3, -9, -3, 500, "O", 0, 0, [0] * 6),
         id="B1000", name="HOH", water=True
        )
        assert hoh2.is_water

        # Create model
        model = Model(chain1, chain2, copper, hoh1, hoh2)
        
        # Model properties
        assert model.file is None
        assert model.chains() == {chain1, chain2}
        assert model.ligands() == {copper}
        assert chain1.ligands() == {copper}
        assert model.waters() == {hoh1, hoh2}
        assert model.molecules() == {chain1, chain2, copper, hoh1, hoh2}
        assert model.residues() == set(chain1.residues() + chain2.residues())
        assert model.residues(name="ALA") == {chain1[0], chain2[0]}
        assert model.ligand() == copper
        assert model.atom(1) == atom1
        assert model.atom(name="N", het__name="ALA", chain__id="A") == atom1

        # Everything points upwards correctly
        assert atom1.model is model
        assert res1.model is model
        assert chain1.model is model
        assert copper.model is model

        # Now that atoms are in a model, find nearby things
        assert atom2.nearby_atoms(1.5) == {atom1, atom3, atom4}
        assert atom4.nearby_atoms(1.5) == {atom2, atom5, atom6}
        assert atom4.nearby_atoms(1.5, het__name="CYS") == {atom6}
        assert atom4.nearby_hets(1.5) == {res2}
        assert atom4.nearby_hets(9) == {res2, res3, chain2[1], copper, hoh1}
        assert atom4.nearby_hets(9, ligands=False) == {res2, res3, chain2[1]}
        assert atom4.nearby_hets(9, residues=False) == {copper, hoh1}
        assert atom4.nearby_hets(9, residues=False, het__is_water=False) == {copper}
        assert atom4.nearby_chains(9) == {chain2}
        assert atom4.nearby_chains(9, chain__id="A") == set()
        assert res2.nearby_hets(3) == {res1, res3}
        assert res2.nearby_hets(6) == {res1, res3, hoh1, copper, chain2[1]}
        assert res2.nearby_hets(6, ligands=False) == {res1, res3, chain2[1]}
        assert copper.nearby_chains(5) == {chain2}
        assert chain2.nearby_chains(5) == {chain1}

        # Dehydrate model
        model.dehydrate()
        assert model.waters() == set()
        assert model.ligands() == {copper}
        assert model.chains() == {chain1, chain2}



class TestFileReading:

    def test_1lol(self):
        for e in ["cif", "mmtf", "pdb", "pdb.gz", "cif.gz"]:
            f = io.open("tests/io/integration/files/1lol." + e)
            assert f.filetype == e.replace(".gz", "")
            assert f.code == "1LOL"
            if e.startswith("pdb"):
                assert (
                 f.title == "CRYSTAL STRUCTURE OF OROTIDINE MONOPHOSPHATE DECARBOXYLASE COMPLEX WITH XMP"
                )
            else:
                assert (
                 f.title == "Crystal structure of orotidine monophosphate decarboxylase complex with XMP"
                )
            assert f.deposition_date == date(2002, 5, 6)
            assert f.classification == (None if e == "mmtf" else "LYASE")
            assert f.keywords == ([] if e == "mmtf" else ["TIM BARREL", "LYASE"] if e.startswith("pdb") else ["TIM barrel", "LYASE"])
            assert f.authors == ([] if e == "mmtf" else ["N.WU", "E.F.PAI"] if e.startswith("pdb") else ["Wu, N.", "Pai, E.F."])
            assert f.technique == "X-RAY DIFFRACTION"
            missing_residues = [{"id": id, "name": name} for id, name in zip([
             "A.1", "A.2", "A.3", "A.4", "A.5", "A.6", "A.7", "A.8", "A.9", "A.10",
             "A.182", "A.183", "A.184", "A.185", "A.186", "A.187", "A.188", "A.189",
             "A.223", "A.224", "A.225", "A.226", "A.227", "A.228", "A.229", "B.1001",
             "B.1002", "B.1003", "B.1004", "B.1005", "B.1006", "B.1007", "B.1008",
             "B.1009", "B.1010", "B.1182", "B.1183", "B.1184", "B.1185", "B.1186"
            ], [
             "LEU", "ARG", "SER", "ARG", "ARG", "VAL", "ASP", "VAL", "MET", "ASP",
             "VAL", "GLY", "ALA", "GLN", "GLY", "GLY", "ASP", "PRO", "LYS", "ASP",
             "LEU", "LEU", "ILE", "PRO", "GLU", "LEU", "ARG", "SER", "ARG", "ARG",
             "VAL", "ASP", "VAL", "MET", "ASP", "VAL", "GLY", "ALA", "GLN", "GLY"
            ])]
            if e.startswith("pdb"):
                assert f.source_organism == "METHANOTHERMOBACTER THERMAUTOTROPHICUS STR. DELTA H"
                assert f.expression_system == "ESCHERICHIA COLI"
                assert f.missing_residues == missing_residues
            else:
                assert (
                 f.source_organism
                 == (None if e == "mmtf" else "Methanothermobacter thermautotrophicus str. Delta H")
                )
                assert (
                 f.expression_system == (None if e == "mmtf" else "Escherichia coli")
                )
                assert f.missing_residues == ([] if e == "mmtf" else missing_residues)
            assert f.resolution == 1.9
            assert f.rvalue == 0.193
            assert f.rfree == 0.229
            assert f.assemblies == [{
             "id": 1,
             "software": None if e == "mmtf" else "PISA",
             "delta_energy": None if e == "mmtf" else -31.0,
             "buried_surface_area": None if e == "mmtf" else 5230,
             "surface_area": None if e == "mmtf" else 16550,
             "transformations": [{
              "chains": ["A", "B"] if e.startswith("pdb") else ["A", "B", "C", "D", "E", "F", "G", "H"],
              "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
              "vector": [0.0, 0.0, 0.0]
             }]
            }]

            assert len(f.models) == 1
            model = f.model
            assert len(model.chains()) == 2
            assert isinstance(model.chains(), set)
            assert len(model.ligands()) == 4
            assert isinstance(model.ligands(), set)
            assert len(model.waters()) == 180
            assert isinstance(model.waters(), set)
            assert len(model.molecules()) == 186
            assert isinstance(model.molecules(), set)
            assert len(model.residues()) == 418
            assert isinstance(model.residues(), set)
            assert len(model.atoms()) == 3431
            assert isinstance(model.atoms(), set)
            assert len(model.chains(length__gt=200)) == 2
            assert len(model.chains(length__gt=210)) == 1
            assert len(model.ligands(name="XMP")) == 2
            assert len(model.residues(name="VAL")) == 28
            assert len(model.residues(name="CYS")) == 6
            assert len(model.residues(name__regex="CYS|VAL")) == 34
            assert model.mass == pytest.approx(46018.5, abs=0.005)

            chaina = model.chain("A")
            chainb = model.chain(id="B")
            assert chaina.model is model
            assert chainb.model is model
            assert chaina.id == "A"
            assert chaina.length == 204
            assert chainb.length == 214
            assert chaina.sequence.startswith("LRSRRVDVMDVMNRLILAMDL")
            assert chaina.sequence.endswith("LADNPAAAAAGIIESIKDLLIPE")
            assert chainb.sequence.startswith("LRSRRVDVMDVMNRLILAMDL")
            assert chainb.sequence.endswith("LADNPAAAAAGIIESIKDLLIPE")
            for res in chaina:
                assert res in chaina
            assert len(chaina.residues()) == 204
            assert isinstance(chaina.residues(), tuple)
            assert len(chaina.ligands()) == 2
            assert isinstance(chaina.ligands(), set)
            assert len(chaina.atoms()) == 1557
            assert isinstance(chaina.atoms(), set)
            assert len(chainb.atoms()) == 1634
            assert isinstance(chainb.atoms(), set)
            res = chaina.residue("A.13")
            assert res.helix
            assert not res.strand
            res = chaina.residue("A.15")
            assert not res.helix
            assert res.strand
            assert res.chain is chaina
            assert res.model is model
            assert res.name == "LEU"
            assert res.code == "L"
            assert res.full_name == "leucine"
            assert len(res.atoms()) == 8
            assert isinstance(chaina.atoms(), set)
            assert len(res.atoms(element="C")) == 6
            assert len(res.atoms(element__regex="C|O")) == 7
            assert len(res.atoms(name__regex="^CD")) == 2
            assert chaina[0] is chaina.residue("A.11")
            assert res.next is chaina[5]
            assert chaina.residue(name="GLN") in [chaina.residue("A.136"), chaina.residue("A.173")]

            # Source information
            if e == "pdb":
                for chain in (chaina, chainb):
                    assert (
                        chain.information["organism_scientific"]
                        == "METHANOTHERMOBACTER THERMAUTOTROPHICUS STR. DELTA H"
                    )
                    assert (
                        chain.information["molecule"]
                        == "OROTIDINE 5'-MONOPHOSPHATE DECARBOXYLASE"
                    )
                    assert (
                        chain.information["engineered"]
                        == "YES"
                    )

            lig = model.ligand(name="XMP")
            assert lig.model is model
            assert len(lig.atoms()) == 24
            assert lig.formula == {"C": 10, "O": 9, "N": 4, "P": 1}
            assert lig.full_name == "XANTHOSINE-5'-MONOPHOSPHATE"
            lig = model.ligand("A.5001")
            assert lig.model is model
            assert lig.chain is chaina
            assert len(lig.atoms()) == 6
            assert lig.mass == 80.0416
            pairs = list(lig.pairwise_atoms())
            assert len(pairs) == 15
            for pair in pairs:
                pair = list(pair)
                assert 0 < pair[0].distance_to(pair[1]) < 5
            hoh = model.water("A.3005")
            assert hoh.name == "HOH"
            assert lig.model is model
            assert lig.chain is chaina
            lig1, lig2 = model.ligands(name="XMP")
            assert lig1.rmsd_with(lig2) == pytest.approx(0.133, abs=0.001)
            assert lig2.rmsd_with(lig1) == pytest.approx(0.133, abs=0.001)

            atom = model.atom(934)
            assert atom.anisotropy == [0, 0, 0, 0, 0, 0]
            assert atom.element == "C"
            assert atom.name == "CA"
            assert atom.location == (4.534, 53.864, 43.326)
            assert atom.bvalue == 17.14
            assert atom.charge == 0
            assert atom.mass == pytest.approx(12, abs=0.1)
            assert atom.atomic_number == 6
            assert atom.chain is chaina
            assert atom.model is model
            assert atom.het is model.residue("A.131")

            assert model.molecule("A") == chaina
            assert model.molecule("A.5001") == lig
            assert model.molecule("A.3005") == hoh
            assert len(model.molecules(mass__gt=18)) == 6
            assert len(model.molecules(mass__gt=90)) == 4
            assert len(model.molecules(mass__gt=1000)) == 2
            assert len(model.molecules(mass__gt=90, mass__lt=1000)) == 2

            for optimise in [False, True]:
                if optimise:
                    model.optimise_distances()
                atom = model.atom(1587 if e.startswith("pdb") else 1586)
                four_angstrom = atom.nearby_atoms(cutoff=4)
                assert len(four_angstrom) == 10
                assert (
                sorted([atom.id for atom in four_angstrom])
                == [n - (not e.startswith("pdb")) for n in [1576, 1582, 1583, 1584, 1586, 1588, 1589, 1590, 1591, 2957]]
                )
                assert len(atom.nearby_atoms(cutoff=4, element="O")) == 1
                four_angstrom = model.atoms_in_sphere(atom.location, 4)
                assert len(four_angstrom) == 11
                assert (
                sorted([atom.id for atom in four_angstrom])
                == [n - (not e.startswith("pdb")) for n in [1576, 1582, 1583, 1584, 1586, 1587, 1588, 1589, 1590, 1591, 2957]]
                )
                assert len(model.atoms_in_sphere(atom.location, 4, element="O")) == 1
                assert len(model.atoms_in_sphere([10, 20, 30], 40)) == 1281
                assert len(model.atoms_in_sphere([10, 20, 30], 41)) == 1360
                assert len(model.atoms_in_sphere([10, 20, 30], 40, element="C")) == 760
                assert len(model.atoms_in_sphere([10, 20, 30], 39, element="C")) == 711

                atom = model.atom(905)
                assert len(atom.nearby_hets(5)) == 9
                assert len(atom.nearby_hets(5, ligands=False)) == 7
                assert len(atom.nearby_hets(5, het__is_water=False)) == 8
                assert len(atom.nearby_hets(5, residues=False)) == 2
                assert len(atom.nearby_hets(5, element="O")) == 4

            model.dehydrate()
            assert model.waters() == set()


    def test_5xme(self):
        for e in ["cif", "mmtf", "pdb"]:
            f = io.open("tests/io/integration/files/5xme." + e)
            assert f.resolution is None
            models = f.models
            assert len(models) == 10
            assert f.model is f.models[0]
            x_values = [
             33.969, 34.064, 37.369, 36.023, 35.245,
             35.835, 37.525, 35.062, 36.244, 37.677
            ]
            all_atoms = set()
            for x, model in zip(x_values, models):
                assert len(model.atoms()) == 1827
                all_atoms.update(model.atoms())
                atom = model.chain()[0].atom(name="N")
                assert atom.location[0] == x
            assert len(all_atoms) == 18270

            # Source information
            if e == "pdb":
                chain = models[0].chain("A")
                assert (
                    chain.information["organism_scientific"]
                    == "HOMO SAPIENS"
                )
                assert (
                    chain.information["molecule"]
                    == "TUMOR NECROSIS FACTOR RECEPTOR TYPE 1-ASSOCIATED DEATH DOMAIN PROTEIN"
                )
                assert (
                    chain.information["engineered"]
                    == "YES"
                )

    def test_1cbn(self):
        for e in ["cif", "mmtf", "pdb"]:
            f = io.open("tests/io/integration/files/1cbn." + e)
            chain = f.model.chain()
            residue1, residue2, residue3 = chain[:3]
            assert len(residue1.atoms()) == 16
            assert len(residue2.atoms()) == 14
            assert len(residue3.atoms()) == 10
            for residue in chain[:3]:
                for name in ["N", "C", "CA", "CB"]:
                    assert len(residue.atoms(name=name)) == 1


    def test_1xda(self):
        for e in ["cif", "mmtf", "pdb"]:
            f = io.open("tests/io/integration/files/1xda." + e)
            assert len(f.model.atoms()) == 1842
            assert len(f.model.atoms(is_metal=True)) == 4
            assert len(f.model.atoms(is_metal=False)) == 1838

            model = f.model
            assert len(model.atoms()) == 1842
            assert len(model.chains()) == 8
            assert len(model.ligands()) == 16

            model = f.generate_assembly(1)
            assert len(model.chains()) == 2
            assert set([c.id for c in model.chains()]) == {"A", "B"}
            assert len(model.ligands()) == 4

            model = f.generate_assembly(2)
            assert len(model.chains()) == 2
            assert set([c.id for c in model.chains()]) == {"C", "D"}
            assert len(model.ligands()) == 4

            model = f.generate_assembly(3)
            assert len(model.chains()) == 2
            assert set([c.id for c in model.chains()]) == {"E", "F"}
            assert len(model.ligands()) == 4

            model = f.generate_assembly(4)
            assert len(model.chains()) == 2
            assert set([c.id for c in model.chains()]) == {"G", "H"}
            assert len(model.ligands()) == 4

            model = f.generate_assembly(7)
            assert len(model.chains()) == 6
            assert set([c.id for c in model.chains()]) == {"A", "B"}
            assert len(model.ligands()) == 12
            zn = model.atom(element="ZN")
            liganding_residues = zn.nearby_hets(3, is_metal=False, element__ne="CL")
            assert len(liganding_residues) == 3
            assert set([r.id for r in liganding_residues]) == {"B.10"}
            assert set([r.name for r in liganding_residues]) == {"HIS"}
            res1, res2, res3 = liganding_residues

            assert res1.atom(name="N").distance_to(res2.atom(name="N")) > 10
        

    def test_4opj(self):
        for e in ["cif", "mmtf", "pdb"]:
            f = io.open("tests/io/integration/files/4opj." + e)
            if e == "cif":
                assert (
                 f.model.residue("B.6").full_name
                 == "(2R,3aS,4aR,5aR,5bS)-2-(6-amino-9H-purin-9-yl)-3a-hydroxyhexahydrocyclopropa[4,5]cyclopenta[1,2-b]furan-5a(4H)-yl dihydrogen phosphate"
                )
            elif e =="mmtf":
                assert f.model.residue("B.6").full_name == "TCY"
            else:
                assert (
                 f.model.residue("B.6").full_name
                 == "(2R,3AS,4AR,5AR,5BS)-2-(6-AMINO-9H-PURIN-9-YL)-3A-HYDROXYHEXAHYDROCYCLOPROPA[4,5]CYCLOPENTA[1,2-B]FURAN-5A(4H)-YL DIHYDROGEN PHOSPHATE"
                )

            if e == "pdb":
                assert (
                    f.model.chain("A").information["molecule"] == "RIBONUCLEASE H"
                )
                assert (
                    f.model.chain("C").information["molecule"] == "RIBONUCLEASE H"
                )
                assert (
                    f.model.chain("B").information["molecule"]
                    == "5'-D(*CP*GP*CP*GP*AP*(TCY)P*TP*TP*CP*GP*CP*G)-3'"
                )
                assert (
                    f.model.chain("D").information["molecule"]
                    == "5'-D(*CP*GP*CP*GP*AP*(TCY)P*TP*TP*CP*GP*CP*G)-3'"
                )
                assert (
                    f.model.chain("A").information["organism_scientific"]
                    == "BACILLUS HALODURANS"
                )
                assert "organism_scientific" not in f.model.chain("B").information

    def test_6xlu(self):
        # Branched chains
        for e in ["cif", "mmtf", "pdb"]:
            f = io.open("tests/io/integration/files/6xlu." + e)
            assert len(f.model.chains()) == (3 if e == "pdb" else 18)
            assert len(f.model.ligands()) == (62 if e == "pdb" else 32)
    
