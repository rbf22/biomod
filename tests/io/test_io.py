"""
Tests for the io module.
"""
from biomod.io import io
from biomod.io.builder import File

def test_read_pdb():
    """
    Tests reading a PDB file.
    """
    # The original project has test files in the test/ directory.
    pdb_file_path = "tests/reference_data/pdb1cbs.ent.gz"

    f = io.open(pdb_file_path)

    assert isinstance(f, File)
    assert f.model is not None


def test_read_cif():
    """
    Tests reading an mmCIF file.
    """
    cif_file_path = "tests/reference_data/1cbs.cif.gz"

    f = io.open(cif_file_path)

    assert isinstance(f, File)
    assert f.model is not None

def test_read_pdb_incomplete_residue(tmp_path):
    """
    Tests that residues with missing backbone atoms are skipped.
    """
    pdb_content = """
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   1.000   1.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   2.000   1.000  1.00  0.00           C
"""
    temp_pdb = tmp_path / "temp.pdb"
    with open(temp_pdb, "w") as f:
        f.write(pdb_content)

    f = io.open(temp_pdb)
    assert len(f.model.residues()) == 1

def test_read_pdb_chain_gap(tmp_path):
    """
    Tests that a gap in the chain is correctly identified.
    """
    pdb_content = """
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   1.000   1.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   2.000   1.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   2.000   1.000  1.00  0.00           O
TER
ATOM      5  N   ALA A   2      10.000  10.000  10.000  1.00  0.00           N
ATOM      6  CA  ALA A   2      11.000  10.000  10.000  1.00  0.00           C
ATOM      7  C   ALA A   2      11.000  11.000  10.000  1.00  0.00           C
ATOM      8  O   ALA A   2      12.000  11.000  10.000  1.00  0.00           O
"""
    temp_pdb = tmp_path / "temp.pdb"
    with open(temp_pdb, "w") as f:
        f.write(pdb_content)

    f = io.open(temp_pdb)
    assert len(f.model.residues()) == 2
