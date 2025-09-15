import pytest
from biomod.io import io
import os


@pytest.fixture
def file_saving_fixture():
    files_at_start = os.listdir("tests/io/integration/files")
    yield
    files_at_end = os.listdir("tests/io/integration/files")
    to_remove = [f for f in files_at_end if f not in files_at_start]
    for f in to_remove:
        os.remove("tests/io/integration/files/" + f)

def check_file_saving(filename):
    """Round-trip a file: open -> save -> reopen -> compare structure."""
    f = io.open("tests/io/integration/files/" + filename)
    f.model.save("tests/io/integration/files/saved_" + filename)
    f2 = io.open("tests/io/integration/files/saved_" + filename)

    # Compare chains
    assert len(f.model.chains()) == len(f2.model.chains())
    for chain1, chain2 in zip(
        sorted(f.model.chains(), key=lambda c: c.id),
        sorted(f2.model.chains(), key=lambda c: c.id),
    ):
        assert chain1.id == chain2.id
        assert chain1.sequence == chain2.sequence

        # Compare residues
        assert len(chain1.residues()) == len(chain2.residues())
        for res1, res2 in zip(
            sorted(chain1.residues(), key=lambda r: r.id),
            sorted(chain2.residues(), key=lambda r: r.id),
        ):
            assert res1.id == res2.id
            assert res1.name == res2.name

    # Compare ligands
    assert len(f.model.ligands()) == len(f2.model.ligands())
    for lig1, lig2 in zip(
        sorted(f.model.ligands(), key=lambda l: l.id),
        sorted(f2.model.ligands(), key=lambda l: l.id),
    ):
        assert lig1.id == lig2.id
        assert lig1.name == lig2.name


@pytest.mark.skip(reason="Temporarily skipping to focus on other tests")
class TestMmcifFileSaving:
    def test_can_save_1lol(self, file_saving_fixture):
        check_file_saving("1lol.cif")

    def test_can_save_1cbn(self, file_saving_fixture):
        check_file_saving("1cbn.cif")

    def test_can_save_1m4x(self, file_saving_fixture):
        check_file_saving("1m4x.cif")

    def test_can_save_1xda(self, file_saving_fixture):
        check_file_saving("1xda.cif")

    def test_can_save_5xme(self, file_saving_fixture):
        check_file_saving("5xme.cif")

    def test_can_save_4y60(self, file_saving_fixture):
        check_file_saving("4y60.cif")

    def test_chain(self, file_saving_fixture):
        f = io.open("tests/io/integration/files/1lol.cif")
        f.model.chain("A").save("tests/io/integration/files/chaina.cif")
        chain = io.open("tests/io/integration/files/chaina.cif").model
        assert f.model.chain("A").id == chain.id
        assert f.model.chain("A").sequence == chain.sequence

    def test_biological_assembly_warns_on_saving(self, file_saving_fixture):
        f = io.open("tests/io/integration/files/1xda.cif")
        model = f.generate_assembly(5)
        with pytest.warns(Warning):
            model.save("tests/io/integration/files/assembly.cif")


class TestMmtfFileSaving:
    def test_can_save_1lol(self, file_saving_fixture):
        check_file_saving("1lol.mmtf")

    def test_can_save_1cbn(self, file_saving_fixture):
        check_file_saving("1cbn.mmtf")

    def test_can_save_1m4x(self, file_saving_fixture):
        check_file_saving("1m4x.mmtf")

    def test_can_save_1xda(self, file_saving_fixture):
        check_file_saving("1xda.mmtf")

    def test_can_save_5xme(self, file_saving_fixture):
        check_file_saving("5xme.mmtf")

    def test_can_save_4y60(self, file_saving_fixture):
        check_file_saving("4y60.mmtf")

    def test_chain(self, file_saving_fixture):
        f = io.open("tests/io/integration/files/1lol.mmtf")
        f.model.chain("A").save("tests/io/integration/files/chaina.mmtf")
        chain = io.open("tests/io/integration/files/chaina.mmtf").model
        assert f.model.chain("A").id == chain.id
        assert f.model.chain("A").sequence == chain.sequence

    def test_biological_assembly_warns_on_saving(self, file_saving_fixture):
        f = io.open("tests/io/integration/files/1xda.cif")
        model = f.generate_assembly(5)
        with pytest.warns(Warning):
            model.save("tests/io/integration/files/assembly.cif")


class TestPdbFileSaving:
    def test_can_save_1lol(self, file_saving_fixture):
        check_file_saving("1lol.pdb")

    def test_can_save_1cbn(self, file_saving_fixture):
        check_file_saving("1cbn.pdb")

    def test_can_save_1m4x(self, file_saving_fixture):
        check_file_saving("1m4x.pdb")

    def test_can_save_1xda(self, file_saving_fixture):
        check_file_saving("1xda.pdb")

    def test_can_save_5xme(self, file_saving_fixture):
        check_file_saving("5xme.pdb")

    def test_can_save_4y60(self, file_saving_fixture):
        check_file_saving("4y60.pdb")

    def test_can_save_1d5t(self, file_saving_fixture):
        check_file_saving("1d5t.pdb")

    def test_can_save_1grm(self, file_saving_fixture):
        check_file_saving("1grm.pdb")
        f = io.open("tests/io/integration/files/1grm.pdb")
        f.model.save("tests/io/integration/files/saved_1grm.pdb")
        with open("tests/io/integration/files/1grm.pdb") as f_old:
            old_text = f_old.read()
            old_text = old_text[: old_text.find("ENDMDL")]
            old_het_count = old_text.count("HETATM")
        with open("tests/io/integration/files/saved_1grm.pdb") as f_new:
            new_het_count = f_new.read().count("HETATM")
        assert old_het_count == new_het_count

    def test_chain(self, file_saving_fixture):
        f = io.open("tests/io/integration/files/1lol.pdb")
        f.model.chain("A").save("tests/io/integration/files/chaina.pdb")
        chain = io.open("tests/io/integration/files/chaina.pdb").model
        assert f.model.chain("A").id == chain.id
        assert f.model.chain("A").sequence == chain.sequence

    def test_biological_assembly_warns_on_saving(self, file_saving_fixture):
        f = io.open("tests/io/integration/files/1xda.pdb")
        model = f.generate_assembly(5)
        with pytest.warns(Warning):
            model.save("tests/io/integration/files/assembly.pdb")