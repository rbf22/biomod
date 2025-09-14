import pytest
from biomod.io import io
import os
from unittest import TestCase
import signal


class SavingTestDebug(TestCase):

    def setUp(self):
        self.files_at_start = os.listdir("tests/io/integration/files")

    def tearDown(self):
        files_at_end = os.listdir("tests/io/integration/files")
        to_remove = [f for f in files_at_end if f not in self.files_at_start]
        for f in to_remove:
            os.remove("tests/io/integration/files/" + f)

    def check_file_saving(self, filename):
        """Open -> save -> reopen -> compare structure fields with full atom-level debug."""
        path = "tests/io/integration/files/" + filename
        print(f"\n=== DEBUG: Opening {path} ===")
        f = io.open(path)

        chains = f.model.chains()
        ligands = f.model.ligands()
        waters = getattr(f.model, 'waters', [])

        print(f"[DEBUG] Model opened. Chains: {[c.id for c in chains]}")
        print(f"[DEBUG] Residues per chain: {[len(c.residues()) for c in chains]}")
        print(f"[DEBUG] Number of ligands: {len(ligands)}")
        print(f"[DEBUG] Number of waters: {len(waters)}")

        for c in chains:
            print(f"[BEFORE] Chain {c.id}: {len(c.residues())} residues, {len(c.atoms())} atoms")
        for l in ligands:
            print(f"[BEFORE] Ligand {l.id}: {len(l.atoms())} atoms")
        for w in waters:
            print(f"[BEFORE] Water {w.id}: {len(w.atoms())} atoms")

        # Save path
        save_path = "tests/io/integration/files/saved_" + filename
        print(f"[DEBUG] Saving to {save_path}...")
        f.model.save(save_path)

        if os.path.exists(save_path):
            size = os.path.getsize(save_path)
            print(f"[DEBUG] File saved, size: {size} bytes")
        else:
            print("[ERROR] File not found after save!")

        # Reopen
        print("[DEBUG] Reopening saved file...")
        f2 = io.open(save_path)
        chains2 = f2.model.chains()
        ligands2 = f2.model.ligands()
        waters2 = getattr(f2.model, 'waters', [])

        print(f"[DEBUG] Reopened. Chains: {[c.id for c in chains2]}")
        print(f"[DEBUG] Residues per chain: {[len(c.residues()) for c in chains2]}")
        print(f"[DEBUG] Number of ligands: {len(ligands2)}")
        print(f"[DEBUG] Number of waters: {len(waters2)}")

        for c in chains2:
            print(f"[AFTER]  Chain {c.id}: {len(c.residues())} residues, {len(c.atoms())} atoms")
        for l in ligands2:
            print(f"[AFTER]  Ligand {l.id}: {len(l.atoms())} atoms")
        for w in waters2:
            print(f"[AFTER]  Water {w.id}: {len(w.atoms())} atoms")

        # Compare chains
        assert len(chains) == len(chains2)
        for chain1, chain2 in zip(sorted(chains, key=lambda c: c.id),
                                  sorted(chains2, key=lambda c: c.id)):
            print(f"[DEBUG] Comparing chain {chain1.id}")
            assert chain1.id == chain2.id
            assert chain1.sequence == chain2.sequence

            # Compare residues and atoms
            res1_list = sorted(chain1.residues(), key=lambda r: r.id)
            res2_list = sorted(chain2.residues(), key=lambda r: r.id)
            assert len(res1_list) == len(res2_list)
            for res1, res2 in zip(res1_list, res2_list):
                print(f"[DEBUG] Comparing residue {res1.id} ({res1.name}) in chain {chain1.id}")
                assert res1.id == res2.id
                assert res1.name == res2.name

                atoms1 = sorted(res1.atoms(), key=lambda a: a.id)
                atoms2 = sorted(res2.atoms(), key=lambda a: a.id)
                assert len(atoms1) == len(atoms2)
                for a1, a2 in zip(atoms1, atoms2):
                    print(f"    [DEBUG] Atom {a1.id}: {a1.name} ({a1.element}) -> {a2.id}: {a2.name} ({a2.element})")
                    assert a1.id == a2.id
                    assert a1.name == a2.name
                    assert a1.element == a2.element

        # Compare ligands
        assert len(ligands) == len(ligands2)
        for lig1, lig2 in zip(sorted(ligands, key=lambda l: l.id),
                              sorted(ligands2, key=lambda l: l.id)):
            print(f"[DEBUG] Comparing ligand {lig1.id} ({lig1.name})")
            assert lig1.id == lig2.id
            assert lig1.name == lig2.name

            atoms1 = sorted(lig1.atoms(), key=lambda a: a.id)
            atoms2 = sorted(lig2.atoms(), key=lambda a: a.id)
            assert len(atoms1) == len(atoms2)
            for a1, a2 in zip(atoms1, atoms2):
                print(f"    [DEBUG] Ligand atom {a1.id}: {a1.name} ({a1.element}) -> {a2.id}: {a2.name} ({a2.element})")
                assert a1.id == a2.id
                assert a1.name == a2.name
                assert a1.element == a2.element

        # Compare waters
        assert len(waters) == len(waters2)
        for w1, w2 in zip(sorted(waters, key=lambda w: w.id),
                          sorted(waters2, key=lambda w: w.id)):
            print(f"[DEBUG] Comparing water {w1.id}")
            atoms1 = sorted(w1.atoms(), key=lambda a: a.id)
            atoms2 = sorted(w2.atoms(), key=lambda a: a.id)
            assert len(atoms1) == len(atoms2)
            for a1, a2 in zip(atoms1, atoms2):
                print(f"    [DEBUG] Water atom {a1.id}: {a1.name} ({a1.element}) -> {a2.id}: {a2.name} ({a2.element})")
                assert a1.id == a2.id
                assert a1.name == a2.name
                assert a1.element == a2.element


class DebugMmtfTests(SavingTestDebug):

    def test_can_save_1lol_mmtf(self):
        self.check_file_saving("1lol.mmtf")

    def test_can_save_1xda_mmtf(self):
        self.check_file_saving("1xda.mmtf")

    @pytest.mark.skip("Skipping assembly save to avoid MMTF hang")
    def test_biological_assembly_warns_on_saving(self):
        f = io.open("tests/io/integration/files/1xda.mmtf")
        print("Generating assembly...")

        def timeout_handler(signum, frame):
            raise TimeoutError("generate_assembly or save took too long!")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        try:
            model = f.generate_assembly(5)
            print("Assembly generated, saving...")
            model.save("tests/io/integration/files/tmp_assembly.mmtf")
            print("Assembly save complete.")
        finally:
            signal.alarm(0)


class DebugPdbTests(SavingTestDebug):

    def test_can_save_1lol_pdb(self):
        self.check_file_saving("1lol.pdb")

    def test_chain_save_debug(self):
        f = io.open("tests/io/integration/files/1lol.pdb")
        print("[DEBUG] Saving chain A only...")
        f.model.chain("A").save("tests/io/integration/files/chainA.pdb")
        chain = io.open("tests/io/integration/files/chainA.pdb").model
        print(f"[DEBUG] Chain reopened. ID: {chain.chains()[0].id}")
        assert f.model.chain("A").id == chain.chains()[0].id
        assert f.model.chain("A").sequence == chain.chains()[0].sequence