"""
Tests for H-bond calculation.
"""

import gzip
from unittest.mock import Mock
import pytest
import numpy as np
from biomod.io import io
from biomod.utilities.hbond import calculate_h_bonds, calculate_h_bond_energy, assign_hydrogen_to_residues
from biomod.config import MIN_HBOND_ENERGY


def parse_reference_dssp(filepath):
    """
    Parses a reference DSSP file in mmCIF format to extract H-bond information.
    """
    # pylint: disable=too-many-branches
    hbonds = {}
    in_loop = False
    columns = []
    col_indices = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Detect start of the bridge_pairs loop
            if line.startswith("loop_"):
                in_loop = False  # reset
                columns = []
                continue

            if line.startswith("_dssp_struct_bridge_pairs."):
                in_loop = True
                columns.append(line.replace("_dssp_struct_bridge_pairs.", ""))
                continue

            # If we are inside the loop and see a new header or section, stop
            if in_loop and (line.startswith("_") or line.startswith("loop_") or line.startswith("data_") or line.startswith("#")):  # pylint: disable=line-too-long
                in_loop = False
                continue

            # Parse data rows
            if in_loop:
                if not col_indices:
                    col_indices = {col: i for i, col in enumerate(columns)}

                fields = line.split()
                if len(fields) < len(columns):
                    continue

                try:
                    res_num = int(fields[col_indices['label_seq_id']])
                except (ValueError, KeyError, IndexError):
                    continue

                if res_num not in hbonds:
                    hbonds[res_num] = {'donor': [], 'acceptor': []}

                # acceptors
                for acc, ene in [
                    ('acceptor_1_label_seq_id', 'acceptor_1_energy'),
                    ('acceptor_2_label_seq_id', 'acceptor_2_energy'),
                ]:
                    try:
                        partner = fields[col_indices[acc]]
                        energy = fields[col_indices[ene]]
                        if partner != '?' and energy != '?':
                            offset = int(partner) - res_num
                            hbonds[res_num]['acceptor'].append({
                                'offset': offset,
                                'energy': float(energy)
                            })
                    except (ValueError, KeyError, IndexError):
                        pass

                # donors
                for don, ene in [
                    ('donor_1_label_seq_id', 'donor_1_energy'),
                    ('donor_2_label_seq_id', 'donor_2_energy'),
                ]:
                    try:
                        partner = fields[col_indices[don]]
                        energy = fields[col_indices[ene]]
                        if partner != '?' and energy != '?':
                            offset = int(partner) - res_num
                            hbonds[res_num]['donor'].append({
                                'offset': offset,
                                'energy': float(energy)
                            })
                    except (ValueError, KeyError, IndexError):
                        pass

    return hbonds

def test_calculate_h_bonds_comparative():
    """
    Tests the calculate_h_bonds function by comparing its output to a
    reference DSSP file.
    """
    # 1. Run dsspy's H-bond calculation
    f = io.open('tests/reference_data/1cbs.cif.gz')
    residues = sorted(list(f.model.residues()), key=lambda r: int(r.id.split('.')[-1]))
    calculate_h_bonds(residues)

    # 2. Parse the reference DSSP file
    reference_hbonds = parse_reference_dssp(
        'tests/reference_data/1cbs-dssp.cif')

    # 3. Compare the results
    assert len(residues) == len(reference_hbonds)

    for i, res in enumerate(residues, 1):
        # Make sure our enumeration index matches the residue's internal number
        assert i == res.number

        ref_bonds = reference_hbonds.get(i)
        if not ref_bonds:
            continue

        # Compare donor bonds (N-H --> O)
        # NOTE: hbond_donor is actually where `res` is the DONOR in our build.
        dsspy_donors = sorted([
            {'offset': h.residue.number - res.number, 'energy': h.energy}
            for h in res.hbond_donor if h.residue is not None
        ], key=lambda x: x['offset'])

        ref_donors = sorted(ref_bonds['donor'], key=lambda x: x['offset'])

        if len(dsspy_donors) != len(ref_donors):
            print(f"Residue {i}: Mismatch in number of donor bonds")
            print(f"  dsspy: {dsspy_donors}")
            print(f"  ref:   {ref_donors}")
        assert len(dsspy_donors) == len(ref_donors)

        for dsspy_d, ref_d in zip(dsspy_donors, ref_donors):
            if dsspy_d['offset'] != ref_d['offset']:
                print(f"Residue {i}: Mismatch in donor bond offset")
                print(f"  dsspy: {dsspy_d}")
                print(f"  ref:   {ref_d}")
            assert dsspy_d['offset'] == ref_d['offset']
            assert dsspy_d['energy'] == pytest.approx(ref_d['energy'], abs=1e-1)

            # Compare acceptor bonds (O --> H-N)
            # NOTE: hbond_acceptor is actually where `res` is the ACCEPTOR in our build.
            dsspy_acceptors = sorted([
                {'offset': h.residue.number - res.number, 'energy': h.energy}
                for h in res.hbond_acceptor if h.residue is not None
            ], key=lambda x: x['offset'])

        ref_acceptors = sorted(ref_bonds['acceptor'], key=lambda x: x['offset'])

        if len(dsspy_acceptors) != len(ref_acceptors):
            print(f"Residue {i}: Mismatch in number of acceptor bonds")
            print(f"  dsspy: {dsspy_acceptors}")
            print(f"  ref:   {ref_acceptors}")
        assert len(dsspy_acceptors) == len(ref_acceptors)

        for dsspy_a, ref_a in zip(dsspy_acceptors, ref_acceptors):
            if dsspy_a['offset'] != ref_a['offset']:
                print(f"Residue {i}: Mismatch in acceptor bond offset")
                print(f"  dsspy: {dsspy_a}")
                print(f"  ref:   {ref_a}")
            assert dsspy_a['offset'] == ref_a['offset']
            assert dsspy_a['energy'] == pytest.approx(ref_a['energy'], abs=1e-1)

from biomod.core.residues import Residue
from biomod.core.atoms import Atom

def test_calculate_h_bond_energy_minimal_distance():
    """
    Tests that the H-bond energy is set to MIN_HBOND_ENERGY when atoms are too close.
    """
    donor_n = Atom("N", 0, 0, 1, 1, "N", 0, 0, [])
    donor_h = Atom("H", 0, 0, 0, 2, "H", 0, 0, [])
    donor = Residue(donor_n, donor_h, name="ALA", id="A1")
    donor.h_coord = np.array([0.0, 0.0, 0.0])

    acceptor_o = Atom("O", 0, 0, 0.1, 3, "O", 0, 0, [])
    acceptor_c = Atom("C", 0, 1, 0.1, 4, "C", 0, 0, [])
    acceptor = Residue(acceptor_o, acceptor_c, name="ALA", id="A2")

    energy = calculate_h_bond_energy(donor, acceptor)
    assert energy == MIN_HBOND_ENERGY

def test_assign_hydrogen_to_residues():
    """
    Tests the assign_hydrogen_to_residues function.
    """
    res1_n = Atom("N", 0, 0, 0, 1, "N", 0, 0, [])
    res1_c = Atom("C", 1, 0, 0, 2, "C", 0, 0, [])
    res1_o = Atom("O", 1, 1, 0, 3, "O", 0, 0, [])
    res1 = Residue(res1_n, res1_c, res1_o, name="ALA", id="A1")

    res2_n = Atom("N", 2, 0, 0, 4, "N", 0, 0, [])
    res2 = Residue(res2_n, name="ALA", id="A2")

    residues = [res1, res2]
    assign_hydrogen_to_residues(residues)

    # Check res1 (no previous residue)
    np.testing.assert_array_equal(res1.h_coord, np.array(res1_n.location))

    # Check res2
    expected_h_coord = np.array(res2_n.location) + (np.array(res1_c.location) - np.array(res1_o.location))
    np.testing.assert_allclose(res2.h_coord, expected_h_coord)
