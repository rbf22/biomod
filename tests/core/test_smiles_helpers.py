# -*- coding: utf-8 -*-
# Copyright 2018 Peter C Kroon
# Licensed under the Apache License, Version 2.0

import pytest

from biomod.io.smiles import (
    fill_valence, remove_explicit_hydrogens,
    add_explicit_hydrogens, correct_aromatic_rings,
    valence,
)
from tests.testhelper import assertEqualGraphs, make_mol

# --------------------------------------------------------------------
# Helpers for building nodes and edges
# --------------------------------------------------------------------

def C(**kwargs):
    return {"element": "C", **kwargs}

def H(**kwargs):
    # Default H has aromatic=False, charge=0
    return {"element": "H", "aromatic": False, "charge": 0, **kwargs}

def bond(i, j, order=1):
    return (i, j, {"order": order})


# --------------------------------------------------------------------
# Parametrized cases
# --------------------------------------------------------------------

hydrogen_cases = [
    # Add hydrogens
    (add_explicit_hydrogens, {}, [(0, C())], [], [(0, C())], []),
    (
        add_explicit_hydrogens, {},
        [(0, C(hcount=2))], [],
        [(0, C()), (1, H()), (2, H())],
        [bond(0, 1), bond(0, 2)],
    ),
    (
        add_explicit_hydrogens, {},
        [(0, C(hcount=2)), (1, C(hcount=2))],
        [bond(0, 1, 2)],
        [(0, C()), (1, H()), (2, H()), (3, C()), (4, H()), (5, H())],
        [bond(0, 1), bond(0, 2), bond(3, 4), bond(3, 5), bond(0, 3, 2)],
    ),

    # Remove hydrogens
    (
        remove_explicit_hydrogens, {},
        [(0, C()), (1, H()), (2, H()), (3, C()), (4, H()), (5, H()), (6, C())],
        [bond(0, 1), bond(0, 2), bond(3, 4), bond(3, 5), bond(0, 3, 2), bond(3, 6)],
        [(0, C(hcount=2)), (1, C(hcount=2)), (2, C(hcount=0))],
        [bond(0, 1, 2), bond(1, 2)],
    ),
    (
        remove_explicit_hydrogens, {},
        [(0, H()), (1, H())],
        [bond(0, 1)],
        [(0, H(hcount=0)), (1, H(hcount=0))],
        [bond(0, 1)],
    ),
]

valence_cases = [
    (
        fill_valence,
        {"respect_hcount": True, "respect_bond_order": True, "max_bond_order": 3},
        [(0, C()), (1, C())],
        [bond(0, 1)],
        [(0, C(hcount=3)), (1, C(hcount=3))],
        [bond(0, 1)],
    ),
    (
        fill_valence,
        {"respect_hcount": False, "respect_bond_order": True, "max_bond_order": 3},
        [(0, C(hcount=1)), (1, C())],
        [bond(0, 1)],
        [(0, C(hcount=3)), (1, C(hcount=3))],
        [bond(0, 1)],
    ),
]

aromatic_cases = [
    (
        correct_aromatic_rings, {},
        [(0, C(hcount=1, charge=0)), (1, C(hcount=1, charge=0)),
         (2, C(hcount=1, charge=0)), (3, C(hcount=0, charge=0)), (4, C(hcount=3, charge=0))],
        [bond(0, 1, 2), bond(1, 2), bond(2, 3, 2), bond(3, 0), bond(3, 4)],
        [(0, C(hcount=1, charge=0, aromatic=True)),
         (1, C(hcount=1, charge=0, aromatic=True)),
         (2, C(hcount=1, charge=0, aromatic=True)),
         (3, C(hcount=0, charge=0, aromatic=True)),
         (4, C(hcount=3, charge=0, aromatic=False))],
        [bond(0, 1, 1.5), bond(1, 2, 1.5), bond(2, 3, 1.5), bond(3, 0, 1.5), bond(3, 4)],
    ),
]


# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------

@pytest.mark.parametrize(
    "helper, kwargs, n_data_in, e_data_in, n_data_out, e_data_out",
    hydrogen_cases + valence_cases + aromatic_cases,
)
def test_helper(helper, kwargs, n_data_in, e_data_in, n_data_out, e_data_out):
    mol = make_mol(n_data_in, e_data_in)
    helper(mol, **(kwargs or {}))
    ref_mol = make_mol(n_data_out, e_data_out)
    assertEqualGraphs(mol, ref_mol)


@pytest.mark.parametrize("atom, expected", [
    ({"element": "C"}, [4]),
    ({"element": "N"}, [3, 5]),
    ({"element": "S"}, [2, 4, 6]),
    ({"element": "P"}, [3, 5]),
    ({"element": "N", "charge": 1}, [4]),
    ({"element": "B", "charge": 1}, [2]),
])
def test_valence(atom, expected):
    found = valence(atom)
    assert found == expected


@pytest.mark.parametrize("atom", [
    {"element": "H", "charge": 5},
    {"element": "H", "charge": -10000},
])
def test_valence_error(atom):
    with pytest.raises(ValueError):
        valence(atom)