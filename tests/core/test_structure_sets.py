import pytest
from unittest.mock import Mock
from biomod.core.base import StructureSet

@pytest.fixture
def structures():
    return [Mock(_id=1), Mock(_id=2), Mock(_id=2)]


def test_can_create_structure_set(structures):
    s = StructureSet(*structures)
    assert s._d == {1: {structures[0]}, 2: set(structures[1:])}


def test_can_add_structure_sets(structures):
    s1 = StructureSet(*structures[:2])
    s2 = StructureSet(structures[2])
    s = s1 + s2
    assert s._d == {1: {structures[0]}, 2: set(structures[1:])}


def test_can_get_structure_set_len(structures):
    s = StructureSet(*structures)
    assert len(s) == 3


def test_can_get_ids(structures):
    s = StructureSet(*structures)
    assert s.ids == {1, 2}


def test_can_get_structures(structures):
    s = StructureSet(*structures)
    assert set(s.structures) == set(structures)


def test_can_get_structure_by_id(structures):
    s = StructureSet(*structures)
    assert s.get(1) == structures[0]
    assert s.get(2) in structures[1:]