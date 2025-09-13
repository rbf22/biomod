"""
Tests for the geometry module.
"""

import numpy as np
import pytest
from biomod.utilities.geometry import dihedral_angle


@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide")
def test_dihedral_angle_trans():
    """
    Tests the dihedral_angle function with a trans configuration (180 degrees).
    """
    p1 = np.array([0, 1, 0])
    p2 = np.array([0, 0, 0])
    p3 = np.array([1, 0, 0])
    p4 = np.array([1, -1, 0])

    # The angle should be 180 degrees for a trans configuration
    expected_angle = 180.0

    # Calculate angle using dsspy's function
    dsspy_angle = dihedral_angle(p1, p2, p3, p4)

    assert abs(dsspy_angle) == pytest.approx(expected_angle, abs=1e-3)


@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide")
def test_dihedral_angle_cis():
    """
    Tests the dihedral_angle function with a cis configuration (0 degrees).
    """
    p1 = np.array([0, 1, 0])
    p2 = np.array([0, 0, 0])
    p3 = np.array([1, 0, 0])
    p4 = np.array([1, 1, 0])

    # The angle should be 0 degrees for a cis configuration
    expected_angle = 0.0

    # Calculate angle using dsspy's function
    dsspy_angle = dihedral_angle(p1, p2, p3, p4)

    assert dsspy_angle == pytest.approx(expected_angle, abs=1e-3)

def test_dihedral_angle_gauche():
    """
    Tests the dihedral_angle function with a gauche configuration (+60 degrees).
    """
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 0, 0])
    p3 = np.array([0, 0, 1])
    p4 = np.array([0.5, 0.866, 1])

    # The angle should be 60 degrees for a gauche+ configuration
    expected_angle = 60.0

    # Calculate angle using dsspy's function
    dsspy_angle = dihedral_angle(p1, p2, p3, p4)

    assert dsspy_angle == pytest.approx(expected_angle, abs=1e-3)
