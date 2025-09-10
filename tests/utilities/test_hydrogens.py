import numpy as np
from pulchra.hydrogens import get_hydrogen_positions

def test_get_hydrogen_positions_serine():
    """
    Tests the get_hydrogen_positions function for Serine.
    """
    # Create a dummy Serine residue
    residue_name = "SER"
    heavy_atoms = {
        'N': np.array([0.0, 0.0, 0.0]),
        'CA': np.array([1.45, 0.0, 0.0]),
        'C': np.array([2.0, 1.0, 0.0]),
        'CB': np.array([1.9, -1.4, 0.0]),
        'OG': np.array([3.3, -1.6, 0.0]),
    }
    prev_c = np.array([-1.0, 1.0, 0.0])

    # Get the hydrogen positions
    hydrogens = get_hydrogen_positions(residue_name, heavy_atoms, prev_c=prev_c)

    # Check that the hydrogens are there
    assert 'H' in hydrogens
    assert 'HG' in hydrogens

    # Check the coordinates of the amide hydrogen (simple check)
    assert np.allclose(np.linalg.norm(hydrogens['H'] - heavy_atoms['N']), 1.01, atol=0.01)

    # Check the coordinates of the hydroxyl hydrogen (simple check)
    assert np.allclose(np.linalg.norm(hydrogens['HG'] - heavy_atoms['OG']), 0.96, atol=0.01)
