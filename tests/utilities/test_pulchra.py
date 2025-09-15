import subprocess
import pytest
from pathlib import Path

TEST_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = TEST_FILE_PATH.parent.parent.parent

GOLDEN_OUTPUTS_DIR = PROJECT_ROOT / "tests/reference_data"


def test_multichain_output(tmp_path):
    # Setup paths for the multichain test
    input_pdb = PROJECT_ROOT / "tests/reference_data/7laf_ca.pdb"
    output_pdb = tmp_path / "7laf_ca.rebuilt.pdb"
    golden_output_path = GOLDEN_OUTPUTS_DIR / "7laf_ca.rebuilt.pdb"

    # Construct the command
    command = ["python", str(PROJECT_ROOT / "biomod/utilities/rebuild/main.py"), str(input_pdb)]

    # Run the command
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    with open(output_pdb, "w") as f:
        f.write(result.stdout)

    # Read the generated output file
    assert output_pdb.exists(), f"Output file {output_pdb} was not generated."
    generated_output = output_pdb.read_text()

    # Read the golden output file
    assert golden_output_path.exists(), f"Golden file {golden_output_path} does not exist."
    golden_output = golden_output_path.read_text()

    # Compare the outputs
    assert generated_output == golden_output, "Multichain output does not match golden file."


def test_python_pulchra_hydrogens(tmp_path):
    """
    Tests the Python version of Pulchra with hydrogen generation.
    """
    input_pdb = PROJECT_ROOT / "tests/reference_data/7laf.pdb"
    output_pdb = tmp_path / "7laf.rebuilt.pdb"

    # Construct the command
    command = ["python", str(PROJECT_ROOT / "biomod/utilities/rebuild/main.py"), "--add-hydrogens", str(input_pdb)]

    # Run the command
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    with open(output_pdb, "w") as f:
        f.write(result.stdout)

    # Read the generated output file
    assert output_pdb.exists(), f"Output file {output_pdb} was not generated."
    generated_output = output_pdb.read_text()

    # Check for hydrogens
    assert "H" in generated_output
