import subprocess
import pytest
from pathlib import Path

TEST_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = TEST_FILE_PATH.parent.parent

PULCHRA_EXECUTABLE = PROJECT_ROOT / "c_legacy/bin/pulchra"
INPUT_PDB = PROJECT_ROOT / "c_legacy/examples/model.pdb"
OUTPUT_PDB = PROJECT_ROOT / "c_legacy/examples/model.rebuilt.pdb"
GOLDEN_OUTPUTS_DIR = PROJECT_ROOT / "tests/golden_outputs"

TEST_CASES = [
    ([], "model.rebuilt.default.pdb"),
    (["-c"], "model.rebuilt.no_ca_opt.pdb"),
    (["-p"], "model.rebuilt.cis_prolin.pdb"),
    (["-e"], "model.rebuilt.amber_backbone.pdb"),
    (["-b"], "model.rebuilt.no_backbone.pdb"),
    (["-q"], "model.rebuilt.h_bond_opt.pdb"),
    (["-s"], "model.rebuilt.no_side_chains.pdb"),
    (["-o"], "model.rebuilt.no_exvol_fix.pdb"),
    (["-z"], "model.rebuilt.no_chirality_check.pdb"),
    (["-h"], "model.rebuilt.with_hydrogens.pdb"),
]

@pytest.mark.parametrize("flags, golden_file_name", TEST_CASES)
def test_pulchra_output(flags, golden_file_name):
    # Clean up any previous output file to ensure a clean run
    if OUTPUT_PDB.exists():
        OUTPUT_PDB.unlink()

    # Construct the command
    command = [str(PULCHRA_EXECUTABLE)] + flags + [str(INPUT_PDB)]

    # Run the command
    subprocess.run(command, check=True, capture_output=True, text=True)

    # Read the generated output file
    assert OUTPUT_PDB.exists(), f"Output file {OUTPUT_PDB} was not generated."
    generated_output = OUTPUT_PDB.read_text()

    # Read the golden output file
    golden_output_path = GOLDEN_OUTPUTS_DIR / golden_file_name
    assert golden_output_path.exists(), f"Golden file {golden_output_path} does not exist."
    golden_output = golden_output_path.read_text()

    # Compare the outputs
    assert generated_output == golden_output, f"Output for flags {flags} does not match golden file {golden_file_name}."

    # Clean up the generated file
    if OUTPUT_PDB.exists():
        OUTPUT_PDB.unlink()


def test_multichain_output():
    # Setup paths for the multichain test
    input_pdb = PROJECT_ROOT / "tests/7laf_ca.pdb"
    output_pdb = PROJECT_ROOT / "tests/7laf_ca.rebuilt.pdb"
    golden_output_path = GOLDEN_OUTPUTS_DIR / "7laf_ca.rebuilt.pdb"

    # Clean up any previous output file
    if output_pdb.exists():
        output_pdb.unlink()

    # Construct the command
    command = [str(PULCHRA_EXECUTABLE), str(input_pdb)]

    # Run the command
    subprocess.run(command, check=True, capture_output=True, text=True)

    # Read the generated output file
    assert output_pdb.exists(), f"Output file {output_pdb} was not generated."
    generated_output = output_pdb.read_text()

    # Read the golden output file
    assert golden_output_path.exists(), f"Golden file {golden_output_path} does not exist."
    golden_output = golden_output_path.read_text()

    # Compare the outputs
    assert generated_output == golden_output, "Multichain output does not match golden file."

    # Clean up the generated file
    if output_pdb.exists():
        output_pdb.unlink()


def test_python_pulchra_hydrogens():
    """
    Tests the Python version of Pulchra with hydrogen generation.
    """
    input_pdb = PROJECT_ROOT / "tests/7laf.pdb"
    output_pdb = PROJECT_ROOT / "tests/7laf.rebuilt.pdb"

    # Clean up any previous output file
    if output_pdb.exists():
        output_pdb.unlink()

    # Construct the command
    command = ["python", str(PROJECT_ROOT / "biomod/utilities/rebuild/main.py"), "--add-hydrogens", str(input_pdb)]

    # Run the command
    subprocess.run(command, check=True, capture_output=True, text=True)

    # Read the generated output file
    assert output_pdb.exists(), f"Output file {output_pdb} was not generated."
    generated_output = output_pdb.read_text()

    # Check for hydrogens
    assert "H" in generated_output

    # Clean up the generated file
    if output_pdb.exists():
        output_pdb.unlink()
