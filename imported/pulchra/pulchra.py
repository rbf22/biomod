#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="PULCHRA Protein Chain Restoration Algorithm",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s [options] <pdb_file>",
        add_help=True,
    )
    parser.add_argument("pdb_file", nargs="?", help="Input PDB file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-n", "--center_chain", action="store_true", help="Center chain")
    parser.add_argument("-x", "--time_seed", action="store_true", help="Time-seed random number generator")
    parser.add_argument("-g", "--pdb_sg", action="store_true", help="Use PDBSG as an input format")
    parser.add_argument("-c", "--no_ca_optimize", action="store_true", help="Skip C-alpha positions optimization")
    parser.add_argument("-p", "--cispro", action="store_true", help="Detect cis-prolins")
    parser.add_argument("-r", "--ca_random", action="store_true", help="Start from a random chain")
    parser.add_argument("-i", "--ini_file", help="Read initial C-alpha coordinates from a PDB file")
    parser.add_argument("-t", "--ca_trajectory", action="store_true", help="Save chain optimization trajectory")
    parser.add_argument("-u", "--ca_start_dist", type=float, default=3.0, help="Maximum shift from the restraint coordinates")
    parser.add_argument("-e", "--bb_rearrange", action="store_true", help="Rearrange backbone atoms")
    parser.add_argument("-b", "--no_rebuild_bb", action="store_true", help="Skip backbone reconstruction")
    parser.add_argument("-q", "--bb_optimize", action="store_true", help="Optimize backbone hydrogen bonds pattern")
    parser.add_argument("--add-hydrogens", action="store_true", help="Outputs hydrogen atoms")
    parser.add_argument("-s", "--no_rebuild_sc", action="store_true", help="Skip side chains reconstruction")
    parser.add_argument("-o", "--no_xvolume", action="store_true", help="Don't attempt to fix excluded volume conflicts")
    parser.add_argument("-z", "--no_chiral", action="store_true", help="Don't check amino acid chirality")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if not args.pdb_file:
        parser.error("the following arguments are required: pdb_file")

    input_path = Path(args.pdb_file)
    output_path = input_path.with_name(f"{input_path.stem}.rebuilt.pdb")

    from pulchra.pdb_parser import read_pdb_file
    from pulchra.core import ca_optimize, rebuild_backbone, rebuild_sidechains, add_hydrogens
    from pulchra.pdb_writer import write_pdb

    molecule = read_pdb_file(input_path, input_path.name)
    if molecule:
        if not args.no_ca_optimize:
            ca_optimize(
                chain=molecule,
                ca_trajectory=args.ca_trajectory,
                ini_file=args.ini_file,
                cispro=args.cispro,
                ca_random=args.ca_random,
                ca_start_dist=args.ca_start_dist
            )

        c_alpha, rbins = None, None
        if not args.no_rebuild_bb:
            c_alpha, rbins = rebuild_backbone(molecule)

        if not args.no_rebuild_sc and c_alpha and rbins:
            rebuild_sidechains(molecule, c_alpha, rbins)

        if args.add_hydrogens:
            add_hydrogens(molecule)

        write_pdb(molecule, output_path)


if __name__ == "__main__":
    main()
