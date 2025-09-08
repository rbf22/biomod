import sys

def filter_ca_atoms(input_pdb_path, output_pdb_path):
    with open(input_pdb_path, 'r') as infile, open(output_pdb_path, 'w') as outfile:
        for line in infile:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                outfile.write(line)
            elif line.startswith("TER"):
                outfile.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_ca_pdb.py <input_pdb_path> <output_pdb_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    filter_ca_atoms(input_path, output_path)
