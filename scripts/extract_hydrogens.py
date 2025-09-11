import json
from pathlib import Path
import numpy as np

def get_local_coordinates(coords, origin, x_axis, y_axis, z_axis):
    """
    Transforms global coordinates to a local coordinate system.
    """
    p = coords - origin
    x = np.dot(p, x_axis)
    y = np.dot(p, y_axis)
    z = np.dot(p, z_axis)
    return np.array([x, y, z])

def is_polar(h_atom_name, res_name):
    """
    Determines if a hydrogen atom is polar based on its name and residue type.
    This is a simplified approach and might need refinement.
    """
    if h_atom_name == 'H': # Amide hydrogen
        return True
    if res_name in ['ARG', 'LYS', 'HIS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'TYR']:
        if h_atom_name.startswith('HH') or h_atom_name.startswith('HD') or h_atom_name.startswith('HE') or h_atom_name.startswith('HG'):
            return True
    return False

def main():
    pdb_file = '1L2K.pdb'
    output_dir = Path('pulchra/parameters')

    residues = {}
    current_model = 1

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                model = int(line.split()[1])
                if model > 1:
                    break
            if line.startswith('ATOM'):
                res_name = line[17:20].strip()
                if res_name == "UNK":
                    continue
                atom_name = line[12:16].strip()
                res_seq = int(line[22:26])

                # Use a unique key for each residue
                res_key = (res_name, res_seq)

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                if res_key not in residues:
                    residues[res_key] = []
                residues[res_key].append({
                    'atom_name': atom_name,
                    'coords': np.array([x, y, z])
                })

    processed_residues = {}
    for res_key, atoms in residues.items():
        res_name = res_key[0]
        if res_name not in processed_residues:
            processed_residues[res_name] = atoms

    for res_name, atoms in processed_residues.items():
        # Find backbone atoms
        n_atom = next((a for a in atoms if a['atom_name'] == 'N'), None)
        ca_atom = next((a for a in atoms if a['atom_name'] == 'CA'), None)
        c_atom = next((a for a in atoms if a['atom_name'] == 'C'), None)

        if not (n_atom and ca_atom and c_atom):
            continue

        # Define local coordinate system
        origin = ca_atom['coords']
        x_axis = (c_atom['coords'] - ca_atom['coords'])
        x_axis /= np.linalg.norm(x_axis)

        temp_vec = n_atom['coords'] - ca_atom['coords']
        z_axis = np.cross(x_axis, temp_vec)
        z_axis /= np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)

        hydrogens = []
        for atom in atoms:
            if atom['atom_name'].startswith('H') or atom['atom_name'].startswith('D'):
                polar = is_polar(atom['atom_name'], res_name)
                local_coords = get_local_coordinates(atom['coords'], origin, x_axis, y_axis, z_axis)
                hydrogens.append({
                    'name': atom['atom_name'],
                    'type': 'polar' if polar else 'non-polar',
                    'ideal_coordinates': local_coords.tolist()
                })

        # Update the existing JSON file for the residue
        json_file = output_dir / f"{res_name}.json"
        if json_file.exists():
            with open(json_file, 'r+') as f:
                try:
                    data = json.load(f)
                    if 'hydrogens' not in data:
                        data['hydrogens'] = hydrogens
                        f.seek(0)
                        json.dump(data, f, indent=4)
                        f.truncate()
                except json.JSONDecodeError:
                    # File is empty or malformed
                    data = {}
                    data['hydrogens'] = hydrogens
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()


if __name__ == '__main__':
    main()
