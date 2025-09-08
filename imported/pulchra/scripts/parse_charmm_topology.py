import json
from pathlib import Path
import numpy as np
import re
from collections import deque

def build_coords_from_ic(atom_name, ic_data, atom_coords):
    """
    Builds coordinates for an atom based on a single IC line.
    """
    a1_name, a2_name, a3_name = ic_data['deps']

    if a1_name not in atom_coords or a2_name not in atom_coords or a3_name not in atom_coords:
        return None

    p1 = atom_coords[a1_name]
    p2 = atom_coords[a2_name]
    p3 = atom_coords[a3_name]

    bond, angle, dihedral, _, _ = ic_data['values']
    angle_rad = np.deg2rad(angle)
    dihedral_rad = np.deg2rad(dihedral)

    a = p2 - p1
    b = p3 - p2

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0: return None

    a /= norm_a
    b /= norm_b

    n = np.cross(a,b)
    norm_n = np.linalg.norm(n)
    if norm_n == 0: return None
    n /= norm_n

    c = np.cross(n,b)

    x = bond * np.cos(np.pi - angle_rad)
    y = bond * np.sin(np.pi - angle_rad) * np.cos(dihedral_rad)
    z = bond * np.sin(np.pi - angle_rad) * np.sin(dihedral_rad)

    d = -b*x + c*y + n*z

    return p3 + d


def main():
    topology_file = 'top_all27_prot_lipid.inp'
    output_dir = Path('pulchra/parameters')

    with open(topology_file, 'r') as f:
        lines = f.readlines()

    residues = {}
    current_res = None
    for line in lines:
        line = re.sub(r'!.*$', '', line).strip()
        if not line:
            continue
        if line.startswith('RESI'):
            parts = line.split()
            if len(parts) > 1:
                res_name = parts[1]
                if len(res_name) > 3: continue
                current_res = res_name
                residues[current_res] = {'atoms': [], 'ics': {}}
        elif current_res and line.startswith('ATOM'):
            parts = line.split()
            residues[current_res]['atoms'].append({'name': parts[1], 'type': parts[2]})
        elif current_res and line.startswith('IC'):
            parts = line.split()
            atom_name = parts[1].replace('*','')
            a1, a2, a3, a4 = [p.replace('*', '') for p in parts[2:6]]
            try:
                values = [float(p) for p in parts[6:11]]
                residues[current_res]['ics'][atom_name] = {'deps':(a2,a3,a4), 'values':values}
            except (ValueError, IndexError):
                pass


    for res_name, data in residues.items():
        if not (output_dir / f"{res_name}.json").exists():
            continue

        atom_coords = {}
        atom_coords['N'] = np.array([0.0, 0.0, 0.0])
        atom_coords['CA'] = np.array([1.45, 0.0, 0.0])
        atom_coords['C'] = np.array([1.45 + 1.52*np.cos(np.deg2rad(111.0)), 1.52*np.sin(np.deg2rad(111.0)), 0.0])
        atom_coords['-C'] = np.array([-0.5, 1.0, 0.0])
        atom_coords['+N'] = np.array([2.0, 2.0, 0.0])


        # Build dependency graph and in_degrees
        adj = {atom['name']: [] for atom in data['atoms']}
        in_degree = {atom['name']: 0 for atom in data['atoms']}

        all_atoms = {atom['name'] for atom in data['atoms']}

        for atom_name, ic_data in data['ics'].items():
            if atom_name not in all_atoms: continue
            for dep in ic_data['deps']:
                if dep in all_atoms:
                    if dep not in adj: adj[dep] = []
                    adj[dep].append(atom_name)
                    in_degree[atom_name] += 1

        q = deque([atom for atom, degree in in_degree.items() if degree == 0 and atom in atom_coords])

        build_order = []
        while q:
            u = q.popleft()
            build_order.append(u)
            if u not in adj: continue
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)

        for atom_name in build_order:
            if atom_name in data['ics']:
                coords = build_coords_from_ic(atom_name, data['ics'][atom_name], atom_coords)
                if coords is not None:
                    atom_coords[atom_name] = coords

        hydrogens = []
        for atom in data['atoms']:
            atom_name = atom['name']
            atom_type = atom['type']
            if atom_type.startswith('H') and atom_name in atom_coords:
                is_polar = atom_type in ['H', 'HS']
                hydrogens.append({
                    'name': atom_name,
                    'type': 'polar' if is_polar else 'non-polar',
                    'ideal_coordinates': atom_coords[atom_name].tolist()
                })

        json_file = output_dir / f"{res_name}.json"
        if json_file.exists():
            with open(json_file, 'r+') as f:
                json_data = json.load(f)
                if 'hydrogens' not in json_data or not json_data['hydrogens']:
                    json_data['hydrogens'] = hydrogens
                    f.seek(0)
                    json.dump(json_data, f, indent=4)
                    f.truncate()

if __name__ == '__main__':
    main()
