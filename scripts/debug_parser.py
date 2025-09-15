import numpy as np

def build_coords_from_ic(atom_name, ic_data, atom_coords):
    """
    Builds coordinates for an atom based on a single IC line.
    """
    a1_name, a2_name, a3_name = ic_data['deps']

    if a1_name not in atom_coords or a2_name not in atom_coords or a3_name not in atom_coords:
        print(f"Missing dependency for {atom_name}: {a1_name}, {a2_name}, {a3_name}")
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
    ala_ic_records = [
        "IC -C   CA   *N   HN    1.3551 126.4900  180.0000 115.4200  0.9996",
        "IC N    C    *CA  CB    1.4592 114.4400  123.2300 111.0900  1.5461",
        "IC N    C    *CA  HA    1.4592 114.4400 -120.4500 106.3900  1.0840",
        "IC C    CA   CB   HB1   1.5390 111.0900  177.2500 109.6000  1.1109",
        "IC HB1  CA   *CB  HB2   1.1109 109.6000  119.1300 111.0500  1.1119",
        "IC HB1  CA   *CB  HB3   1.1109 109.6000 -119.5800 111.6100  1.1114",
    ]

    atom_coords = {}
    atom_coords['N'] = np.array([0.0, 0.0, 0.0])
    atom_coords['CA'] = np.array([1.45, 0.0, 0.0])
    atom_coords['C'] = np.array([1.45 + 1.52*np.cos(np.deg2rad(111.0)), 1.52*np.sin(np.deg2rad(111.0)), 0.0])
    atom_coords['-C'] = np.array([-0.5, 1.0, 0.0])
    atom_coords['+N'] = np.array([2.0, 2.0, 0.0])

    ics = {}
    for ic_line in ala_ic_records:
        parts = ic_line.split()
        atom_name = parts[1].replace('*','')
        a1, a2, a3, a4 = [p.replace('*', '') for p in parts[2:6]]
        values = [float(p) for p in parts[6:11]]
        ics[atom_name] = {'deps':(a2,a3,a4), 'values':values}

    for ic_line in ala_ic_records:
        parts = ic_line.split()
        atom_name = parts[1].replace('*','')
        if atom_name not in atom_coords:
            coords = build_coords_from_ic(atom_name, ics[atom_name], atom_coords)
            if coords is not None:
                atom_coords[atom_name] = coords
                print(f"Built {atom_name}: {coords}")

    print("\nFinal coordinates:")
    for name, coords in atom_coords.items():
        print(f"{name}: {coords}")

if __name__ == '__main__':
    main()
