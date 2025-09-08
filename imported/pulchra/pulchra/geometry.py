# pypulchra/geometry.py

import numpy as np

def superimpose(coords1, coords2, tpoints):
    """
    Superimposes two sets of coordinates and applies the transformation to a third set.
    This is a Python/NumPy port of the superimpose2 function from the C code.
    """
    npoints = len(coords1)

    c1 = np.mean(coords1, axis=0)
    c2 = np.mean(coords2, axis=0)

    coords1_centered = coords1 - c1
    coords2_centered = coords2 - c2
    tpoints_centered = tpoints - c2

    # Covariance matrix
    mat_u = coords1_centered.T @ coords2_centered

    # SVD
    u, _, vh = np.linalg.svd(mat_u)

    # Rotation matrix
    mat_s = u @ vh

    if np.linalg.det(mat_s) < 0:
        vh[2, :] *= -1
        mat_s = u @ vh

    # Apply transformation
    tpoints_transformed = tpoints_centered @ mat_s.T

    # Translate back
    tpoints_final = tpoints_transformed + c1

    # Calculate RMSD
    coords2_transformed = coords2_centered @ mat_s.T
    rmsd = np.sqrt(np.sum((coords1_centered - coords2_transformed)**2) / npoints)

    return rmsd, tpoints_final

def cross(v1, v2):
    """Calculates the cross product of two vectors."""
    return np.cross(v1, v2)

def norm(v):
    """Normalizes a vector."""
    return v / np.linalg.norm(v)

def calc_distance(p1, p2):
    """Calculates the distance between two points."""
    return np.linalg.norm(p1 - p2)

def calc_torsion(a1, a2, a3, a4):
    """Calculates the torsion angle between four points."""
    v12 = a1 - a2
    v43 = a4 - a3
    z = a2 - a3

    p = np.cross(z, v12)
    x = np.cross(z, v43)
    y = np.cross(z, x)

    u = np.dot(p, x)
    v = np.dot(p, y)

    angle = np.degrees(np.arctan2(v, u))
    return angle

def rot_point_vector(p, v, angle):
    """Rotates a point around a vector."""
    angle_rad = np.radians(angle)
    u, v, w = v
    x, y, z = p
    sa = np.sin(angle_rad)
    ca = np.cos(angle_rad)

    px = u*(u*x + v*y + w*z) + (x*(v*v + w*w) - u*(v*y + w*z))*ca + (-w*y + v*z)*sa
    py = v*(u*x + v*y + w*z) + (y*(u*u + w*w) - v*(u*x + w*z))*ca + (w*x - u*z)*sa
    pz = w*(u*x + v*y + w*z) + (z*(u*u + v*v) - w*(u*x + v*y))*ca + (-v*x + u*y)*sa

    return np.array([px, py, pz])

def calc_r14(p1, p2, p3, p4):
    """
    Calculates the signed distance between p1 and p4.
    The sign is determined by the handedness of the vectors p1-p2, p2-p3, p3-p4.
    """
    dx = p4[0] - p1[0]
    dy = p4[1] - p1[1]
    dz = p4[2] - p1[2]

    r = np.sqrt(dx*dx + dy*dy + dz*dz)

    vx1 = p2[0] - p1[0]
    vy1 = p2[1] - p1[1]
    vz1 = p2[2] - p1[2]
    vx2 = p3[0] - p2[0]
    vy2 = p3[1] - p2[1]
    vz2 = p3[2] - p2[2]
    vx3 = p4[0] - p3[0]
    vy3 = p4[1] - p3[1]
    vz3 = p4[2] - p3[2]

    hand = (vy1*vz2 - vy2*vz1)*vx3 + \
           (vz1*vx2 - vz2*vx1)*vy3 + \
           (vx1*vy2 - vx2*vy1)*vz3

    if hand < 0:
        r = -r

    return r

def find_atom(res, atom_name):
    """Finds an atom in a residue by name."""
    for atom in res.atoms:
        if atom.name == atom_name:
            return atom
    return None

def add_replace_atom(res, atom_name, x, y, z, flag=0):
    """Adds or replaces an atom in a residue."""
    atom = find_atom(res, atom_name)
    if atom:
        atom.x = x
        atom.y = y
        atom.z = z
        atom.flag |= flag
    else:
        new_atom = Atom(name=atom_name, x=x, y=y, z=z, flag=flag)
        res.add_atom(new_atom)
