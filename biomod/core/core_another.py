import math
import random
import numpy as np
from .pdb_datastructures import Molecule
from .energy import calc_ca_energy
from .data import AA_NAMES, SHORT_AA_NAMES, AA_NUMS, NHEAVY, HEAVY_ATOM_NAMES, NCO_STAT, NCO_STAT_PRO
from .geometry import calc_distance, calc_r14, superimpose, cross, norm
from .rotamer_data import ROT_STAT_IDX, ROT_STAT_COORDS

def rebuild_sidechains(chain, c_alpha, rbins):
    """
    Rebuilds the side chains of the protein.
    """
    print("Rebuilding side chains...")
    chain_length = len(chain.residues)
    res_list = []
    for res in chain.residues:
        res_list.append(res)

    ca_offset = 5

    for i in range(chain_length):
        res = res_list[i]
        if res.name == "GLY" or not res.protein:
            continue

        x1, y1, z1 = c_alpha[ca_offset + i - 2]
        x2, y2, z2 = c_alpha[ca_offset + i - 1]
        x3, y3, z3 = c_alpha[ca_offset + i]
        x4, y4, z4 = c_alpha[ca_offset + i + 1]

        bin13_1, bin13_2, bin14 = rbins[i]

        v1 = np.array([x4 - x2, y4 - y2, z4 - z2])
        v2a = np.array([x4 - x3, y4 - y3, z4 - z3])
        v2b = np.array([x3 - x2, y3 - y2, z3 - z2])

        v2 = cross(v2a, v2b)
        v3 = cross(v1, v2)

        v1 = norm(v1)
        v2 = norm(v2)
        v3 = norm(v3)

        vv = np.array([v1, v2, v3])
        lsys = np.identity(3)

        sorted_rotamers = []
        for j in range(len(ROT_STAT_IDX)):
            if ROT_STAT_IDX[j][0] == res.type:
                hit = abs(ROT_STAT_IDX[j][1] - bin13_1) + abs(ROT_STAT_IDX[j][2] - bin13_2) + 0.2 * abs(ROT_STAT_IDX[j][3] - bin14)
                sorted_rotamers.append((hit, j))

        sorted_rotamers.sort()

        bestpos = sorted_rotamers[0][1]
        pos = ROT_STAT_IDX[bestpos][5]
        nsc = NHEAVY[res.type] + 1

        sc = []
        for j in range(nsc):
            sc.append(ROT_STAT_COORDS[pos+j])

        rmsd, transformed_coords = superimpose(lsys, vv, sc)

        transformed_coords = np.array(transformed_coords)
        transformed_coords += np.array([x3, y3, z3])

        for j in range(1, nsc):
            atom_name = HEAVY_ATOM_NAMES[res.type][j-1]
            res.add_or_replace_atom(atom_name, transformed_coords[j][0], transformed_coords[j][1], transformed_coords[j][2], 4)


def rebuild_backbone(chain):
    """
    Rebuilds the protein backbone.
    """
    print("Rebuilding backbone...")

    # Initialize data structures
    chain_length = len(chain.residues)
    c_alpha_coords = []
    res_list = []
    for res in chain.residues:
        res_list.append(res)
        for atom in res.atoms:
            if atom.name == 'CA':
                c_alpha_coords.append([atom.x, atom.y, atom.z])
                break

    x_coords = []
    for i in range(5):
        x_coords.append([0.0, 0.0, 0.0])
    x_coords.extend(c_alpha_coords)
    for i in range(5):
        x_coords.append([0.0, 0.0, 0.0])

    c_alpha = x_coords
    ca_offset = 5

    # Rebuild ends
    # Beginning
    tmpcoords = [c_alpha[ca_offset + i] for i in range(5)]
    cacoords = [c_alpha[ca_offset + i] for i in range(2, 5)]
    tmpstat = [c_alpha[ca_offset + i] for i in range(3)]

    rmsd, transformed_coords = superimpose(tmpstat, cacoords, tmpcoords)

    for i in range(2):
        for j in range(3):
            c_alpha[ca_offset + i - 2][j] = transformed_coords[i][j]

    # End
    tmpcoords = [c_alpha[ca_offset + i] for i in range(chain_length - 5, chain_length)]
    cacoords = [c_alpha[ca_offset + i] for i in range(chain_length - 5, chain_length - 2)]
    tmpstat = [c_alpha[ca_offset + i] for i in range(chain_length - 3, chain_length)]

    rmsd, transformed_coords = superimpose(tmpstat, cacoords, tmpcoords)

    c_alpha[ca_offset + chain_length] = transformed_coords[3]
    c_alpha[ca_offset + chain_length + 1] = transformed_coords[4]


    # Loop through the chain
    rbins = []
    for i in range(chain_length + 1):
        x1, y1, z1 = c_alpha[ca_offset + i - 2]
        x2, y2, z2 = c_alpha[ca_offset + i - 1]
        x3, y3, z3 = c_alpha[ca_offset + i]
        x4, y4, z4 = c_alpha[ca_offset + i + 1]

        r13_1 = calc_distance(np.array([x1, y1, z1]), np.array([x3, y3, z3]))
        r13_2 = calc_distance(np.array([x2, y2, z2]), np.array([x4, y4, z4]))
        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])
        p3 = np.array([x3, y3, z3])
        p4 = np.array([x4, y4, z4])
        r14 = calc_r14(p1, p2, p3, p4)

        bin13_1 = int((r13_1 - 4.6) / 0.3)
        bin13_2 = int((r13_2 - 4.6) / 0.3)
        bin14 = int((r14 + 11.0) / 0.3)

        if bin13_1 < 0: bin13_1 = 0
        if bin13_2 < 0: bin13_2 = 0
        if bin14 < 0: bin14 = 0
        if bin13_1 > 9: bin13_1 = 9
        if bin13_2 > 9: bin13_2 = 9
        if bin14 > 73: bin14 = 73

        rbins.append([bin13_1, bin13_2, bin14])

        cacoords = [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]]

        prevres = res_list[i-1] if i > 0 else None

        pro = False
        if prevres and prevres.name == "PRO":
            pro = True

        if pro:
            nco_stat_list = NCO_STAT_PRO
        else:
            nco_stat_list = NCO_STAT

        besthit = 1000.0
        bestpos = 0
        for j, (bins, data) in enumerate(nco_stat_list):
            hit = abs(bins[0] - bin13_1) + abs(bins[1] - bin13_2) + 0.2 * abs(bins[2] - bin14)
            if hit < besthit:
                besthit = hit
                bestpos = j

        tmpstat = nco_stat_list[bestpos][1][:4]
        tmpcoords = nco_stat_list[bestpos][1]

        rmsd, transformed_coords = superimpose(cacoords, tmpstat, tmpcoords)

        if prevres:
            prevres.add_or_replace_atom("C", transformed_coords[4][0], transformed_coords[4][1], transformed_coords[4][2], 1)
            prevres.add_or_replace_atom("O", transformed_coords[5][0], transformed_coords[5][1], transformed_coords[5][2], 1)

        if i < chain_length:
            res = res_list[i]
            res.add_or_replace_atom("N", transformed_coords[6][0], transformed_coords[6][1], transformed_coords[6][2], 1)

    return c_alpha, rbins

def ca_optimize(chain, ca_trajectory, ini_file, cispro, ca_random, ca_start_dist):
    """
    Optimizes the positions of the C-alpha atoms.
    """
    print("Optimizing C-alpha atoms...")

    c_alpha = []
    for res in chain.residues:
        for atom in res.atoms:
            if atom.name == 'CA':
                c_alpha.append(atom)
                break

    chain_length = len(c_alpha)
    if not chain_length:
        return

    init_c_alpha = [[atom.x, atom.y, atom.z] for atom in c_alpha]

    if cispro:
        for i in range(1, chain_length):
            dx = c_alpha[i].x - c_alpha[i-1].x
            dy = c_alpha[i].y - c_alpha[i-1].y
            dz = c_alpha[i].z - c_alpha[i-1].z
            dd = math.sqrt(dx*dx + dy*dy + dz*dz)
            # A simple check for cis-proline, more sophisticated logic might be needed
            if c_alpha[i].res.name == 'PRO' and 2.8 < dd < 3.0:
                c_alpha[i].cispro = True

    if ca_random:
        c_alpha[0].x = 0.0
        c_alpha[0].y = 0.0
        c_alpha[0].z = 0.0
        for i in range(1, chain_length):
            dx = 0.01 * (100 - random.randint(0, 199))
            dy = 0.01 * (100 - random.randint(0, 199))
            dz = 0.01 * (100 - random.randint(0, 199))
            dd = 3.8 / math.sqrt(dx*dx + dy*dy + dz*dz)
            dx *= dd
            dy *= dd
            dz *= dd
            c_alpha[i].x = c_alpha[i-1].x + dx
            c_alpha[i].y = c_alpha[i-1].y + dy
            c_alpha[i].z = c_alpha[i-1].z + dz

    gradient = [[0.0, 0.0, 0.0] for _ in range(chain_length)]
    energies = [0.0, 0.0, 0.0, 0.0]
    new_c_alpha = [[0.0, 0.0, 0.0] for _ in range(chain_length)]

    num_steps = 0
    fcnt = 0
    last_gnorm = 1000.0

    while fcnt < 3 and num_steps < 100: # Simplified loop condition for now
        # Calculate gradients
        for i in range(chain_length):
            gradient[i][0] = gradient[i][1] = gradient[i][2] = 0.0

        e_pot = calc_ca_energy(c_alpha, new_c_alpha, init_c_alpha, gradient, 0.0, energies, True, ca_start_dist)

        # Line search
        alpha1 = -1.0
        alpha2 = 0.0
        alpha3 = 1.0

        ene1 = calc_ca_energy(c_alpha, new_c_alpha, init_c_alpha, gradient, alpha1, energies, False, ca_start_dist)
        ene2 = e_pot
        ene3 = calc_ca_energy(c_alpha, new_c_alpha, init_c_alpha, gradient, alpha3, energies, False, ca_start_dist)

        # Simplified line search
        last_alpha = 0.01

        # Update coordinates
        for i in range(chain_length):
            c_alpha[i].x += last_alpha * gradient[i][0]
            c_alpha[i].y += last_alpha * gradient[i][1]
            c_alpha[i].z += last_alpha * gradient[i][2]

        gnorm = 0.0
        for i in range(chain_length):
            gnorm += gradient[i][0]**2 + gradient[i][1]**2 + gradient[i][2]**2
        gnorm = math.sqrt(gnorm / chain_length)

        if abs(last_gnorm - gnorm) < 1e-3:
            fcnt += 1
        last_gnorm = gnorm

        num_steps += 1

def add_hydrogens(chain: Molecule):
    """
    Adds hydrogen atoms to the protein chain.
    """
    from .hydrogens import get_hydrogen_positions
    from .pdb_datastructures import Atom

    print("Adding hydrogens...")

    prev_c_coord = None
    for res in chain.residues:

        heavy_atoms = {atom.name: np.array([atom.x, atom.y, atom.z]) for atom in res.atoms}

        # Get the C-atom of the previous residue for amide H placement
        if prev_c_coord is None and res.num > 1:
            # This is a fallback for the first residue in a chain that is not the first chain
            # A more robust solution would be to get the C atom from the previous residue object
            pass

        hydrogens = get_hydrogen_positions(res.name, heavy_atoms, prev_c=prev_c_coord)

        for h_name, h_coords in hydrogens.items():
            res.add_or_replace_atom(h_name, h_coords[0], h_coords[1], h_coords[2], 5)

        # Store the C-atom of the current residue for the next iteration
        if 'C' in heavy_atoms:
            prev_c_coord = heavy_atoms['C']
