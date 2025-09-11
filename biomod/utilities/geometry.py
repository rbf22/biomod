"""
Contains various geometry-related utility functions from the different
constituent libraries.
"""
import torch
import numpy as np
from ..energy import math_utils
from ..energy import hashings
from ..config import PADDING_INDEX

# --- Functions from pypulchra (`biomod/utilities/geometry.py`) ---

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

def _generate_fibonacci_sphere(num_points: int):
    """
    Creates equidistant points on the surface of a sphere using Fibonacci sphere algorithm.
    """
    ga = (3 - np.sqrt(5)) * np.pi # golden angle
    # Create a list of golden angle increments along tha range of number of points
    theta = ga * np.arange(num_points)
    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = np.linspace(1/num_points-1, 1-1/num_points, num_points)
    # a list of the radii at each height step of the unit circle
    radius = np.sqrt(1 - z * z)
    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    y = radius * np.sin(theta)
    x = radius * np.cos(theta)
    return np.array(list(zip(x, y, z))), 1.0

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

dihedral_angle = calc_torsion

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
    # This will need to be fixed to use the core Atom class
    from ..core.structures import Atom
    atom = find_atom(res, atom_name)
    if atom:
        atom.x = x
        atom.y = y
        atom.z = z
        atom.flag |= flag
    else:
        new_atom = Atom(name=atom_name, x=x, y=y, z=z, flag=flag)
        res.add_atom(new_atom)

# --- Functions from Vitra (`biomod/utilities/geometry_another.py`) ---

def get_interaction_angles(acceptCoords, donorCoords, acceptorPartners1, donorPartners1, acceptorPartners2,
                           donorPartners2, hydrogens, freeOrb):
    DonProton = donorCoords - hydrogens

    freeProt = freeOrb - hydrogens
    AccFree = acceptCoords - freeOrb

    # angles involved ##

    freeProtDon, nanMaskFpd, _ = math_utils.angle2dVectors(freeProt, DonProton)

    protFreeAcc, nanMaskPfa, test = math_utils.angle2dVectors(-freeProt, AccFree)

    dihed, nanMaskDihed, test = math_utils.dihedral2dVectors(donorCoords, hydrogens, freeOrb, acceptCoords,
                                                             testing=True)
    test = hydrogens
    ang_planes, nanMaskPlane = math_utils.plane_angle(acceptCoords, acceptorPartners1, acceptorPartners2, donorCoords,
                                                      donorPartners1, donorPartners2)

    assert not np.isnan(torch.sum(protFreeAcc).cpu().data.numpy())

    # non usi i plane angles, metti il nan separato
    fullNanMask = nanMaskFpd * nanMaskPfa * nanMaskDihed

    return torch.cat([freeProtDon, protFreeAcc, dihed, ang_planes], dim=1), fullNanMask.squeeze(1), nanMaskPlane, test


def get_interaction_anglesStatisticalPotential(acceptCoords, donorCoords, acceptorPartners1, donorPartners1):
    DonProton = donorPartners1 - donorCoords

    freeProt = acceptCoords - donorCoords

    AccFree = acceptorPartners1 - acceptCoords

    # angles involved ##

    freeProtDon, nanMaskFpd, _ = math_utils.angle2dVectors(freeProt, DonProton)

    protFreeAcc, nanMaskPfa, test = math_utils.angle2dVectors(-freeProt, AccFree)

    dihed, nanMaskDihed, test = math_utils.dihedral2dVectors(donorPartners1, donorCoords, acceptCoords,
                                                             acceptorPartners1, testing=True)

    assert not np.isnan(torch.sum(protFreeAcc).cpu().data.numpy())

    # non usi i plane angles, metti il nan separato
    fullNanMask = nanMaskFpd * nanMaskPfa * nanMaskDihed

    return torch.cat([freeProtDon, protFreeAcc, dihed], dim=1), fullNanMask.squeeze(1)


def get_standard_angles(donorCoords, acceptCoords, hydrogens, freeOrb):
    DonProton = donorCoords - hydrogens
    freeProt = freeOrb - hydrogens
    AccFree = acceptCoords - freeOrb

    # angles involved ##
    freeProtDon, nanMaskFpd = math_utils.angle2dVectors(freeProt, DonProton)
    protFreeAcc, nanMaskPfa = math_utils.angle2dVectors(-freeProt, AccFree)
    dihed, nanMaskDihed = math_utils.dihedral2dVectors(donorCoords, hydrogens, freeOrb, acceptCoords)
    fullNanMask = nanMaskFpd & nanMaskPfa & nanMaskDihed

    return torch.cat([freeProtDon, protFreeAcc, dihed], dim=1), fullNanMask.squeeze(1)


def define_partners_coords(coords, atom_hashing_position, res_num, atom_names, res_names, partners):
    mask_partners = []
    batch_len = len(coords)
    orig_tot = []
    indexes1_tot = []
    indexes2_tot = []

    for batch in range(batch_len):
        indexes1 = []
        indexes2 = []
        orig = []
        maxres = max(res_num[batch])
        for i in range(len(coords[batch])):  # every atom

            respos = res_num[batch][i]
            atom_n = atom_names[batch][i]
            res_n = res_names[batch][i]

            if atom_n in partners[res_n]:
                # atoms on the peptide bond define the plane using atoms from the preevious/successive residue
                par1 = partners[res_n][atom_n][0][0]
                residue_correction1 = partners[res_n][atom_n][0][1]
                par2 = partners[res_n][atom_n][1][0]
                residue_correction2 = partners[res_n][atom_n][1][1]

                if not (
                        respos + residue_correction1 < 0 or respos + residue_correction1 >= maxres or respos + residue_correction2 < 0 or respos + residue_correction2 >= maxres) and par1 in \
                        atom_hashing_position[batch][respos + residue_correction1] and par2 in \
                        atom_hashing_position[batch][respos + residue_correction2]:  # for missing atoms
                    indexes1 += [atom_hashing_position[batch][respos + residue_correction1][par1]]
                    indexes2 += [atom_hashing_position[batch][respos + residue_correction2][par2]]
                    orig += [i]

        indexes1_tot += [indexes1]
        indexes2_tot += [indexes2]

        orig_tot += [orig]
        mp = torch.zeros(coords[batch].shape[0])
        mp[orig] = 1
        mp = mp.ge(1).unsqueeze(1)
        mask_partners += [mp]

    return indexes1_tot, indexes2_tot, orig_tot, mask_partners


def calculateTorsionAngles(coords_tot, atom_description, angle_coords, float_type=torch.float):
    # bb #
    c_hahsingMask = (atom_description[:, :, hashings.property_hashings["atom_description"]["atom_types"]] ==
                     hashings.atom_hash["ALA"]["C"]) | (
                            atom_description[:, :, hashings.property_hashings["atom_description"]["atom_types"]] ==
                            hashings.atom_hash["PRO"]["C"])
    ca_hahsingMask = (atom_description[:, :, hashings.property_hashings["atom_description"]["atom_types"]] ==
                      hashings.atom_hash["ALA"]["CA"]) | (
                             atom_description[:, :, hashings.property_hashings["atom_description"]["atom_types"]] ==
                             hashings.atom_hash["PRO"]["CA"])
    n_hahsingMask = (atom_description[:, :, hashings.property_hashings["atom_description"]["atom_types"]] ==
                     hashings.atom_hash["ALA"]["N"]) | (
                            atom_description[:, :, hashings.property_hashings["atom_description"]["atom_types"]] ==
                            hashings.atom_hash["PRO"]["N"])

    batch = coords_tot.shape[0]
    L = coords_tot.shape[1]

    coords = coords_tot[:, :, hashings.property_hashings["coords_total"]["coords"], :]
    resiMinus1 = atom_description[:, :, hashings.property_hashings["atom_description"]["res_num"]] - 1
    resiPlus1 = atom_description[:, :, hashings.property_hashings["atom_description"]["res_num"]] + 1
    resi = atom_description[:, :, hashings.property_hashings["atom_description"]["res_num"]]
    chain_ind = atom_description[:, :, hashings.property_hashings["atom_description"]["chain"]].long()

    maxResi = resi.max() + 1
    max_chain_ind = chain_ind.max() + 1

    batch_ind = torch.arange(0, batch, device=coords_tot.device).unsqueeze(1).expand(batch, L)

    anglesFull = torch.full((batch, max_chain_ind, maxResi, 9, 3), PADDING_INDEX, device=coords_tot.device,
                            dtype=float_type)

    padding_mask = ~(resi == PADDING_INDEX)

    nMask = n_hahsingMask & padding_mask
    full_ind = (batch_ind[nMask].long(), chain_ind[nMask], resi[nMask].long(),
                torch.zeros(batch_ind.shape, device=coords_tot.device, dtype=torch.long)[nMask])
    anglesFull = anglesFull.index_put_(full_ind, coords[nMask])

    caMask = ca_hahsingMask & padding_mask
    full_ind = (batch_ind[caMask].long(), chain_ind[caMask], resi[caMask].long(),
                torch.full(batch_ind.shape, 1, device=coords_tot.device, dtype=torch.long)[caMask])
    anglesFull = anglesFull.index_put_(full_ind, coords[caMask])

    cMask = c_hahsingMask & padding_mask
    full_ind = (batch_ind[cMask].long(), chain_ind[cMask], resi[cMask].long(),
                torch.full(batch_ind.shape, 2, device=coords_tot.device, dtype=torch.long)[cMask])
    anglesFull = anglesFull.index_put_(full_ind, coords[cMask])

    c_plus1_Mask = c_hahsingMask & padding_mask & (resiPlus1 < maxResi)
    full_ind = (batch_ind[c_plus1_Mask].long(), chain_ind[c_plus1_Mask], resiPlus1[c_plus1_Mask].long(),
                torch.full(c_plus1_Mask.shape, 3, device=coords_tot.device, dtype=torch.long)[c_plus1_Mask])
    anglesFull = anglesFull.index_put_(full_ind, coords[c_plus1_Mask])

    n_minus_oneMaks = n_hahsingMask & (resiMinus1 >= 0) & padding_mask
    full_ind = (batch_ind[n_minus_oneMaks].long(), chain_ind[n_minus_oneMaks], resiMinus1[n_minus_oneMaks].long(),
                torch.full(batch_ind.shape, 4, device=coords_tot.device, dtype=torch.long)[n_minus_oneMaks])
    anglesFull = anglesFull.index_put_(full_ind, coords[n_minus_oneMaks])

    ca_minus_oneMaks = ca_hahsingMask & (resiMinus1 >= 0) & padding_mask
    full_ind = (batch_ind[ca_minus_oneMaks].long(), chain_ind[ca_minus_oneMaks], resiMinus1[ca_minus_oneMaks].long(),
                torch.full(batch_ind.shape, 5, device=coords_tot.device, dtype=torch.long)[ca_minus_oneMaks])
    anglesFull = anglesFull.index_put_(full_ind, coords[ca_minus_oneMaks])

    residue_padding_mask = ~(anglesFull[:, :, :, :, 0] == PADDING_INDEX)

    everythingForPhi = residue_padding_mask[:, :, :, (3, 0, 1, 2)].prod(dim=-1).bool()
    phi_angles = anglesFull[everythingForPhi]
    phi, _ = math_utils.dihedral2dVectors(phi_angles[:, 3], phi_angles[:, 0], phi_angles[:, 1], phi_angles[:, 2])

    everythingForPsi = residue_padding_mask[:, :, :, (0, 1, 2, 4)].prod(dim=-1).bool()
    psi_angles = anglesFull[everythingForPsi]
    psi, _ = math_utils.dihedral2dVectors(psi_angles[:, 0], psi_angles[:, 1], psi_angles[:, 2], psi_angles[:, 4])

    everythingForOmega = residue_padding_mask[:, :, :, (1, 2, 4, 5)].prod(dim=-1).bool()
    omega_angles = anglesFull[everythingForOmega]
    omega, _ = math_utils.dihedral2dVectors(omega_angles[:, 1], omega_angles[:, 2], omega_angles[:, 4],
                                            omega_angles[:, 5])

    # full angles #
    phiFull = torch.full((batch, max_chain_ind, maxResi), PADDING_INDEX, device=coords_tot.device)
    psiFull = torch.full((batch, max_chain_ind, maxResi), PADDING_INDEX, device=coords_tot.device)
    omegaFull = torch.full((batch, max_chain_ind, maxResi), PADDING_INDEX, device=coords_tot.device)

    phiFull[everythingForPhi] = phi.squeeze(-1)
    psiFull[everythingForPsi] = psi.squeeze(-1)
    omegaFull[everythingForOmega] = omega.squeeze(-1)

    sidechain_todo = (~(angle_coords[:, :, :, :, 0].eq(PADDING_INDEX))).prod(-1).bool()
    # calculate angles for which all atoms are present
    todo = angle_coords[sidechain_todo]
    sidechain, _ = math_utils.dihedral2dVectors(todo[:, 0], todo[:, 1], todo[:, 2], todo[:, 3])

    fullanglesSC = torch.full((batch, max_chain_ind, maxResi, 5), PADDING_INDEX).type_as(sidechain)
    fullanglesSC[sidechain_todo] = sidechain.squeeze(-1)
    fullangles = torch.cat([phiFull.unsqueeze(-1), psiFull.unsqueeze(-1), omegaFull.unsqueeze(-1), fullanglesSC], dim=3)

    return fullangles


def calculateTorsionAnglesBuildModels(coords_tot, atom_description, angle_coords, alternatives, alternative_resi,
                                      seqChain, seqNum, float_type=torch.float):
    # bb #
    c_hahsingMask = (atom_description[:, :, hashings.property_hashings["atom_description"]["atom_types"]] ==
                     hashings.atom_hash["ALA"]["C"]) | (
                            atom_description[:, :, hashings.property_hashings["atom_description"]["atom_types"]] ==
                            hashings.atom_hash["PRO"]["C"])
    ca_hahsingMask = (atom_description[:, :, hashings.property_hashings["atom_description"]["atom_types"]] ==
                      hashings.atom_hash["ALA"]["CA"]) | (
                             atom_description[:, :, hashings.property_hashings["atom_description"]["atom_types"]] ==
                             hashings.atom_hash["PRO"]["CA"])
    n_hahsingMask = (atom_description[:, :, hashings.property_hashings["atom_description"]["atom_types"]] ==
                     hashings.atom_hash["ALA"]["N"]) | (
                            atom_description[:, :, hashings.property_hashings["atom_description"]["atom_types"]] ==
                            hashings.atom_hash["PRO"]["N"])

    batch = coords_tot.shape[0]
    L = coords_tot.shape[1]
    max_alternatives = alternatives.shape[-1]
    nrots = coords_tot.shape[-3]

    coords = coords_tot[:, :, :, hashings.property_hashings["coords_total"]["coords"], :]
    resiMinus1 = atom_description[:, :, hashings.property_hashings["atom_description"]["res_num"]] - 1
    resiPlus1 = atom_description[:, :, hashings.property_hashings["atom_description"]["res_num"]] + 1
    resi = atom_description[:, :, hashings.property_hashings["atom_description"]["res_num"]]
    chain_ind = atom_description[:, :, hashings.property_hashings["atom_description"]["chain"]].long()

    maxResi = resi.max() + 1
    max_chain_ind = chain_ind.max() + 1

    batch_ind = torch.arange(0, batch, device=coords_tot.device).unsqueeze(1).expand(batch, L)

    anglesFull = torch.full((batch, max_chain_ind, maxResi, 9, nrots, 3), PADDING_INDEX, device=coords_tot.device,
                            dtype=float_type)

    fullangles = []
    for alt in range(max_alternatives):
        padding_mask = ~(resi == PADDING_INDEX) & alternatives[:, :, alt]

        alternative_Resi_mask = alternative_resi[:, :, alt]
        nMask = n_hahsingMask & padding_mask
        full_ind = (batch_ind[nMask].long(), chain_ind[nMask], resi[nMask].long(),
                    torch.zeros(batch_ind.shape, device=coords_tot.device, dtype=torch.long)[nMask])
        anglesFull = anglesFull.index_put_(full_ind, coords[nMask])

        caMask = ca_hahsingMask & padding_mask
        full_ind = (batch_ind[caMask].long(), chain_ind[caMask], resi[caMask].long(),
                    torch.full(batch_ind.shape, 1, device=coords_tot.device, dtype=torch.long)[caMask])
        anglesFull = anglesFull.index_put_(full_ind, coords[caMask])

        cMask = c_hahsingMask & padding_mask
        full_ind = (batch_ind[cMask].long(), chain_ind[cMask], resi[cMask].long(),
                    torch.full(batch_ind.shape, 2, device=coords_tot.device, dtype=torch.long)[cMask])
        anglesFull = anglesFull.index_put_(full_ind, coords[cMask])

        c_plus1_Mask = c_hahsingMask & padding_mask & (resiPlus1 < maxResi)
        full_ind = (batch_ind[c_plus1_Mask].long(), chain_ind[c_plus1_Mask], resiPlus1[c_plus1_Mask].long(),
                    torch.full(c_plus1_Mask.shape, 3, device=coords_tot.device, dtype=torch.long)[c_plus1_Mask])
        anglesFull = anglesFull.index_put_(full_ind, coords[c_plus1_Mask])

        n_minus_oneMaks = n_hahsingMask & (resiMinus1 >= 0) & padding_mask
        full_ind = (
            batch_ind[n_minus_oneMaks].long(), chain_ind[n_minus_oneMaks], resiMinus1[n_minus_oneMaks].long(),
            torch.full(batch_ind.shape, 4, device=coords_tot.device, dtype=torch.long)[n_minus_oneMaks])
        anglesFull = anglesFull.index_put_(full_ind, coords[n_minus_oneMaks])

        ca_minus_oneMaks = ca_hahsingMask & (resiMinus1 >= 0) & padding_mask
        full_ind = (
            batch_ind[ca_minus_oneMaks].long(), chain_ind[ca_minus_oneMaks], resiMinus1[ca_minus_oneMaks].long(),
            torch.full(batch_ind.shape, 5, device=coords_tot.device, dtype=torch.long)[ca_minus_oneMaks])
        anglesFull = anglesFull.index_put_(full_ind, coords[ca_minus_oneMaks])

        residue_padding_mask = ~(anglesFull[:, :, :, :, :, 0] == PADDING_INDEX)

        everythingForPhi = residue_padding_mask[:, :, :, (3, 0, 1, 2)].prod(dim=-2).bool()
        phi_angles = anglesFull.transpose(-2, -3)[everythingForPhi]
        phi, _ = math_utils.dihedral2dVectors(phi_angles[:, 3], phi_angles[:, 0], phi_angles[:, 1], phi_angles[:, 2])

        everythingForPsi = residue_padding_mask[:, :, :, (0, 1, 2, 4)].prod(dim=-2).bool()
        psi_angles = anglesFull.transpose(-2, -3)[everythingForPsi]
        psi, _ = math_utils.dihedral2dVectors(psi_angles[:, 0], psi_angles[:, 1], psi_angles[:, 2], psi_angles[:, 4])

        everythingForOmega = residue_padding_mask[:, :, :, (1, 2, 4, 5)].prod(dim=-2).bool()
        omega_angles = anglesFull.transpose(-2, -3)[everythingForOmega]
        omega, _ = math_utils.dihedral2dVectors(omega_angles[:, 1], omega_angles[:, 2], omega_angles[:, 4],
                                                omega_angles[:, 5])

        # full angles #
        phiFull = torch.full((batch, max_chain_ind, maxResi, nrots), PADDING_INDEX, device=coords_tot.device)
        psiFull = torch.full((batch, max_chain_ind, maxResi, nrots), PADDING_INDEX, device=coords_tot.device)
        omegaFull = torch.full((batch, max_chain_ind, maxResi, nrots), PADDING_INDEX, device=coords_tot.device)

        phiFull[everythingForPhi] = phi.squeeze(-1)
        psiFull[everythingForPsi] = psi.squeeze(-1)
        omegaFull[everythingForOmega] = omega.squeeze(-1)

        alterAngleCoords = angle_coords[alternative_Resi_mask]
        sidechain_todo = (~(alterAngleCoords[:, :, :, :, 0].eq(PADDING_INDEX))).prod(-1).bool()

        # calculate angles for which all atoms are present
        todo = alterAngleCoords[sidechain_todo]
        sidechain, _ = math_utils.dihedral2dVectors(todo[:, 0], todo[:, 1], todo[:, 2], todo[:, 3])

        seqshape = seqNum.shape[1]

        batch_SC = \
            torch.arange(angle_coords.shape[0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch, seqshape, nrots,
                                                                                                 5)[
                alternative_Resi_mask][sidechain_todo]
        chain_SC = seqChain.unsqueeze(-1).unsqueeze(-1).expand(batch, seqshape, nrots, 5)[alternative_Resi_mask][
            sidechain_todo]
        resi_SC = seqNum.unsqueeze(-1).unsqueeze(-1).expand(batch, seqshape, nrots, 5)[alternative_Resi_mask][
            sidechain_todo]
        numrot = torch.arange(nrots).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(batch, seqshape, nrots, 5)[
            alternative_Resi_mask][sidechain_todo]
        numtor = \
            torch.arange(5).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch, seqshape, nrots, 5)[
                alternative_Resi_mask][
                sidechain_todo]
        indices = (batch_SC, chain_SC, resi_SC, numrot, numtor)

        fullanglesSC = torch.full((batch, max_chain_ind, maxResi, nrots, 5), PADDING_INDEX).type_as(sidechain)
        fullanglesSC.index_put_(indices, sidechain.squeeze(-1))

        fullangles += [torch.cat([phiFull.unsqueeze(-1), psiFull.unsqueeze(-1), omegaFull.unsqueeze(-1), fullanglesSC],
                                 dim=-1).unsqueeze(-3)]

    fullangles = torch.cat(fullangles, dim=-3)
    return fullangles
