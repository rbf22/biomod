import math
from . import constants

def calc_ca_energy(c_alpha, new_c_alpha, init_c_alpha, gradient, alpha, ene, calc_gradient, ca_start_dist):
    """
    Calculates the energy of the C-alpha chain.
    """
    chain_length = len(c_alpha)
    if not new_c_alpha:
        new_c_alpha = [[atom.x, atom.y, atom.z] for atom in c_alpha]

    for i in range(chain_length):
        new_c_alpha[i][0] = c_alpha[i].x + alpha * gradient[i][0]
        new_c_alpha[i][1] = c_alpha[i].y + alpha * gradient[i][1]
        new_c_alpha[i][2] = c_alpha[i].z + alpha * gradient[i][2]

    new_e_pot = 0.0
    ene[0] = ene[1] = ene[2] = ene[3] = 0.0

    # CALC_C_ALPHA_START
    for i in range(chain_length):
        dx = new_c_alpha[i][0] - init_c_alpha[i][0]
        dy = new_c_alpha[i][1] - init_c_alpha[i][1]
        dz = new_c_alpha[i][2] - init_c_alpha[i][2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist > ca_start_dist:
            ddist2 = (dist - ca_start_dist)**2
            new_e_pot += constants.CA_START_K * ddist2
            ene[1] += constants.CA_START_K * ddist2
            if calc_gradient:
                grad = (dist - ca_start_dist) * (2.0 * constants.CA_START_K) / dist
                gradient[i][0] -= grad * dx
                gradient[i][1] -= grad * dy
                gradient[i][2] -= grad * dz

    # CALC_C_ALPHA
    for i in range(1, chain_length):
        dx = new_c_alpha[i][0] - new_c_alpha[i-1][0]
        dy = new_c_alpha[i][1] - new_c_alpha[i-1][1]
        dz = new_c_alpha[i][2] - new_c_alpha[i-1][2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if c_alpha[i].cispro:
            ddist = constants.CA_DIST_CISPRO - dist
        else:
            ddist = constants.CA_DIST - dist

        ddist2 = ddist*ddist
        new_e_pot += constants.CA_K * ddist2
        ene[0] += constants.CA_K * ddist2
        if calc_gradient:
            grad = ddist * (-2.0 * constants.CA_K) / dist
            gradient[i][0] -= grad * dx
            gradient[i][1] -= grad * dy
            gradient[i][2] -= grad * dz
            gradient[i-1][0] += grad * dx
            gradient[i-1][1] += grad * dy
            gradient[i-1][2] += grad * dz

    # CALC_C_ALPHA_XVOL
    for i in range(chain_length):
        for j in range(i + 3, chain_length):
            dx = new_c_alpha[i][0] - new_c_alpha[j][0]
            dy = new_c_alpha[i][1] - new_c_alpha[j][1]
            dz = new_c_alpha[i][2] - new_c_alpha[j][2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < constants.CA_XVOL_DIST:
                ddist = dist - constants.CA_XVOL_DIST
                ddist2 = ddist * ddist
                new_e_pot += constants.CA_XVOL_K * ddist2
                ene[3] += constants.CA_XVOL_K * ddist2
                if calc_gradient:
                    grad = ddist * (2.0 * constants.CA_XVOL_K) / dist
                    gradient[i][0] -= grad * dx
                    gradient[i][1] -= grad * dy
                    gradient[i][2] -= grad * dz
                    gradient[j][0] += grad * dx
                    gradient[j][1] += grad * dy
                    gradient[j][2] += grad * dz

    # CALC_C_ALPHA_ANGLES
    for i in range(1, chain_length - 1):
        r12x = new_c_alpha[i-1][0] - new_c_alpha[i][0]
        r12y = new_c_alpha[i-1][1] - new_c_alpha[i][1]
        r12z = new_c_alpha[i-1][2] - new_c_alpha[i][2]
        r32x = new_c_alpha[i+1][0] - new_c_alpha[i][0]
        r32y = new_c_alpha[i+1][1] - new_c_alpha[i][1]
        r32z = new_c_alpha[i+1][2] - new_c_alpha[i][2]

        d12 = math.sqrt(r12x*r12x + r12y*r12y + r12z*r12z)
        d32 = math.sqrt(r32x*r32x + r32y*r32y + r32z*r32z)

        if d12 == 0 or d32 == 0:
            continue

        cos_theta = (r12x*r32x + r12y*r32y + r12z*r32z) / (d12 * d32)
        cos_theta = min(1.0, max(-1.0, cos_theta))
        theta = math.acos(cos_theta)

        if 80 * constants.DEGRAD < theta < 150 * constants.DEGRAD:
            diff = 0.0
        elif theta <= 80 * constants.DEGRAD:
            diff = theta - 80 * constants.DEGRAD
        else:
            diff = theta - 150 * constants.DEGRAD

        new_e_pot += constants.CA_ANGLE_K * diff * diff
        ene[2] += constants.CA_ANGLE_K * diff * diff

        if calc_gradient:
            sin_theta = math.sqrt(1.0 - cos_theta*cos_theta)
            if sin_theta == 0:
                continue

            d12inv = 1.0 / d12
            d32inv = 1.0 / d32

            c = diff * (-2.0 * constants.CA_ANGLE_K) / sin_theta

            c1 = c * d12inv
            c2 = c * d32inv

            f1x = c1 * (r12x * d12inv * cos_theta - r32x * d32inv)
            f1y = c1 * (r12y * d12inv * cos_theta - r32y * d32inv)
            f1z = c1 * (r12z * d12inv * cos_theta - r32z * d32inv)

            f3x = c2 * (r32x * d32inv * cos_theta - r12x * d12inv)
            f3y = c2 * (r32y * d32inv * cos_theta - r12y * d12inv)
            f3z = c2 * (r32z * d32inv * cos_theta - r12z * d12inv)

            f2x = -f1x - f3x
            f2y = -f1y - f3y
            f2z = -f1z - f3z

            gradient[i-1][0] += f1x
            gradient[i-1][1] += f1y
            gradient[i-1][2] += f1z

            gradient[i][0] += f2x
            gradient[i][1] += f2y
            gradient[i][2] += f2z

            gradient[i+1][0] += f3x
            gradient[i+1][1] += f3y
            gradient[i+1][2] += f3z

    return new_e_pot
