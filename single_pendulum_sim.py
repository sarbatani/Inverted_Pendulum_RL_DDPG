import numpy as np

def calc_mass(m, l_rod):
    M = np.array([[m, 0., 0.], [0., m, 0.], [0., 0., (1. / 12.) * m * l_rod * l_rod]])
    return M


def calc_rot(q):
    theta = q[2][0]
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))
    return R

# Location of the rev joint - 1st end
def calc_rD1(q, l_rod):
    R = calc_rot(q)
    rD = np.array([q[0][0], q[1][0], 0.]) + np.matmul(R, np.array([-l_rod / 2., 0., 0.]))
    return rD

def calc_A(q, l):
    rG1 = np.array([q[0][0], q[1][0], 0.])

    rOrg = calc_rD1(q, l)
    rOrgG1 = rOrg - rG1

    A = np.array([[0., 1., rOrgG1[0]]])
    return A

def calc_Qg(m_p, g):
    Qg = m_p*g
    return Qg

def calcDrivingForce(q, l, f):
    fVec = np.array([[f],[0.],[(f*np.sin(q[2][0]))*(l/2.0)]])
    return fVec


def step_sim(f, q, qd, mass, l, h):
    # print(f)
    g = np.array([[0.], [-9.81], [0.]])
    m = calc_mass(mass, l)
    W = np.zeros((4, 4))
    b = np.zeros((4, 1))
    Qg = calc_Qg(mass, g)

    Org = np.array([0., 0., 0.])

    # Compliance
    # C = 1.e-8 * np.identity(6)

    # start the step calculations
    rO = calc_rD1(q, l)
    phi = -(rO - Org) / h

    A = calc_A(q, l)
    W[0:3, 0:3] = m
    W[0:3, 3:4] = A.transpose()
    W[3:4, 0:3] = A
    # W[9:15, 9:15] = C

    fVec = calcDrivingForce(q,l,f)
    b[0:3, ] = h * Qg + np.matmul(m, qd) + h * fVec
    b[3,] = np.array([phi[1]])

    X = np.linalg.solve(W, b)
    qd = X[0:3, ]
    qp = q
    q = qp + h * qd

    # print(q[2][0])

    if q[2][0] > 2. * np.pi:
        q[2][0] = q[2][0] - 2.*np.pi
    if q[2][0] < -2. * np.pi:
        q[2][0] = q[2][0] + 2. * np.pi

    return q, qd



