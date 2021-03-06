import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
from math import sin, cos, atan2, asin, pi
from numpy import array


# Transformation of vector to scew-symmetric matrix
def vec_to_scew(vec):
    return array([[0, -vec[2], vec[1]],
                  [vec[2], 0, -vec[0]],
                  [-vec[1], vec[0], 0]])


# Transformation of scew-symmetric matrix to vector
def scew_to_vec(matr):
    return array([matr[2][1], matr[0][2], matr[1][0]])


# Q is an Euler angles matrix
def q_matr(angles_vec):
    psi = angles_vec[0]
    theta = angles_vec[1]
    phi = angles_vec[2]
    return array([[cos(psi) * cos(theta), cos(psi) * sin(phi) * sin(theta) - cos(phi) * sin(psi),
                   sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)],
                  [cos(theta) * sin(psi), cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta),
                   cos(phi) * sin(psi) * sin(theta) - cos(psi) * sin(phi)],
                  [-sin(theta), cos(theta) * sin(phi), cos(phi) * cos(theta)]])


# P = Q_desired.T * Q -> I is a goal of control
def p_matr(q):
    return q_des.T @ q


# Delta is a goal of control
def delta_matr(q):
    return p_matr(q) - p_matr(q).T


# Omega is a scew symmetric matrix made from angular velocities in the body frame
def omega_matr(omega_vec):
    return vec_to_scew(omega_vec)


# Theta is a matrix that contains components of dot Omega matrix
def theta_matr(omega_vec):
    p = omega_vec[0]
    q = omega_vec[1]
    r = omega_vec[2]
    return vec_to_scew(
        array([((I_y - I_z) / I_x) * q * r, ((I_z - I_x) / I_y) * p * r, ((I_x - I_y) / I_z) * p * q]))


# Tau is a matrix of controls, solution of matrix equation
# Normalized by inertia moment matrix
def tau_norm_matr(q_matr, omega_vec, lmbd):
    p = p_matr(q_matr)
    theta = theta_matr(omega_vec)
    omega = omega_matr(omega_vec)

    a = array([[p[1][1] + p[2][2], -p[1][0], -p[2][0]],
               [-p[0][1], p[0][0] + p[2][2], -p[2][1]],
               [-p[0][2], -p[1][2], p[0][0] + p[1][1]]])

    b = -p @ omega @ omega - p @ theta + omega @ omega @ p.T - omega @ p.T - 2 * lmbd * (p @ omega + omega @ p.T) - (
            lmbd ** 2) * (p - p.T)
    result = scew_to_vec(b)

    det_a = np.linalg.det(a)
    if det_a == 0:
        return array([0, 0, 0])

    tau_norm = np.linalg.solve(a, result)
    return tau_norm


# Approximation function
def is_close(x, y, rt=1.e-5, at=1.e-8):
    return abs(y - x) <= rt * abs(y) + at


# Approximation function
def clean_sin(sin_angle):
    return min(1, max(sin_angle, -1))


# Euler matrix to angles
def q_to_angles(q_matr):
    phi = 0.0
    if is_close(q_matr[2, 0], 1.0):
        psi = atan2(-q_matr[0, 1], q_matr[0, 2])
        theta = -pi / 2.0
    elif is_close(q_matr[2, 0], -1.0):
        psi = atan2(q_matr[0, 1], q_matr[0, 2])
        theta = pi / 2.0
    else:
        theta = -asin(clean_sin(q_matr[2, 0]))
        phi = atan2(q_matr[2, 1] / cos(theta), q_matr[2, 2] / cos(theta))
        psi = atan2(q_matr[1, 0] / cos(theta), q_matr[0, 0] / cos(theta))
    return psi, theta, phi


# Returns true if one of given angle is too big
def has_big_angle(angles, angle_0=1.5):
    for angle in angles:
        if angle <= -angle_0 or angle >= angle_0:
            return True
    return False


# State vector time derivative function
def dy_dt(y, t):
    q = y[:9].reshape(3, 3)
    omega_vector = y[9:]
    omega_matrix = omega_matr(omega_vector)
    tau_norm = tau_norm_matr(q, omega_vector, lmbd)

    dq_dt = q @ omega_matrix
    dw_dt = tau_norm - np.cross(I_vec, omega_vector) @ np.linalg.inv(I_matr)
    return np.concatenate((dq_dt.reshape(9), dw_dt))


# Convert q coordinates to vector delta
def q_vec_to_delta_vec(q_vec):
    q = array(q_vec).reshape(3, 3)
    delta = delta_matr(q)
    return scew_to_vec(delta).tolist()


# Convert state vector y to saturated squared rotor speeds
def y_to_rotor_sq_sat(y):
    y = array(y)
    q = y[:9].reshape(3, 3)
    omega_vector = y[9:]
    tau_norm = tau_norm_matr(q, omega_vector, lmbd)
    tau = I_matr @ tau_norm
    r = array([tau[0] / (b * l), tau[1] / (b * l), tau[2] / (d * l), (m * g) / b])

    rotor_sq = array([(-2 * r[0] - r[2] + r[3]) / 4,
                      (-2 * r[1] + r[2] + r[3]) / 4,
                      (2 * r[0] - r[2] + r[3]) / 4,
                      (2 * r[1] + r[2] + r[3]) / 4])
    for i in range(len(rotor_sq)):
        if rotor_sq[i] < 0:
            rotor_sq[i] = 0
        if rotor_sq[i] > sat_upper_boundary:
            rotor_sq[i] = sat_upper_boundary
    return rotor_sq


# Convert saturated square rotor speeds to saturated controls
def sat_tau(sat_omega_sq):
    return array([b * l * (-sat_omega_sq[0] + sat_omega_sq[2]),
                  b * l * (-sat_omega_sq[1] + sat_omega_sq[3]),
                  d * l * (-sat_omega_sq[0] + sat_omega_sq[1] - sat_omega_sq[2] + sat_omega_sq[3])])


# State vector time derivative function with saturated control
def dy_sat_dt(y, t):
    q = y[:9].reshape(3, 3)
    omega_vector = y[9:]
    omega_matrix = omega_matr(omega_vector)
    tau_norm_sat = sat_tau(y_to_rotor_sq_sat(y))

    dq_dt = q @ omega_matrix
    dw_dt = tau_norm_sat @ np.linalg.inv(I_matr) - np.cross(I_vec, omega_vector) @ np.linalg.inv(I_matr)
    return np.concatenate((dq_dt.reshape(9), dw_dt))


# Integrating ODE without boundaries, building graphs for Delta(t)
def build_delta_solution(angles_init, omega_init):
    y_init = np.concatenate((q_matr(angles_init).reshape(9), omega_init))  # Initial state
    solution = spi.odeint(dy_dt, y_init, t_span)
    delta_sol = array(list(map(q_vec_to_delta_vec, solution[:, :9])))

    title = "Lambda: {}, Angles: [{}, {}, {}], Omega: [{}, {}, {}]" \
        .format(
        lmbd,
        round(angles_init[0], 2), round(angles_init[1], 2), round(angles_init[2], 2),
        round(omega_init[0], 2), round(omega_init[1], 2), round(omega_init[2], 2))

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("Delta")
    plt.plot(t_span, delta_sol[:, 0], 'b', label='Delta_1(t)')
    plt.plot(t_span, delta_sol[:, 1], 'g', label='Delta_2(t)')
    plt.plot(t_span, delta_sol[:, 2], 'r', label='Delta_3(t)')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    return delta_sol


# Constructing saturated rotor speeds, building graphs for Rotor_sq_speed(t)
def build_saturated_rotor_speeds(angles_init, omega_init):
    y_init = np.concatenate((q_matr(angles_init).reshape(9), omega_init))  # Initial state
    solution = spi.odeint(dy_dt, y_init, t_span)
    rotor_sol_sat = array(list(map(y_to_rotor_sq_sat, solution)))

    title = "Lambda: {}, Angles: [{}, {}, {}], Omega: [{}, {}, {}]" \
        .format(
        lmbd,
        round(angles_init[0], 2), round(angles_init[1], 2), round(angles_init[2], 2),
        round(omega_init[0], 2), round(omega_init[1], 2), round(omega_init[2], 2))

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("Square rotor speeds")
    plt.plot(t_span, rotor_sol_sat[:, 0], 'b', label='Rotor_sq_1(t)')
    plt.plot(t_span, rotor_sol_sat[:, 1], 'g', label='Rotor_sq_2(t)')
    plt.plot(t_span, rotor_sol_sat[:, 2], 'r', label='Rotor_sq_3(t)')
    plt.plot(t_span, rotor_sol_sat[:, 3], 'y', label='Rotor_sq_4(t)')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    return rotor_sol_sat


# Integrating ODE with boundaries, building graphs for Delta_sat(t)
def build_delta_sat_solution(angles_init, omega_init):
    y_init = np.concatenate((q_matr(angles_init).reshape(9), omega_init))  # Initial state
    solution_sat = spi.odeint(dy_sat_dt, y_init, t_span)
    delta_sol_sat = array(list(map(q_vec_to_delta_vec, solution_sat[:, :9])))

    title = "Lambda: {}, Angles: [{}, {}, {}], Omega: [{}, {}, {}]" \
        .format(
        lmbd,
        round(angles_init[0], 2), round(angles_init[1], 2), round(angles_init[2], 2),
        round(omega_init[0], 2), round(omega_init[1], 2), round(omega_init[2], 2))

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("Delta_sat")
    plt.plot(t_span, delta_sol_sat[:, 0], 'b', label='Delta_sat_1(t)')
    plt.plot(t_span, delta_sol_sat[:, 1], 'g', label='Delta_sat_2(t)')
    plt.plot(t_span, delta_sol_sat[:, 2], 'r', label='Delta_sat_3(t)')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    return delta_sol_sat


# Method that marks if delta_sat_solution converges for given initial conditions
def is_delta_sat_converges(angles_init, omega_init):
    y_init = np.concatenate((q_matr(angles_init).reshape(9), omega_init))  # Initial state
    solution_sat = spi.odeint(dy_sat_dt, y_init, t_span)
    angles_data = [q_to_angles(q.reshape(3, 3)) for q in solution_sat[:, :9]]
    for angles_vec in angles_data:
        if has_big_angle(angles_vec):
            return False
    return True


# Check if element is in array
def is_elem_in_array(test, array):
    return any(np.array_equal(x, test) for x in array)


# Method that builds inner inclusion of region of attraction for given angular and angular speed diapasons
def build_roa_with_border(angles_s, omega_s):
    roa = array([0, 0, True])
    border = array([0, 0])

    for om in omega_s:
        prev_bool = False
        prev_roa = array([])
        for an in angles_s:
            data = array([an, om, is_delta_sat_converges(array([an, an, an]), array([om, om, om]))])
            roa = np.vstack((roa, data))

            if not prev_bool and data[2]:
                border = np.vstack((border, array([an, om])))
            elif prev_bool and not data[2]:
                border = np.vstack((border, prev_roa[:2]))

            prev_bool = data[2]
            prev_roa = data

    roa = array(list(filter(lambda x: x[2] == 1, roa[1:])))
    roa = roa[:, :2]

    for an in angles_s:
        prev_bool = False
        prev_roa = array([])
        for om in omega_s:
            curr_bool = is_elem_in_array(array([an, om]), roa)

            if not prev_bool and curr_bool:
                border = np.vstack((border, array([an, om])))
            elif prev_bool and not curr_bool:
                border = np.vstack((border, prev_roa))

            prev_bool = curr_bool
            prev_roa = array([an, om])

    border = border[1:]
    border = np.unique(border, axis=0)

    plt.title("Lambda: {}".format(round(lmbd, 2)))
    plt.xlabel("Angle")
    plt.ylabel("Omega")
    plt.plot(roa[:, 0], roa[:, 1], 'bo')
    plt.plot(border[:, 0], border[:, 1], 'ro')
    plt.grid()
    plt.show()

    return roa, border


# Quadrotor constants

I_x = I_y = 1.8 * (10 ** (-2))
I_z = 3.6 * (10 ** (-2))
m = 0.5
b = 2.04 * (10 ** (-4))
d = 2.04 * (10 ** (-5))
l = 0.3
g = 9.83  # Gravity constant
I_vec = array([I_x, I_y, I_z])  # Inertia moment vector
I_matr = np.eye(3) * I_vec  # Inertia moment matrix
angles_des = array([0, 0, 0])  # Desired angles

q_des = q_matr(angles_des)  # Desired Q

lmbd = 1.0  # Desired rate of convergence
sat_upper_boundary = 25600  # Upper boundary for square rotor angular speed

angles_0 = array([-1, -1, -1])  # Initial angles
omega_0 = array([-1.5, -1.5, -1.5])  # Initial angular velocities in the body frame

t_span = np.linspace(0, 25 / lmbd, 1000)  # Time diapason

omega_seq = np.linspace(-4, 4, 100)
angles_seq = np.linspace(-1.5, 1.5, 100)


# Execution part. Algorithms run there

build_delta_solution(angles_0, omega_0)
build_saturated_rotor_speeds(angles_0, omega_0)
build_delta_sat_solution(angles_0, omega_0)
build_roa_with_border(angles_seq, omega_seq)
