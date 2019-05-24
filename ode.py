from math import sin
from math import cos
from math import pi
from numpy import array
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt


# Q — Euler angles matrix
def q_matr(angles_vec):
    psi = angles_vec[0]
    theta = angles_vec[1]
    phi = angles_vec[2]
    return array([[cos(psi) * cos(theta), cos(psi) * sin(phi) * sin(theta) - cos(phi) * sin(psi),
                   sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)],
                  [cos(theta) * sin(psi), cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta),
                   cos(phi) * sin(psi) * sin(theta) - cos(psi) * sin(phi)],
                  [-sin(theta), cos(theta) * sin(phi), cos(phi) * cos(theta)]])


# P = Q_desired.T * Q -> I — goal of control
def p_matr(q):
    return q_des.T @ q


# Delta — goal of control
def delta_matr(q):
    p = p_matr(q)
    return p - p.T


# Omega — scew symmetric matrix made from angular velocities in the body frame
def omega_matr(omega_vec):
    return vec_to_scew(omega_vec)


# Theta — matrix that contains components of dot Omega matrix
def theta_matr(omega_vec):
    p = omega_vec[0]
    q = omega_vec[1]
    r = omega_vec[2]
    return vec_to_scew(
        array([((I_y - I_z) / I_x) * q * r, ((I_z - I_x) / I_y) * p * r, ((I_x - I_y) / I_z) * p * q]))


# Tau — matrix of controls, solution of matrix equation
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

    tau_norm = np.linalg.solve(a, result)
    return tau_norm


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


# Transformation of vector to scew-symmetric matrix
def vec_to_scew(vec):
    return array([[0, -vec[2], vec[1]],
                  [vec[2], 0, -vec[0]],
                  [-vec[1], vec[0], 0]])


# Transformation of scew-symmetric matrix to vector
def scew_to_vec(matr):
    return array([matr[2][1], matr[0][2], matr[1][0]])


# Quadrotor constants
I_x = I_y = 7.5 * (10 ** (-3))
I_z = 1.3 * (10 ** (-2))
m = 1
b = 3.13 * (10 ** (-5))
d = 7.5 * (10 ** (-7))
l = 0.25
I_vec = array([I_x, I_y, I_z])  # Inertia moment vector
I_matr = np.eye(3) * I_vec  # Inertia moment matrix
angles_des = array([0, 0, 0])  # Desired angles

q_des = q_matr(angles_des)  # Desired Q

lmbd = 10.0  # Desired rate of convergence

angles_0 = array([7 * pi / 16, 7 * pi / 16, 7 * pi / 16])  # Initial angles
omega_0 = array([0, 0, 0])  # Initial angular velocities in the body frame




# Solving ODE

t_span = np.linspace(0, 10 / lmbd, 100)  # Time diapason
y_0 = np.concatenate((q_matr(angles_0).reshape(9), omega_0))  # Initial state


sol = spi.odeint(dy_dt, y_0, t_span)
delta_sol = array(list(map(q_vec_to_delta_vec, sol[:, :9])))

plt.xlabel("t")
plt.ylabel("Delta")
plt.plot(t_span, delta_sol[:, 0], 'b', label='Delta_1(t)')
plt.plot(t_span, delta_sol[:, 1], 'g', label='Delta_2(t)')
plt.plot(t_span, delta_sol[:, 2], 'r', label='Delta_3(t)')
plt.legend(loc='best')
plt.grid()
plt.show()



