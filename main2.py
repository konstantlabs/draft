import time
import casadi as ca
import numpy as np


if __name__ == '__main__':
    N = 10

    opti = ca.Opti()
    x = opti.variable(3, N + 1)
    u = opti.variable(2, N)
    x0 = opti.parameter(3)
    r = opti.parameter(3)

    J = 0  # objective function

    # vehicle dynamics
    f = lambda x,u: ca.vertcat(
        u[0] * ca.cos(x[2]),
        u[0] * ca.sin(x[2]),
        u[1],
    )

    Q = np.diag([1.0, 5.0, 0.1])  # state weighing matrix
    R = np.diag([0.5, 0.05])  # controls weighing matrix

    T = 20
    dt = T / N
    for k in range(N):
        J += ca.mtimes((x[:, k] - r).T, ca.mtimes(Q, (x[:, k] - r))) \
            + ca.mtimes(u[:, k].T, ca.mtimes(R, u[:, k]))

        x_next = x[:, k] + f(x[:, k], u[:, k]) * dt
        opti.subject_to(x[:, k+1] == x_next)

    opti.minimize(J)

    opti.subject_to(x[:, 0] == x0)

    opti.subject_to(x[0, :] >= -2)
    opti.subject_to(x[0, :] <= 2)
    opti.subject_to(x[1, :] >= -2)
    opti.subject_to(x[1, :] <= 2)
    opti.subject_to(x[2, :] >= -np.inf)
    opti.subject_to(x[2, :] <= np.inf)
    opti.subject_to(u[0, :] >= -0.6)
    opti.subject_to(u[0, :] <= 0.6)
    opti.subject_to(u[1, :] >= -np.pi / 4)
    opti.subject_to(u[1, :] <= np.pi / 4)

    opti.set_value(x0, ca.vertcat(0, 0, 0))
    opti.set_value(r, ca.vertcat(1.5, -2, np.pi / 2))

    k = 0
    x_current = x0

    p_opts = {
        "expand": True,
    }
    s_opts = {
        "max_iter": 1000,
        "print_level": 0,
        "acceptable_tol": 1e-8,
        "acceptable_obj_change_tol": 1e-6
    }

    opti.solver("ipopt", p_opts, s_opts)

    x_current = opti.value(x0)
    while np.linalg.norm(x_current - opti.value(r)) > 1e-2 and k < T / dt:
        error = np.linalg.norm(x_current - opti.value(r))
        print(f"Step {k}: Error = {error:.4f}")

        opti.set_value(x0, x_current)
        sol = opti.solve()
        u_opt = sol.value(u)
        x_current = x_current + dt * f(x_current, u_opt[:, 0])

        k += 1
