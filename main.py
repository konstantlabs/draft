import time
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    N = 15

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
    dt = 0.2

    for k in range(N):
        J += ca.mtimes((x[:, k] - r).T, ca.mtimes(Q, (x[:, k] - r))) \
            + ca.mtimes(u[:, k].T, ca.mtimes(R, u[:, k]))

        x_next = x[:, k] + f(x[:, k], u[:, k]) * dt
        opti.subject_to(x[:, k+1] == x_next)

    opti.minimize(J)

    opti.subject_to(x[:, 0] == x0)

    # opti.subject_to(x[0, :] >= -2)
    # opti.subject_to(x[0, :] <= 2)
    # opti.subject_to(x[1, :] >= -2)
    # opti.subject_to(x[1, :] <= 2)
    opti.subject_to(x[2, :] >= -np.inf)
    opti.subject_to(x[2, :] <= np.inf)
    opti.subject_to(u[0, :] >= -1)
    opti.subject_to(u[0, :] <= 1)
    opti.subject_to(u[1, :] >= -np.pi / 4)
    opti.subject_to(u[1, :] <= np.pi / 4)

    opti.set_value(x0, ca.vertcat(0, 0, 0))
    opti.set_value(r, ca.vertcat(1.5, 15, 0))

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

    x_history = np.zeros((3, int(T / dt)))
    u_history = np.zeros((2, int(T / dt)))

    x_current = opti.value(x0)
    while np.linalg.norm(x_current - opti.value(r)) > 1e-3 and k < T / dt:
        error = np.linalg.norm(x_current - opti.value(r))
        print(f"Step = {k} Timestep = {k * dt:.2f} Error = {error:.4f}")

        opti.set_value(x0, x_current)
        sol = opti.solve()
        u_opt = sol.value(u)

        x_history[:, k] = x_current
        u_history[:, k] = u_opt[:, 0]

        x_current = x_current + f(x_current, u_opt[:, 0]).full().flatten() * dt

        k += 1

    t = np.arange(0, T, dt)

    x_des, y_des, theta_des = opti.value(r)
    xt_des = np.full(len(t), x_des)
    yt_des = np.full(len(t), y_des)
    thetat_des = np.full(len(t), theta_des)

    plt.figure(figsize=(8, 6))
    plt.plot(x_history[0, :], x_history[1, :])
    plt.plot(x_des, y_des, 'ro', markersize=10)
    plt.xlabel('$x$ [$m$]')
    plt.ylabel('$y$ [$m$]')
    plt.title('Unicycle Trajectory (MPC)')
    plt.grid(True)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, x_history[0, :], label='$x$')
    plt.plot(t, xt_des, '--', label='$x_d$')
    plt.plot(t, x_history[1, :], label='$y$')
    plt.plot(t, yt_des, '--', label='$y_d$')
    plt.xlabel('Time [$s$]')
    plt.ylabel('Position [$m$]')
    plt.title('Vehicle Position')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, np.mod(x_history[2, :], 2 * np.pi), label=r'$\theta$')
    plt.xlabel('Time [$s$]')
    plt.ylabel(r'$\theta$ [$rad$]')
    plt.title('Vehicle Orientation')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, u_history[0, :], label=r'$v$')
    plt.plot(t, u_history[1, :], label=r'$\omega$')
    plt.xlabel('Time [$s$]')
    plt.ylabel('Control Input')
    plt.title('Vehicle Control')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
