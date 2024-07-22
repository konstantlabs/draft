import time
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def unicycle_kinematics(x, u):
    theta = x[2]
    v = u[0]
    omega = u[1]

    return ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        omega,
    )


def unicycle_dynamics(x, u):
    theta = x[2]
    v = x[3]
    omega = x[4]
    a = u[0]
    alpha = u[1]

    return ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        omega,
        a,
        alpha,
    )


def diff_drive_kinematics(x, u):
    r = 1
    d = 1
    theta = x[2]
    u_l = u[0] - u[1]
    u_r = u[0] + u[1]

    return ca.vertcat(
        r / 2 * ca.cos(theta) * (u_r + u_l),
        r / 2 * ca.sin(theta) * (u_r + u_l),
        r / d * (u_r - u_l),
    )


if __name__ == "__main__":
    N = 20
    num_states = 5
    num_inputs = 2

    opti = ca.Opti()
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N)
    x0 = opti.parameter(num_states)
    r = opti.parameter(num_states)

    J = 0  # objective function

    # vehicle dynamics
    f = unicycle_dynamics

    Q = np.diag([1.0, 5.0, 0.1, 10.0, 5.0])  # state weighing matrix
    R = np.diag([0.5, 0.05])  # controls weighing matrix
    a = 0.5

    T = 100
    dt = 0.2

    for k in range(N):
        J += (x[:, k] - r).T @ Q @ (x[:, k] - r) + u[:, k].T @ R @ u[:, k]
        x_next = x[:, k] + f(x[:, k], u[:, k]) * dt
        opti.subject_to(x[:, k + 1] == x_next)

    opti.minimize(J)
    opti.subject_to(x[:, 0] == x0)
    opti.subject_to(u[0, :] >= -2)
    opti.subject_to(u[0, :] <= 2)
    opti.subject_to(u[1, :] >= -2)
    opti.subject_to(u[1, :] <= 2)

    opti.set_value(x0, ca.vertcat(0, 0, 0, 0, 0))
    opti.set_value(r, ca.vertcat(1.5, 15, 0, 0, 0))

    k = 0

    p_opts = {
        "expand": True,
    }
    s_opts = {
        "max_iter": 1000,
        "print_level": 0,
        "acceptable_tol": 1e-8,
        "acceptable_obj_change_tol": 1e-6,
    }

    opti.solver("ipopt", p_opts, s_opts)

    x_history = np.zeros((num_states, int(T / dt)))
    u_history = np.zeros((num_inputs, int(T / dt)))
    error_history = np.zeros((num_states, int(T / dt)))

    start = time.time()
    x_current = opti.value(x0)
    while np.linalg.norm(x_current - opti.value(r)) > 1e-3 and k < T / dt:
        inner_start = time.time()

        error = np.linalg.norm(x_current - opti.value(r))

        error_history[:, k] = np.linalg.norm(
            x_current.reshape(-1, 1) - opti.value(r).reshape(-1, 1),
            axis=1,
        )

        print(f"Step = {k} Timestep = {k * dt:.2f} Error = {error:.4f}")

        sol = opti.solve()
        u_opt = sol.value(u)
        x_history[:, k] = x_current
        u_history[:, k] = u_opt[:, 0]

        x_current += f(x_current, u_opt[:, 0]).full().flatten() * dt

        opti.set_value(x0, x_current)

        k += 1

        print(f"MPC time step: {(time.time() - inner_start):.4f}")

    print(f"Total MPC time: {(time.time() - start):.2f}")

    t = np.arange(0, T, dt)

    x_des, y_des, theta_des, v_des, omega_des = opti.value(r)
    xt_des = np.full(len(t), x_des)
    yt_des = np.full(len(t), y_des)
    thetat_des = np.full(len(t), theta_des)
    v_des = np.full(len(t), v_des)
    omegat_des = np.full(len(t), omega_des)

    plt.figure(figsize=(8, 6))
    plt.plot(x_history[0, :], x_history[1, :])
    plt.plot(x_des, y_des, "ro", markersize=10)
    plt.xlabel("$x$ [$m$]")
    plt.ylabel("$y$ [$m$]")
    plt.title("Unicycle Trajectory (MPC)")
    plt.grid(True)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    plt.plot(t, x_history[0, :], label="$x$")
    plt.plot(t, xt_des, "--", label="$x_d$")
    plt.plot(t, x_history[1, :], label="$y$")
    plt.plot(t, yt_des, "--", label="$y_d$")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Position [$m$]")
    plt.title("Vehicle Position")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(t, thetat_des, "--", label=r"$\theta_d$")
    plt.plot(t, np.mod(x_history[2, :], 2 * np.pi), label=r"$\theta$")
    plt.xlabel("Time [$s$]")
    plt.ylabel(r"$\theta$ [$rad$]")
    plt.title("Vehicle Orientation")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(t, v_des, "--", label="$v_d$")
    plt.plot(t, x_history[3, :], label="$v$")
    plt.xlabel("Time [$s$]")
    plt.ylabel("$v$ [$rad$]")
    plt.title("Vehicle Velocity")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(t, omegat_des, "--", label=r"$\omega_d$")
    plt.plot(t, np.mod(x_history[4, :], 2 * np.pi), label=r"$\omega$")
    plt.xlabel("Time [$s$]")
    plt.ylabel(r"$v$ [$\frac{rad}{s}$]")
    plt.title("Vehicle Angular Velocity")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(t, u_history[0, :], label="$a$")
    plt.plot(t, u_history[1, :], label=r"$\alpha$")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Control Input")
    plt.title("Vehicle Control")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(t, error_history[0, :], label="error$_x$")
    plt.plot(t, error_history[1, :], label="error$_y$")
    plt.plot(t, error_history[2, :], label=r"error$_\theta$")
    plt.plot(t, error_history[3, :], label="error$_v$")
    plt.plot(t, error_history[4, :], label=r"error$_\omega$")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Error [$m$]")
    plt.title("Pose error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
