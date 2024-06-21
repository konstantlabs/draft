import time
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline


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


def unicycle_kinematics(x, u):
    theta = x[2]
    v = u[0]
    omega = u[1]

    return ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        omega,
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
        r / (2 * d) * (u_r - u_l),
    )


def spline_trajectory(t, points):
    cs = CubicSpline(points[1], points[0])

    xy = cs(t)
    dxy = np.diff(xy, axis=0)
    angles = np.atan2(dxy[:, 1], dxy[:, 0])
    angles = np.insert(angles, 0, angles[0]).reshape(-1, 1)

    return np.hstack((xy, angles)).T


if __name__ == "__main__":
    N = 10
    num_states = 5
    num_inputs = 2

    opti = ca.Opti()
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N)
    x0 = opti.parameter(num_states)
    r = opti.parameter(num_states, N + 1)

    J = 0  # objective function

    # vehicle dynamics
    f = unicycle_dynamics

    Q = np.diag([3000, 3000, 10.0])  # state weighing matrix
    R = np.diag([0.1, 0.01])  # controls weighing matrix

    T = 100
    dt = 0.2

    for k in range(N):
        J += (x[:3, k] - r[:3, k]).T @ Q @ (x[:3, k] - r[:3, k]) + u[:, k].T @ R @ u[
            :, k
        ]
        x_next = x[:, k] + f(x[:, k], u[:, k]) * dt
        opti.subject_to(x[:, k + 1] == x_next)

    opti.minimize(J)
    opti.subject_to(x[:, 0] == x0)
    opti.subject_to(u[0, :] >= -5)
    opti.subject_to(u[0, :] <= 5)
    opti.subject_to(u[1, :] >= -10)
    opti.subject_to(u[1, :] <= 10)

    opti.set_value(x0, ca.vertcat(1, 0, np.pi, 0, 0))

    p_opts = {"expand": True}
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

    t = np.arange(0, T, dt)

    spline_points = (
        np.array(
            [
                [0, 0],
                [0, 4],
                [0, 5],
                [1, 7],
                [2, 8],
                [8, 0],
            ]
        ),
        np.linspace(0, T, 6),
    )

    ref_traj = spline_trajectory(t, spline_points)

    ref_traj = np.vstack(
        (
            ref_traj,
            0.0 * np.ones((1, len(t))),
            0.0 * np.ones((1, len(t))),
        )
    )

    opti.set_value(r, ref_traj[:, : N + 1])

    k = 0
    start = time.time()
    x_current = opti.value(x0)

    while np.linalg.norm(x_current[:3] - ref_traj[:3, -1]) > 1e-3 and k < T / dt:
        inner_start = time.time()

        error = np.linalg.norm(x_current[:3] - ref_traj[:3, -1])

        # horizon reference trajectory
        ref = ref_traj[:, k : N + 1 + k]

        if ref.shape[1] < N + 1:
            padding = N + 1 - ref.shape[1]
            ref = np.pad(ref, ((0, 0), (0, padding)), mode="edge")

        opti.set_value(r, ref)

        error_history[:3, k] = np.linalg.norm(
            x_current[:3].reshape(-1, 1) - opti.value(r)[:3, 0].reshape(-1, 1),
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

    plt.figure(figsize=(8, 6))
    plt.plot(x_history[0, :], x_history[1, :])
    plt.plot(ref_traj[0, :], ref_traj[1, :], "--", color="red")
    plt.plot(ref_traj[0, -1], ref_traj[1, -1], "ro", markersize=10)
    plt.xlabel("$x$ [$m$]")
    plt.ylabel("$y$ [$m$]")
    plt.title("Unicycle Trajectory Tracking w/ MPC")
    plt.grid(True)

    plt.figure(figsize=(8, 8))
    plt.subplot(3, 3, 1)
    plt.plot(t, x_history[0, :], label="$x$")
    plt.plot(t, ref_traj[0, :], "--", label="$x_d$")
    plt.plot(t, x_history[1, :], label="$y$")
    plt.plot(t, ref_traj[1, :], "--", label="$y_d$")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Position [$m$]")
    plt.title("Vehicle Position")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 2)
    plt.plot(t, np.mod(ref_traj[2, :], 2 * np.pi), "--", label=r"$\theta_d$")
    plt.plot(t, np.mod(x_history[2, :], 2 * np.pi), label=r"$\theta$")
    plt.xlabel("Time [$s$]")
    plt.ylabel(r"$\theta$ [$rad$]")
    plt.title("Vehicle Orientation")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 3)
    plt.plot(t, u_history[0, :], label="$a$")
    plt.plot(t, u_history[1, :], label=r"$\alpha$")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Control Input")
    plt.title("Vehicle Control")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 4)
    plt.plot(t, error_history[0, :], label="error$_x$")
    plt.plot(t, error_history[1, :], label="error$_y$")
    plt.plot(t, error_history[2, :], label=r"error$_\theta$")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Error [$m$]")
    plt.title("Pose error")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 5)
    plt.plot(t, x_history[3, :])
    plt.plot(t, ref_traj[3, :], "--", color="red")
    plt.xlabel("$t$ [$s$]")
    plt.ylabel(r"$v$ [$\frac{m}{s}$]")
    plt.title("Vehicle Linear Velocity")
    plt.grid(True)

    plt.subplot(3, 3, 6)
    plt.plot(t, np.mod(x_history[4, :], 2 * np.pi))
    plt.xlabel("$t$ [$s$]")
    plt.ylabel(r"$\omega$ [$\frac{rad}{s}$]")
    plt.title("Vehicle Angular Velocity")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
