import time
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from scipy.signal import StateSpace


class Cartpole:
    def __init__(self, dt) -> None:
        m = 0.25
        M = 0.029 * 2
        b = 0.1
        I = 0.006
        l = 0.3
        g = 9.8

        z = I * (M + m) + M * m * l**2

        A = np.array(
            [
                [0, 1, 0, 0],
                [0, -(I + m * l**2) * b / z, -(m**2) * l**2 * g / z, 0],
                [0, 0, 0, 1],
                [0, m * l * b / z, m * l * g * (M + m) / z, 0],
            ]
        )

        B = np.array(
            [
                [0],
                [(I + m * l**2) / z],
                [0],
                [-m * l / z],
            ]
        )

        C = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        D = np.zeros((4, 1))

        self.sys = StateSpace(A, B, C, D).to_discrete(dt)

    def transition_fn(self, x, u):
        return self.sys.A @ x + self.sys.B * u


if __name__ == "__main__":
    dt = 0.01
    cartpole = Cartpole(dt)

    # t = np.arange(0, 10, dt)
    # f = cartpole.transition_fn
    # state = np.zeros((len(t), 4))
    # x = np.array([0, 0, 0, 0]).reshape(-1, 1)
    # u = np.ones(len(t))
    # for k in range(len(t)):
    #     x = f(x, u[k])
    #     state[k, :] = x.flatten()

    N = 50
    num_states = 4
    num_inputs = 1

    opti = ca.Opti()
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N)
    x0 = opti.parameter(num_states)
    r = opti.parameter(num_states)

    J = 0  # objective function

    # vehicle dynamics
    f = cartpole.transition_fn

    Q = np.diag([1000.0, 0.0, 100.0, 0.0])  # state weighing matrix
    R = np.diag([1.0])  # controls weighing matrix

    T = 20

    for k in range(N):
        J += (x[:, k] - r).T @ Q @ (x[:, k] - r) + u[:, k].T @ R @ u[:, k]
        x_next = f(x[:, k], u[:, k])
        opti.subject_to(x[:, k + 1] == x_next)

    opti.minimize(J)
    opti.subject_to(x[:, 0] == x0)
    opti.subject_to(u[0, :] >= -5)
    opti.subject_to(u[0, :] <= 5)

    opti.set_value(x0, ca.vertcat(0, 0, 0.34, 0))
    opti.set_value(r, ca.vertcat(0.0, 0, 0.0, 0))

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

    while np.linalg.norm(x_current[2] - opti.value(r)[2]) > 1e-6 and k < T / dt:
        inner_start = time.time()

        error = np.linalg.norm(x_current[2] - opti.value(r)[2])

        error_history[:, k] = np.linalg.norm(
            x_current.reshape(-1, 1) - opti.value(r).reshape(-1, 1),
            axis=1,
        )

        print(f"Step = {k} Timestep = {k * dt:.2f} Error = {error:.4f}")

        sol = opti.solve()
        u_opt = sol.value(u)
        x_history[:, k] = x_current.flatten()
        u_history[:, k] = u_opt[0]

        x_current = f(x_current.reshape((-1, 1)), u_opt[0])

        opti.set_value(x0, x_current)

        k += 1

        print(f"MPC time step: {(time.time() - inner_start):.4f}")

    print(f"Total MPC time: {(time.time() - start):.2f}")

    t = np.arange(0, T, dt)

    x_des, v_des, theta_des, dtheta_des = opti.value(r)
    thetat_des = np.full(len(t), theta_des)
    xt_des = np.full(len(t), x_des)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, thetat_des, "--", label=r"$\theta_d$")
    plt.plot(t, np.mod(x_history[2, :], 2 * np.pi), label=r"$\theta$")
    plt.xlabel("Time [$s$]")
    plt.ylabel(r"$\theta$ [$rad$]")
    plt.title("Pole Tilt Angle")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, xt_des, "--", label="$x_d$")
    plt.plot(t, x_history[0, :], label="$x$")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Position [$m$]")
    plt.title("Cart Position")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
