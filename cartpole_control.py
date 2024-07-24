from abc import ABC, abstractmethod
import time
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from scipy.signal import StateSpace


class Cartpole(ABC):
    def __init__(self, dt) -> None:
        self.m = 0.25
        self.M = 0.029 * 2
        self.b = 0.1
        self.I = 0.006
        self.l = 0.3
        self.g = 9.8

    @abstractmethod
    def transition_fn(self, x, u):
        pass


class LinearCartpole(Cartpole):
    def __init__(self, dt) -> None:
        super().__init__(dt)

        z = self.I * (self.M + self.m) + self.M * self.m * self.l**2

        A = np.array(
            [
                [0, 1, 0, 0],
                [
                    0,
                    -(self.I + self.m * self.l**2) * self.b / z,
                    -(self.m**2) * self.l**2 * self.g / z,
                    0,
                ],
                [0, 0, 0, 1],
                [
                    0,
                    self.m * self.l * self.b / z,
                    self.m * self.l * self.g * (self.m + self.m) / z,
                    0,
                ],
            ]
        )

        B = np.array(
            [
                [0],
                [(self.I + self.m * self.l**2) / z],
                [0],
                [-self.m * self.l / z],
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
        return ca.DM(self.sys.A) @ x + ca.DM(self.sys.B) * u


class NonlinearCartpole(Cartpole):
    def __init__(self, dt):
        super().__init__(dt)

    def transition_fn(self, x, u):
        dx = x[1]
        theta = x[2]
        dtheta = x[3]

        beta = (
            self.I * (self.M + self.m)
            + self.M * self.m * self.l**2
            + ca.sin(theta) ** 2 * self.m**2 * self.l**2
        )

        return ca.vertcat(
            dx,
            (
                (self.I + self.m * self.l**2)
                * (self.m * self.l * dtheta**2 * ca.sin(theta) - self.b * dx)
                - self.m**2 * self.l**2 * self.g * ca.cos(theta) * ca.sin(theta)
                + (self.I + self.m * self.l**2) * u
            )
            / beta,
            dtheta,
            (
                self.m
                * self.l
                * ca.cos(theta)
                * (self.b * dx - self.m * self.l * dtheta**2 * ca.sin(theta))
                + (self.M + self.m) * self.m * self.l * self.g * ca.sin(theta)
                - self.m * self.l * ca.cos(dtheta) * u
            )
            / beta,
        )


def runge_kutta(f, x, u, h):
    k1 = f(x, u)
    k2 = f(x + h / 2 * k1, u)
    k3 = f(x + h / 2 * k2, u)
    k4 = f(x + h * k3, u)

    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == "__main__":
    dt = 0.05
    nl_cartpole = NonlinearCartpole(dt)
    linear_cartpole = LinearCartpole(dt)

    N = 20
    num_states = 4
    num_inputs = 1

    opti = ca.Opti()
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N)
    x0 = opti.parameter(num_states)
    r = opti.parameter(num_states)

    J = 0  # objective function

    # vehicle dynamics
    f = nl_cartpole.transition_fn
    # f = linear_cartpole.transition_fn

    Q = np.diag([1000.0, 0.0, 2000.0, 0.0])  # state weighing matrix
    R = np.diag([0.01])  # controls weighing matrix

    T = 20

    for k in range(N):
        J += (x[:, k] - r).T @ Q @ (x[:, k] - r) + u[:, k].T @ R @ u[:, k]
        x_next = runge_kutta(f, x[:, k], u[:, k], dt)
        # x_next = x[:, k] + f(x[:, k], u[:, k]) * dt
        # x_next = f(x[:, k], u[:, k])
        opti.subject_to(x[:, k + 1] == x_next)

    opti.minimize(J)
    opti.subject_to(x[:, 0] == x0)
    opti.subject_to(u[0, :] >= -5)
    opti.subject_to(u[0, :] <= 5)

    opti.set_value(x0, ca.vertcat(0.0, 0, 0.01745329, 0))
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

    while k < T / dt:
        inner_start = time.time()

        error = np.linalg.norm(x_current[2] - opti.value(r)[2])

        error_history[:, k] = np.linalg.norm(
            x_current.reshape((-1, 1)) - opti.value(r).reshape((-1, 1)),
            axis=1,
        )

        print(f"Step = {k} Timestep = {k * dt:.2f} Error = {error:.4f}")

        sol = opti.solve()
        u_opt = sol.value(u)
        x_history[:, k] = x_current
        u_history[:, k] = u_opt[0]

        x_current = runge_kutta(f, x_current, u_opt[0], dt)
        # x_current += f(x_current.reshape((-1, 1)), u_opt[0]) * dt
        # x_current = f(x_current.reshape((-1, 1)), u_opt[0])
        x_current = x_current.full().flatten()

        opti.set_value(x0, x_current)

        k += 1

        print(f"MPC time step: {(time.time() - inner_start):.4f}")

    print(f"Total MPC time: {(time.time() - start):.2f}")

    t = np.arange(0, T, dt)

    x_des, v_des, theta_des, dtheta_des = opti.value(r)
    thetat_des = np.full(len(t), theta_des)
    xt_des = np.full(len(t), x_des)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, thetat_des, "--", label=r"$\theta_d$")
    plt.plot(t, np.mod(x_history[2, :], 2 * np.pi), label=r"$\theta$")
    plt.xlabel("Time [$s$]")
    plt.ylabel(r"$\theta$ [$rad$]")
    plt.title("Pole Tilt Angle")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, xt_des, "--", label="$x_d$")
    plt.plot(t, x_history[0, :], label="$x$")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Position [$m$]")
    plt.title("Cart Position")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, u_history[0, :], label="$u$")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Control [$N$]")
    plt.title("Control Input")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
