import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


if __name__ == "__main__":
    m = 0.2
    M = 0.5
    b = 0.1
    I = 0.006
    l = 0.3
    g = 9.8

    def impulse(t):
        return 1.0 if t == 0 else 0.0

    def step(t):
        return 1.0

    def nonlinear_dynamics(x, u):
        x, dx, theta, dtheta = x

        beta = I * (M + m) + M * m * l**2 + np.sin(theta) ** 2 * m**2 * l**2

        return np.array(
            [
                dx,
                (
                    (I + m * l**2) * (m * l * dtheta**2 * np.sin(theta) - b * dx)
                    - m**2 * l**2 * g * np.cos(theta) * np.sin(theta)
                    + (I + m * l**2) * u
                )
                / beta,
                dtheta,
                (
                    m * l * np.cos(theta) * (b * dx - m * l * dtheta**2 * np.sin(theta))
                    + (M + m) * m * l * g
                    - m * l * np.cos(dtheta)
                )
                / beta,
            ]
        )

    f = nonlinear_dynamics

    def euler(x, u, dt):
        return x + f(x, u) * dt

    def runge_kutta(x, u, h):
        k1 = f(x, u)
        k2 = f(x + h / 2 * k1, u)
        k3 = f(x + h / 2 * k2, u)
        k4 = f(x + h * k3, u)

        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    dt = 0.01
    t = np.arange(0, 50, dt)

    def solve_discrete_dyn(x0, u, t, dt, method="runge_kutta"):
        states = [x0]
        x = x0
        for i, tk in enumerate(t[:-1]):
            x = (
                runge_kutta(x, u(tk), dt)
                if method == "runge_kutta"
                else euler(x, u(tk), dt)
            )
            states.append(x)
        return np.array(states)

    def solve_dynamics(x, t, u):
        return nonlinear_dynamics(x, u(t))

    sol1 = odeint(solve_dynamics, x0, t, args=(step,))
    sol2 = solve_discrete_dyn(x0, step, t, dt)
    sol3 = solve_discrete_dyn(x0, step, t, dt, method="euler")

    x1 = sol1[:, 0]
    dx1 = sol1[:, 1]
    theta1 = sol1[:, 2]
    dtheta1 = sol1[:, 3]

    x2 = sol2[:, 0]
    dx2 = sol2[:, 1]
    theta2 = sol2[:, 2]
    dtheta2 = sol2[:, 3]

    x3 = sol3[:, 0]
    dx3 = sol3[:, 1]
    theta3 = sol3[:, 2]
    dtheta3 = sol3[:, 3]

    plt.figure()
    plt.plot(t, x1, label="nonlinear continuous")
    plt.plot(t, x2, label="nonlinear discrete (rk)")
    plt.plot(t, x3, label="nonlinear discrete (euler)")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(t, x1, label="nonlinear continuous")
    plt.plot(t, x2, label="nonlinear discrete (rk)")
    plt.plot(t, x3, label="nonlinear discrete (euler)")
    plt.xlabel(r"$t$ $[s]$")
    plt.ylabel(r"$x$ $[m]$")
    plt.title("Cart position")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t, np.mod(theta1, 2 * np.pi), label="nonlinear")
    # plt.plot(t, np.mod(theta2, 2 * np.pi), label="linear")
    plt.xlabel(r"$t$ $[s]$")
    plt.ylabel(r"$\theta$ $[rad]$")
    plt.title("Pole angle")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(t, dx1, label="nonlinear")
    # plt.plot(t, dx2, label="linear")
    plt.xlabel(r"$t$ $[s]$")
    plt.ylabel(r"$\dot{x}$ $[\frac{m}{s}]$")
    plt.title("Cart linear velocity")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(t, np.mod(dtheta1, 2 * np.pi), label="nonlinear")
    # plt.plot(t, np.mod(dtheta2, 2 * np.pi), label="linear")
    plt.xlabel(r"$t$ $[s]$")
    plt.ylabel(r"$\dot{\theta}$ $[\frac{rad}{s}]$")
    plt.title("Pole angular velocity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
