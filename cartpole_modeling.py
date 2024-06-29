import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


if __name__ == "__main__":
    m = 10
    M = 5
    b = 0.1
    I = 10
    l = 10
    g = 9.81

    def nonlinear_dynamics(x, t, u):
        x, dx, theta, dtheta = x

        beta = I * (M + m) + M * m * l**2 + np.sin(theta) ** 2 * m**2 * l**2

        return [
            dx,
            (I + m * l**2) * (m * l * dtheta**2 * np.sin(theta) - b * dx)
            - m**2 * l**2 * g * np.cos(theta) * np.sin(theta)
            + (I + m * l**2) * u[0],
            dtheta,
            m * l * np.cos(theta) * (b * dx - m * l * dtheta**2 * np.sin(theta))
            + (M + m) * m * l * g
            - m * l * np.cos(dtheta),
        ] / beta

    def linear_dynamics(x, t, u):
        z = I * (M + m) + M * m * l**2

        A = np.array(
            [
                [0, 1, 0, 0],
                [0, -(I + m * l**2) * b, -(m**2) * l**2 * g, 0],
                [0, 0, 0, 1],
                [0, m * l * b, (m + M) * m * l * g, 0],
            ]
        )

        B = np.array([[0], [I + m * l**2], [0], [-m * l]])

        y = (A.dot(x.reshape(-1, 1)) + B * u) / z

        return y.flatten()

    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    u = np.array([500.0])

    t = np.arange(0, 50, 0.2)

    sol1 = odeint(nonlinear_dynamics, x0, t, args=(u,))
    sol2 = odeint(linear_dynamics, x0, t, args=(u,))

    x1 = sol1[:, 0]
    dx1 = sol2[:, 1]
    theta1 = sol1[:, 2]
    dtheta1 = sol1[:, 3]

    x2 = sol2[:, 0]
    dx2 = sol2[:, 1]
    theta2 = sol2[:, 2]
    dtheta2 = sol2[:, 3]

    plt.subplot(2, 2, 1)
    plt.plot(t, x1, label="nonlinear")
    plt.plot(t, x2, label="linear")
    plt.xlabel(r"$t$ $[s]$")
    plt.ylabel(r"$x$ $[m]$")
    plt.title("Cart position")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t, np.mod(theta1, 2 * np.pi), label="nonlinear")
    plt.plot(t, np.mod(theta2, 2 * np.pi), label="linear")
    plt.xlabel(r"$t$ $[s]$")
    plt.ylabel(r"$\theta$ $[rad]$")
    plt.title("Pole angle")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(t, dx1, label="nonlinear")
    plt.plot(t, dx2, label="linear")
    plt.xlabel(r"$t$ $[s]$")
    plt.ylabel(r"$\dot{x}$ $[\frac{m}{s}]$")
    plt.title("Cart linear velocity")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(t, np.mod(dtheta1, 2 * np.pi), label="nonlinear")
    plt.plot(t, np.mod(dtheta2, 2 * np.pi), label="linear")
    plt.xlabel(r"$t$ $[s]$")
    plt.ylabel(r"$\dot{\theta}$ $[\frac{rad}{s}]$")
    plt.title("Pole angular velocity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
