import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import StateSpace


class Cartpole:
    def __init__(self, dt) -> None:
        m = 0.2
        M = 0.5
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
        return self.sys.A.dot(x) + self.sys.B * u


if __name__ == "__main__":
    dt = 0.01
    cartpole = Cartpole(dt)

    t = np.arange(0, 10, dt)
    f = cartpole.transition_fn
    state = np.zeros((len(t), 4))
    x = np.array([0, 0, 0, 0]).reshape(-1, 1)
    u = np.ones(len(t))
    for k in range(len(t)):
        x = f(x, u[k])
        state[k, :] = x.flatten()

    plt.figure()
    plt.plot(t, state[:, 0], label="Cart position")
    plt.ylim([0, 50])
    plt.xlim([0, 5])
    plt.legend()
    plt.grid()
    plt.show()
