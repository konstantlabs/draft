from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import StateSpace


class Cartpole(ABC):
    def __init__(self, dt) -> None:
        self.m = 0.25
        self.M = 0.029 * 2
        self.b = 0.1
        self.l = 0.3
        self.I = 1 / 3 * self.m * self.l**2
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
        return self.sys.A @ x + self.sys.B * u


if __name__ == "__main__":
    dt = 0.025
    cartpole = LinearCartpole(dt)

    f = cartpole.transition_fn

    T = 20

    t = np.arange(0, 20, dt)

    x0 = np.array([0, 0, 0.017, 0]).reshape((-1, 1))

    set_point = 0
    integral = 0
    error_prev = 0

    kp = 7.0
    ki = 3.2
    kd = 0.5

    x = x0

    x_history = np.empty((4, len(t)))
    u_history = np.empty((1, len(t)))

    for k in range(len(t)):
        error = x[2] - set_point
        integral += error

        u = kp * error + ki * integral * dt + kd * (error - error_prev) / dt

        x = f(x, u)

        error_prev = error

        x_history[:, k] = x.flatten()
        u_history[:, k] = u

    desired_pitch = np.full(len(t), set_point)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, desired_pitch, "--", label=r"$\theta_d$")
    plt.plot(t, x_history[2, :], label=r"$\theta$")
    plt.xlabel("Time [$s$]")
    plt.ylabel(r"$\theta$ [$rad$]")
    plt.title("Pole Tilt Angle")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, x_history[0, :], label="$x$")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Position [$m$]")
    plt.title("Cart Position")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, u_history[0, :], label="$u$")
    plt.xlabel("Time [$s$]")
    plt.ylabel(r"Control [$kg \cdot \frac{m}{s^2}$]")
    plt.title("Control Input")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
