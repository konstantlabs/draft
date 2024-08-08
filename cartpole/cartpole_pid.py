# %% Imports
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

from lib.models import LinearCartpole

# %% PID
T = 20
dt = 0.025
cartpole = LinearCartpole(dt)

f = cartpole.transition_fn

t = np.arange(0, 20, dt)

x_history = np.empty((4, len(t)))
u_history = np.empty((1, len(t)))

set_point, integral, error_prev = 0, 0, 0
kp, ki, kd = 7.0, 3.2, 0.5

x = np.array([0, 0, 0.017, 0]).reshape((-1, 1))

for k in range(len(t)):
    error = x[2] - set_point
    integral += error

    u = kp * error + ki * integral * dt + kd * (error - error_prev) / dt

    x = f(x, u)

    error_prev = error

    x_history[:, k] = x.flatten()
    u_history[:, k] = u

# %% Plot results
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
