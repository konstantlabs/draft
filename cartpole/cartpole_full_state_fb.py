# %% Imports
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import eigvals, matrix_rank as rank

from control import ctrb, place

from lib.models import LinearCartpole
T = 20
dt = 0.025
cartpole = LinearCartpole(dt)

# %% Stability
eig = eigvals(cartpole.A)
print(f"eig={eig}")

# %% Controllability
print(f"rank={rank(ctrb(cartpole.A, cartpole.B))}")

# %% Compute gain K
K = place(cartpole.A, cartpole.B, np.array([0.7, 0.6, 0.5, 0.4]))
print(K)

# %% Check new poles
print(eigvals((cartpole.A - cartpole.B * K)))

# %% Run Control
f = cartpole.transition_fn

t = np.arange(0, 20, dt)

x_history = np.empty((4, len(t)))
u_history = np.empty((1, len(t)))

x = np.array([0, 0, 0.017, 0]).reshape((-1, 1))

set_point = 0.0

for k in range(len(t)):
    error = x[2] - set_point

    u = -K * error

    x = f(x, u)

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
