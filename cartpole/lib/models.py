from abc import ABC, abstractmethod
from typing import overload
import casadi as ca
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
    def __init__(self, dt: float | None = None) -> None:
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

        self.sys: StateSpace = StateSpace(A, B, C, D)

        self.A: np.ndarray = A
        self.B: np.ndarray = B
        self.C: np.ndarray = C
        self.D: np.ndarray = D

        if dt:
            dsys = self.sys.to_discrete(dt)
            self.A = dsys.A
            self.B = dsys.B
            self.C = dsys.C
            self.D = dsys.D

            self.sys = dsys

    def transition_fn(self, x, u):
        return self.sys.A @ x + self.sys.B * u
