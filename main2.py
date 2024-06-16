import time
import casadi as ca
import numpy as np


if __name__ == '__main__':
    N = 10

    opti = ca.Opti()

    x = opti.variable(3 * (N + 1), 1)
    u = opti.variable(2 * N, 1)
    p = opti.parameter(6, 1)

    print(p.shape)
    print(ca.vertcat(x, u).shape)
