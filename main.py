import time
import casadi as ca
import numpy as np


if __name__ == '__main__':
    dt = 0.2  # timestep in seconds
    N = 10  # horizon
    rob_diam = 0.3

    v_max = 0.6
    v_min = -v_max
    omega_max = np.pi / 4
    omega_min = -omega_max

    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    states = ca.vertcat(x, y, theta)
    n_states = states.numel()

    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')
    controls = ca.vertcat(v, omega)
    n_controls = controls.numel()
    rhs = ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        omega,
    )  # system right-hand side

    f = ca.Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)

    U = ca.SX.sym('U', n_controls, N)

    P = ca.SX.sym('P', 2 * n_states)
    X = ca.SX.sym('X', n_states, (N + 1))

    Q = np.zeros((3, 3))
    Q[0, 0] = 1
    Q[1, 1] = 5
    Q[2, 2] = 0.1  # weighing matrices (states)
    R = np.zeros((2, 2))
    R[0, 0] = 0.5
    R[1, 1] = 0.05  # weighing matrices (controls)

    st = X[:, 0]

    obj = 0
    g = st - P[:3]
    for k in range(N):
        st = X[:, k]
        u = U[:, k]
        obj += ca.mtimes((st - P[3:6]).T, ca.mtimes(Q, (st - P[3:6]))) + ca.mtimes(u.T, ca.mtimes(R, u))
        st += f(st, u) * dt
        g = ca.vertcat(g, X[:, k + 1] - st)

    # make the decision variable one column vector
    OPT_variables = ca.vertcat(
        ca.reshape(X, n_states * (N + 1), 1),
        ca.reshape(U, n_controls * N, 1),
    )

    nlp_prob = {
        'f': obj,
        'x': OPT_variables,
        'g': g,
        'p': P,
    }

    opts = {
        'print_time': 0,
        'ipopt': {
            'max_iter': 2000,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6
        }
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    args = {
        'lbg': np.zeros((3 * (N + 1), 1)),
        'ubg': np.zeros((3 * (N + 1), 1)),
        'lbx': np.zeros((3 * (N + 1) + 2 * N, 1)),
        'ubx': np.zeros((3 * (N + 1) + 2 * N, 1))
    }

    args['lbx'][:3*(N+1):3, 0] = -2  # state x lower bound
    args['ubx'][:3*(N+1):3, 0] = 2  # state x upper bound
    args['lbx'][1:3*(N+1):3, 0] = -2  # state y lower bound
    args['ubx'][1:3*(N+1):3, 0] = 2  # state y upper bound
    args['lbx'][2:3*(N+1):3, 0] = -np.inf  # state theta lower bound
    args['ubx'][2:3*(N+1):3, 0] = np.inf  # state theta upper bound

    args['lbx'][3 * (N + 1):(3 * (N + 1) + 2 * N):2, 0] = v_min  # v lower bound
    args['ubx'][3 * (N + 1):(3 * (N + 1) + 2 * N):2, 0] = v_max  # v upper bound
    args['lbx'][3 * (N + 1) + 1:(3 * (N + 1) + 2 * N):2, 0] = omega_min  # omega lower bound
    args['ubx'][3 * (N + 1) + 1:(3 * (N + 1) + 2 * N):2, 0] = omega_max  # omega upper bound

    t = np.arange(0, 20, dt)

    # Initial conditions
    u0 = np.zeros((2, N))
    x0 = np.zeros((3, 1))

    # Reference
    r = np.array([[1.5], [1.5], [0.0]])

    # History bufffers (state, horizon, and control trajectories)
    x_trajectory = np.zeros((3, len(t)))
    x_trajectory[:, 0] = x0.T.flatten()
    u_trajectory = []
    horizon_trajectory = []

    X = np.tile(x0, (1, N + 1))

    xk = x0
    u = u0
    k = 0

    start_time = time.time()
    while np.linalg.norm(xk - r) > 1e-2 and k < len(t) - 1:
        print(f"x={xk.flatten()}")
        print(f"r={r.flatten()}")

        args['p'] = np.vstack((xk, r))
        args['x0'] = ca.vertcat(
            X.reshape((3 * (N + 1), 1)),
            u.reshape((2 * N), 1),
        )

        print(f"x0={args['x0'][:3]}")

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p'],
        )

        # get controls only from the solution
        u = (
            sol['x'][3 * (N + 1):]
            .full()
            .reshape((2, N))
        )

        xk = (xk + f(xk, u[:, 0]) * dt).full()

        u = np.hstack((u[:, 1:], u[:, -1:]))  # shift controls

        X = (
            sol['x'][:3 * (N + 1)]
            .full()
            .reshape((3, N + 1))
        )  # predicted trajectory

        X = np.hstack((X[:, 1:], X[:, -1:]))  # shift predicted trajectory
        k += 1

        print(f"dt={t[k]}, error={np.linalg.norm(xk - r)}")

    end_time = time.time()
