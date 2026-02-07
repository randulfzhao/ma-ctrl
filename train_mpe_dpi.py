import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


@dataclass
class DPIParams:
    # Pendulum parameters
    m: float = 1.0
    l: float = 1.0
    g: float = 9.8
    mu: float = 0.01

    # RBFN parameters
    n1: int = 11
    n2: int = 11
    x1_bound: float = math.pi
    x2_bound: float = 6.0
    sig1: float = 1.0
    sig2: float = 0.5

    # Control / learning parameters
    u_max: float = 5.0
    ts: float = 0.01
    gamma: float = 0.1
    control_gain: float = 0.01

    # Modes
    reward_type: str = "Con"  # Con, Bin, Opt
    control_type: str = "Normal"  # Normal, B-bang


def build_centers(p: DPIParams) -> Tuple[np.ndarray, np.ndarray]:
    centers1 = np.linspace(-p.x1_bound, p.x1_bound, p.n1)
    centers2 = np.linspace(-p.x2_bound, p.x2_bound, p.n2)
    return centers1, centers2


def nn_hidden_fire(x: np.ndarray, centers1: np.ndarray, centers2: np.ndarray, p: DPIParams) -> np.ndarray:
    e1 = x[0] - centers1
    e2 = x[1] - centers2
    z = p.sig1 * (e1 ** 2)
    z2 = p.sig2 * (e2 ** 2)
    Z = z2[:, None] + z[None, :]
    return np.exp(-Z).reshape(-1)


def nn_hidden_gradient_fire(
    x: np.ndarray, centers1: np.ndarray, centers2: np.ndarray, p: DPIParams
) -> np.ndarray:
    e1 = x[0] - centers1
    e2 = x[1] - centers2
    E1 = e1[None, :]
    E2 = e2[:, None]
    exp_term = np.exp(-p.sig1 * (E1 ** 2) - p.sig2 * (E2 ** 2))
    g1 = -2.0 * exp_term * (E1 * p.sig1)
    g2 = -2.0 * exp_term * (E2 * p.sig2)
    return np.stack([g1, g2], axis=2).reshape(-1, 2)


def nn_fire(theta: np.ndarray, x: np.ndarray, centers1: np.ndarray, centers2: np.ndarray, p: DPIParams) -> float:
    return float(theta @ nn_hidden_fire(x, centers1, centers2, p))


def nn_gradient_fire(
    theta: np.ndarray, x: np.ndarray, centers1: np.ndarray, centers2: np.ndarray, p: DPIParams
) -> np.ndarray:
    return theta @ nn_hidden_gradient_fire(x, centers1, centers2, p)


def input_cost(u: float, p: DPIParams) -> float:
    u_nor = u / p.u_max
    temp1 = 1.0 + u_nor
    temp2 = 1.0 - u_nor
    return p.control_gain * (p.u_max ** 2) * math.log((temp1 ** temp1) * (temp2 ** temp2)) / 2.0


def reward_fn(x: np.ndarray, u: float, p: DPIParams) -> float:
    if p.reward_type == "Opt":
        alpha = 0.01
        r = -(x[0] ** 2) - alpha * (x[1] ** 2)
    elif p.reward_type == "Con":
        r = math.cos(x[0])
    elif p.reward_type == "Bin":
        r = 1.0 if (abs(x[0]) <= math.pi / 6.0 and abs(x[1]) <= 0.5) else 0.0
    else:
        raise ValueError("reward_type should be Con, Bin, or Opt")

    if p.control_type == "Normal":
        r = r - input_cost(u, p)
    return float(r)


def action_generator(
    z: np.ndarray, theta_v: np.ndarray, centers1: np.ndarray, centers2: np.ndarray, p: DPIParams
) -> float:
    x = z[:2]
    grad_v = nn_gradient_fire(theta_v, x, centers1, centers2, p)
    u = -math.cos(x[0]) * grad_v[1]

    if p.control_type == "Normal":
        u = p.ts * u
        denom = p.control_gain * p.m * (p.l ** 2) * p.u_max
        u = p.u_max * math.tanh(u / denom)
    else:
        if u > 0:
            u = p.u_max
        elif u == 0:
            u = 0.0
        else:
            u = -p.u_max
    return float(u)


def closed_loop_sys(z: np.ndarray, u: float, p: DPIParams, opt: Optional[str] = None) -> np.ndarray:
    x1 = z[0]
    x2 = z[1]

    z_dot = np.zeros_like(z, dtype=np.float64)
    z_dot[0] = x2
    z_dot[1] = -p.mu * x2 + p.m * p.g * p.l * math.sin(x1) - math.cos(x1) * u
    z_dot[1] = z_dot[1] / p.m / (p.l ** 2)
    z_dot[2] = x2

    if opt == "reward":
        z_dot[3] = reward_fn(z[:2], u, p) + math.log(p.gamma) * z[3]

    return z_dot


def rk4_closed(
    X: np.ndarray, theta_v: np.ndarray, centers1: np.ndarray, centers2: np.ndarray, p: DPIParams, opt: Optional[str]
) -> Tuple[np.ndarray, float]:
    K1 = p.ts * closed_loop_sys(X, action_generator(X, theta_v, centers1, centers2, p), p, opt)
    X1 = X + 0.5 * K1

    K2 = p.ts * closed_loop_sys(X1, action_generator(X1, theta_v, centers1, centers2, p), p, opt)
    X2 = X + 0.5 * K2

    K3 = p.ts * closed_loop_sys(X2, action_generator(X2, theta_v, centers1, centers2, p), p, opt)
    X3 = X + K3

    K4 = p.ts * closed_loop_sys(X3, action_generator(X3, theta_v, centers1, centers2, p), p, opt)

    X_n = X + np.column_stack([K1, K2, K3, K4]) @ (np.array([1.0, 2.0, 2.0, 1.0]) / 6.0)
    U_n = action_generator(X_n, theta_v, centers1, centers2, p)
    return X_n, U_n


def normalize_x1(x: np.ndarray) -> np.ndarray:
    if x[0] > math.pi:
        x[0] = x[0] - 2.0 * math.pi
    elif x[0] < -math.pi:
        x[0] = x[0] + 2.0 * math.pi
    return x


def num2ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def plot_value_surface(theta_v: np.ndarray, centers1: np.ndarray, centers2: np.ndarray, p: DPIParams, title: str):
    if not HAVE_MPL:
        print("matplotlib not available; skipping plots.")
        return
    grid_n1 = 50
    grid_n2 = 50
    x1 = np.linspace(-p.x1_bound, p.x1_bound, grid_n1)
    x2 = np.linspace(-p.x2_bound, p.x2_bound, grid_n2)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(grid_n1):
        for j in range(grid_n2):
            Z[j, i] = nn_fire(theta_v, np.array([X1[j, i], X2[j, i]]), centers1, centers2, p)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X1, X2, Z, edgecolor="none", linewidth=0.0)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Amplitude")


def main():
    parser = argparse.ArgumentParser(description="DPI algorithm ported from alg/Main.m (MATLAB).")
    parser.add_argument("--reward-type", type=str, default="Con", choices=["Con", "Bin", "Opt"])
    parser.add_argument("--control-type", type=str, default="Normal", choices=["Normal", "B-bang"])
    parser.add_argument("--grid-n1", type=int, default=20)
    parser.add_argument("--grid-n2", type=int, default=21)
    parser.add_argument("--npi", type=int, default=50)
    parser.add_argument("--rbf-n1", type=int, default=11)
    parser.add_argument("--rbf-n2", type=int, default=11)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--ts", type=float, default=0.01)
    parser.add_argument("--u-max", type=float, default=5.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    p = DPIParams(
        reward_type=args.reward_type,
        control_type=args.control_type,
        n1=args.rbf_n1,
        n2=args.rbf_n2,
        gamma=args.gamma,
        ts=args.ts,
        u_max=args.u_max,
    )

    centers1, centers2 = build_centers(p)

    grid_n1 = args.grid_n1
    grid_n2 = args.grid_n2
    x1_grids = np.linspace(-p.x1_bound, p.x1_bound, grid_n1)
    x2_grids = np.linspace(-p.x2_bound, p.x2_bound, grid_n2)

    theta_v = np.zeros(p.n1 * p.n2, dtype=np.float64)
    theta_v_trj = np.zeros((p.n1 * p.n2, args.npi + 1), dtype=np.float64)

    log_gamma_ts = math.log(p.gamma ** p.ts)

    for iPI in range(1, args.npi + 1):
        C = np.zeros((grid_n1 * grid_n2, p.n1 * p.n2), dtype=np.float64)
        b = np.zeros((grid_n1 * grid_n2,), dtype=np.float64)

        counter = -1
        for i in range(grid_n1):
            for j in range(grid_n2):
                counter += 1
                x0 = np.array([x1_grids[i], x2_grids[j], x1_grids[i]], dtype=np.float64)
                u = action_generator(x0, theta_v, centers1, centers2, p)

                reward = reward_fn(x0[:2], u, p)
                x_dot = closed_loop_sys(x0, u, p)

                phi_grad = nn_hidden_gradient_fire(x0[:2], centers1, centers2, p)
                phi = nn_hidden_fire(x0[:2], centers1, centers2, p)

                C[counter, :] = -(p.ts * (phi_grad @ x_dot[:2]) + log_gamma_ts * phi)
                b[counter] = reward

                if args.verbose:
                    idx = (i * grid_n2) + j + 1
                    print(f"{num2ordinal(idx)} (trj) evaluation, {num2ordinal(iPI)} iteration")

        theta_v = np.linalg.lstsq(C, b, rcond=None)[0]
        theta_v_trj[:, iPI] = theta_v
        print(f"{num2ordinal(iPI)} iteration done...")

    if args.plot:
        if HAVE_MPL:
            plt.figure()
            for k in range(theta_v_trj.shape[0]):
                plt.plot(np.arange(0, args.npi + 1), theta_v_trj[k, :])
            plt.grid(True)
            plt.title("Evolution of the weights theta_i of the RBFVN")
            plt.xlabel("Iteration")
            plt.ylabel("Amplitude")

            plot_value_surface(theta_v, centers1, centers2, p, f"The value function v_i(x) (i = {args.npi})")
        else:
            print("matplotlib not available; skipping plots.")

    if not args.skip_test:
        print("Learning completed.....")
        print("Test the policies in the learning process....")

        t_f = 10.0
        t_f_iter = int(t_f / p.ts)
        t_line = np.linspace(0.0, t_f, t_f_iter + 1)

        X = np.zeros((4, t_f_iter + 1), dtype=np.float64)
        U = np.zeros((t_f_iter + 1,), dtype=np.float64)

        eps = 0.0
        init_bias = 1.0

        i_index = [1, 2, 3, 4, args.npi]
        i_index = sorted({i for i in i_index if 0 <= i <= args.npi})

        if args.plot and HAVE_MPL:
            plt.figure()

        for iPI in i_index:
            theta_v = theta_v_trj[:, iPI]
            X[:, 0] = init_bias * np.array([math.pi, 0.0, math.pi, 0.0])

            for k in range(t_f_iter):
                U[k] = action_generator(X[:, k], theta_v, centers1, centers2, p)
                X[:, k + 1], _ = rk4_closed(X[:, k], theta_v, centers1, centers2, p, "reward")
                if abs(X[0, k + 1]) > (1.0 + eps) * math.pi:
                    X[:, k + 1] = normalize_x1(X[:, k + 1])

            U[t_f_iter] = action_generator(X[:, t_f_iter], theta_v, centers1, centers2, p)

            if args.plot and HAVE_MPL:
                plt.plot(t_line, X[2, :], linewidth=1.3, label=f"i = {iPI}")

            print(f"the cumulative reward for the {num2ordinal(iPI)} policy is {X[3, t_f_iter]}")

        if args.plot and HAVE_MPL:
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.title("Pendulum state trajectory X1t under pi_i")
            plt.legend()

            plt.figure()
            plt.plot(t_line, U, "-b", linewidth=1.3)
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.title(f"Action trajectory U_t under pi_i (i = {args.npi})")

        print("Simulation completed...")

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", f"DPI_{p.reward_type}_{p.control_type}.npz")
    np.savez(
        out_path,
        theta_v_trj=theta_v_trj,
        rbf_centers1=centers1,
        rbf_centers2=centers2,
        params=np.array(
            [
                p.m,
                p.l,
                p.g,
                p.mu,
                p.n1,
                p.n2,
                p.x1_bound,
                p.x2_bound,
                p.sig1,
                p.sig2,
                p.u_max,
                p.ts,
                p.gamma,
                p.control_gain,
            ],
            dtype=np.float64,
        ),
    )
    print(f"Saved data to {out_path}")

    if args.plot and HAVE_MPL:
        plt.show()


if __name__ == "__main__":
    main()
