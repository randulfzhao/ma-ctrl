#!/usr/bin/env python3
import argparse
import csv
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from runner_coop_ma_enode import FIXED_EPISODE_SECONDS, MAMUJOCO_ENV_SPECS, MaMuJoCoEnv
from utils.utils import numpy_to_torch


@dataclass
class EvalTarget:
    env_alias: str
    dt: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate uniform random policy on MaMuJoCo envs used by run_enode_mujoco.sh "
            "with the same reward aggregation strategy as runner_coop_ma_enode.py"
        )
    )
    p.add_argument("--run-script", type=str, default="run_enode_mujoco.sh")
    p.add_argument(
        "--envs",
        type=str,
        default="",
        help="Comma-separated env aliases. If empty, parse from --run-script.",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="111,112,113,114",
        help="Comma-separated evaluation seeds.",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-episodes", type=int, default=10, help="Equivalent to Ntest in train_loop.")
    p.add_argument(
        "--eval-horizon-sec",
        type=float,
        default=FIXED_EPISODE_SECONDS,
        help="Equivalent to Htest in train_loop.",
    )
    p.add_argument("--solver", type=str, default="rk4")
    p.add_argument("--ts-grid", type=str, default="fixed")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--consensus-weight", type=float, default=0.02)
    p.add_argument("--collect-parallel-workers", type=int, default=1)
    p.add_argument("--agent-obsk", type=int, default=1)
    p.add_argument("--csv", type=str, default="", help="Optional output CSV path.")
    p.add_argument(
        "--show-per-seed",
        action="store_true",
        help="Also print one row per environment seed.",
    )
    return p.parse_args()


def _parse_csv_list(raw: str, cast_fn):
    if raw is None:
        return []
    items = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            items.append(cast_fn(token))
    return items


def discover_targets_from_run_script(run_script_path: Path) -> List[EvalTarget]:
    if not run_script_path.is_file():
        raise FileNotFoundError(f"run script not found: {run_script_path}")

    run_text = run_script_path.read_text(encoding="utf-8")
    script_names = re.findall(r'^\s*"([^"]+\.sh)"', run_text, flags=re.MULTILINE)
    if not script_names:
        raise RuntimeError(f"no shell scripts found in {run_script_path}")

    targets = OrderedDict()
    for name in script_names:
        child_path = (run_script_path.parent / name).resolve()
        if not child_path.is_file():
            print(f"[warn] script not found, skip: {child_path}")
            continue

        text = child_path.read_text(encoding="utf-8")
        env_match = re.search(r"--env\s+([^\s\\]+)", text)
        dt_match = re.search(r"--dt\s+([0-9]*\.?[0-9]+)", text)
        if env_match is None:
            print(f"[warn] --env not found in {child_path}, skip.")
            continue
        env_alias = env_match.group(1).strip()
        dt = float(dt_match.group(1)) if dt_match is not None else 0.05
        if env_alias not in targets:
            targets[env_alias] = dt
        elif not np.isclose(float(targets[env_alias]), dt):
            print(
                f"[warn] inconsistent dt for env={env_alias}: "
                f"{targets[env_alias]} vs {dt}; using first one."
            )
    return [EvalTarget(env_alias=k, dt=float(v)) for k, v in targets.items()]


def resolve_targets(args: argparse.Namespace) -> List[EvalTarget]:
    if args.envs.strip():
        envs = _parse_csv_list(args.envs, str)
        if len(envs) == 0:
            raise ValueError("--envs is provided but empty after parsing.")
        return [EvalTarget(env_alias=e, dt=0.05) for e in envs]
    return discover_targets_from_run_script(Path(args.run_script))


def make_uniform_random_policy(env: MaMuJoCoEnv, seed: int):
    rng = np.random.default_rng(int(seed))
    lb = env.ac_lb.detach().cpu().numpy().astype(np.float32)
    ub = env.ac_ub.detach().cpu().numpy().astype(np.float32)
    m = int(env.m)

    def g(s, t):
        if isinstance(s, torch.Tensor):
            batch = int(s.shape[0])
            device = s.device
            dtype = s.dtype
        else:
            batch = 1
            device = env.device
            dtype = torch.float32
        acts = rng.uniform(lb, ub, size=(batch, m)).astype(np.float32)
        return torch.as_tensor(acts, device=device, dtype=dtype)

    return g


def eval_with_runner_strategy(
    env: MaMuJoCoEnv,
    policy_fn,
    num_episodes: int,
    eval_horizon_sec: float,
    device: torch.device,
) -> Dict[str, float]:
    # Same evaluation strategy as ctrl.utils.train_loop:
    # Htest=eval_horizon_sec, Ntest=num_episodes, Ttest=round(Htest/dt), Tup=0.
    Ttest = max(1, int(np.round(float(eval_horizon_sec) / float(env.dt))))
    Tup = 0
    s0 = torch.stack([numpy_to_torch(env.reset(), env.device) for _ in range(int(num_episodes))]).to(device)
    _, _, test_rewards, _ = env.integrate_system(T=Ttest, s0=s0, g=policy_fn)

    rewards_are_accumulated = bool(getattr(env, "rewards_are_accumulated", False))
    if rewards_are_accumulated:
        reward_tensor = test_rewards[..., -1]
        true_reward = reward_tensor.mean().item()
        min_reward = reward_tensor.min().item()
    else:
        reward_tensor = test_rewards[..., Tup:]
        true_reward = reward_tensor.mean().item()
        min_reward = reward_tensor.min().item()
    solved_ratio = (reward_tensor >= 0.8).float().mean().item()

    return {
        "true_reward": float(true_reward),
        "min_reward": float(min_reward),
        "solved_ratio": float(solved_ratio),
        "num_steps": int(Ttest),
    }


def build_ascii_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    cols = len(headers)
    widths = [len(str(headers[i])) for i in range(cols)]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells: Sequence[str]) -> str:
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "|-" + "-|-".join("-" * widths[i] for i in range(cols)) + "-|"
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def summarize(values: Iterable[float]) -> Tuple[float, float, float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), float(arr.max())


def maybe_write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "env",
        "dt",
        "n_seeds",
        "mean_true_reward",
        "std_true_reward",
        "min_true_reward",
        "max_true_reward",
        "mean_min_reward",
        "mean_solved_ratio",
        "seeds",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to: {output}")


def main() -> None:
    args = parse_args()
    seeds = _parse_csv_list(args.seeds, int)
    if len(seeds) == 0:
        raise ValueError("No valid seeds found. Please set --seeds.")

    targets = resolve_targets(args)
    if len(targets) == 0:
        raise RuntimeError("No environments to evaluate.")

    device = torch.device(args.device)
    print(
        f"Eval config: device={device}, horizon_sec={args.eval_horizon_sec}, "
        f"num_episodes={args.num_episodes}, seeds={seeds}"
    )

    per_seed_rows = []
    summary_rows = []
    for target in targets:
        env_alias = target.env_alias
        if env_alias not in MAMUJOCO_ENV_SPECS:
            print(f"[warn] skip unknown env alias: {env_alias}")
            continue

        scenario, agent_conf = MAMUJOCO_ENV_SPECS[env_alias]
        true_rewards = []
        min_rewards = []
        solved_ratios = []
        ok_seeds = []
        num_steps_seen = set()
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            try:
                env = MaMuJoCoEnv(
                    scenario=scenario,
                    agent_conf=agent_conf,
                    dt=float(target.dt),
                    device=device,
                    obs_noise=float(args.noise),
                    ts_grid=str(args.ts_grid),
                    solver=str(args.solver),
                    consensus_weight=float(args.consensus_weight),
                    num_env_workers=int(args.collect_parallel_workers),
                    agent_obsk=int(args.agent_obsk),
                )
                policy_fn = make_uniform_random_policy(env, seed=seed)
                stats = eval_with_runner_strategy(
                    env=env,
                    policy_fn=policy_fn,
                    num_episodes=int(args.num_episodes),
                    eval_horizon_sec=float(args.eval_horizon_sec),
                    device=device,
                )
            except Exception as e:
                print(f"[warn] env={env_alias} seed={seed} failed: {e}")
                continue

            true_rewards.append(stats["true_reward"])
            min_rewards.append(stats["min_reward"])
            solved_ratios.append(stats["solved_ratio"])
            ok_seeds.append(int(seed))
            num_steps_seen.add(int(stats["num_steps"]))
            per_seed_rows.append(
                {
                    "env": env_alias,
                    "dt": float(target.dt),
                    "seed": int(seed),
                    "true_reward": float(stats["true_reward"]),
                    "min_reward": float(stats["min_reward"]),
                    "solved_ratio": float(stats["solved_ratio"]),
                    "steps": int(stats["num_steps"]),
                }
            )

        if len(true_rewards) == 0:
            print(f"[warn] env={env_alias} has no successful evaluations, skip summary row.")
            continue

        mean_r, std_r, min_r, max_r = summarize(true_rewards)
        mean_min_r, _, _, _ = summarize(min_rewards)
        mean_solved, _, _, _ = summarize(solved_ratios)
        step_value = sorted(num_steps_seen)[0] if len(num_steps_seen) > 0 else "-"
        summary_rows.append(
            {
                "env": env_alias,
                "dt": float(target.dt),
                "n_seeds": len(ok_seeds),
                "steps": step_value,
                "mean_true_reward": mean_r,
                "std_true_reward": std_r,
                "min_true_reward": min_r,
                "max_true_reward": max_r,
                "mean_min_reward": mean_min_r,
                "mean_solved_ratio": mean_solved,
                "seeds": ",".join(str(s) for s in ok_seeds),
            }
        )

    if len(summary_rows) == 0:
        raise SystemExit("No valid environments evaluated.")

    table_headers = [
        "env",
        "dt",
        "steps",
        "n_seeds",
        "mean_true_reward",
        "std_true_reward",
        "min_true_reward",
        "max_true_reward",
        "mean_min_reward",
        "mean_solved_ratio",
    ]
    table_rows = []
    for row in summary_rows:
        table_rows.append(
            [
                row["env"],
                f'{row["dt"]:.4f}',
                str(row["steps"]),
                str(row["n_seeds"]),
                f'{row["mean_true_reward"]:.6f}',
                f'{row["std_true_reward"]:.6f}',
                f'{row["min_true_reward"]:.6f}',
                f'{row["max_true_reward"]:.6f}',
                f'{row["mean_min_reward"]:.6f}',
                f'{row["mean_solved_ratio"]:.6f}',
            ]
        )
    print("\nSummary (random policy reward with runner eval strategy):")
    print(build_ascii_table(table_headers, table_rows))

    if args.show_per_seed:
        per_headers = ["env", "dt", "seed", "steps", "true_reward", "min_reward", "solved_ratio"]
        per_rows = [
            [
                r["env"],
                f'{r["dt"]:.4f}',
                str(r["seed"]),
                str(r["steps"]),
                f'{r["true_reward"]:.6f}',
                f'{r["min_reward"]:.6f}',
                f'{r["solved_ratio"]:.6f}',
            ]
            for r in per_seed_rows
        ]
        print("\nPer-seed details:")
        print(build_ascii_table(per_headers, per_rows))

    maybe_write_csv(args.csv, summary_rows)


if __name__ == "__main__":
    main()
