import argparse
import os
import random
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_default_dtype(torch.float32)

ROOT = Path(__file__).resolve().parent
ONPOLICY_DIR = ROOT / "on-policy"
if str(ONPOLICY_DIR) not in sys.path:
    sys.path.insert(0, str(ONPOLICY_DIR))

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.utils.shared_buffer import SharedReplayBuffer

from runner_coop_ma_enode import (
    FIXED_EPISODE_SECONDS,
    MAMUJOCO_ENV_SPECS,
    MPE_ENV_ALIASES,
    MaMuJoCoEnv,
    MPECooperativeEnv,
    _make_indexed_output_prefix,
)

try:
    from gymnasium.spaces import Box
except Exception:
    from gym.spaces import Box


def _t2n(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _wandb_log(wandb_run, metrics: Dict[str, object], step: int = None) -> None:
    if wandb_run is None:
        return
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.item()
        elif isinstance(value, np.generic):
            value = value.item()
        clean[key] = value
    if not clean:
        return
    if step is None:
        wandb_run.log(clean)
    else:
        wandb_run.log(clean, step=step)


def _to_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().mean().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(np.asarray(value, dtype=np.float32).mean())
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _tensor_stats(prefix: str, tensor: torch.Tensor) -> Dict[str, float]:
    t = tensor.detach().to(torch.float32).reshape(-1)
    if t.numel() == 0:
        return {}
    out = {
        f"{prefix}_mean": float(t.mean().item()),
        f"{prefix}_std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        f"{prefix}_min": float(t.min().item()),
        f"{prefix}_max": float(t.max().item()),
    }
    return out


class MAPPOEpisodeWorker:
    """Single runtime environment worker built on runner_coop_ma_enode adapters."""

    def __init__(self, env, env_kind: str, worker_id: int = 0):
        self.env = env
        self.env_kind = str(env_kind)
        self.worker_id = int(worker_id)
        self.runtime_env = None
        self.last_obs_agents = None

    def close(self) -> None:
        if self.runtime_env is not None:
            self.runtime_env.close()
            self.runtime_env = None

    def reset(self, seed: int = None) -> np.ndarray:
        self.close()
        self.runtime_env = self.env._make_env()
        if seed is None:
            obs_dict, _ = self.runtime_env.reset()
        else:
            obs_dict, _ = self.runtime_env.reset(seed=int(seed))
        obs_agents = self._obs_dict_to_agents(obs_dict)
        self.last_obs_agents = obs_agents.copy()
        return obs_agents

    def _obs_dict_to_agents(self, obs_dict: dict) -> np.ndarray:
        try:
            joint = self.env._obs_dict_to_joint(obs_dict)
            return joint.reshape(self.env.n_agents, self.env.obs_dim).astype(np.float32)
        except Exception:
            if self.last_obs_agents is not None:
                return self.last_obs_agents.copy().astype(np.float32)
            return np.zeros((self.env.n_agents, self.env.obs_dim), dtype=np.float32)

    def _joint_action_from_agents(self, actions_agents: np.ndarray) -> np.ndarray:
        actions_agents = np.asarray(actions_agents, dtype=np.float32)
        if actions_agents.ndim != 2:
            raise RuntimeError(f"Expected [n_agents, act_dim], got shape={actions_agents.shape}")
        parts = []
        for aid, d in enumerate(self.env.act_dims):
            parts.append(actions_agents[aid, : int(d)])
        return np.concatenate(parts, axis=0).astype(np.float32)

    def step(self, actions_agents: np.ndarray) -> Tuple[np.ndarray, float, bool, List[dict]]:
        if self.runtime_env is None:
            raise RuntimeError("Worker env is not initialized. Call reset() first.")

        joint_action = self._joint_action_from_agents(actions_agents)

        if self.env_kind == "mamujoco":
            act_dict = self.env._split_joint_action(joint_action)
            next_obs, rewards, terms, truncs, _ = self.runtime_env.step(act_dict)
            reward_vec = np.asarray([rewards[a] for a in self.env.agent_ids], dtype=np.float32)
            if np.allclose(reward_vec, reward_vec[0], atol=1e-6):
                team_reward = float(reward_vec[0])
            else:
                team_reward = float(reward_vec.mean())
            done = bool(all(terms.values()) or all(truncs.values()))
            infos = [{"individual_reward": float(rewards.get(aid, 0.0))} for aid in self.env.agent_ids]
        elif self.env_kind == "mpe":
            env_actions, _ = self.env._build_action_dict(joint_action, self.runtime_env)
            next_obs, rewards, terms, truncs, _ = self.runtime_env.step(env_actions)
            team_reward = float(self.env._team_reward(rewards))
            done = bool(
                len(self.runtime_env.agents) == 0
                or (len(terms) > 0 and all(terms.values()))
                or (len(truncs) > 0 and all(truncs.values()))
            )
            infos = [{"individual_reward": float(rewards.get(aid, 0.0))} for aid in self.env.agent_ids]
        else:
            raise ValueError(f"Unsupported env_kind={self.env_kind}")

        obs_agents = self._obs_dict_to_agents(next_obs)
        self.last_obs_agents = obs_agents.copy()
        return obs_agents, team_reward, done, infos


def _build_share_obs(obs: np.ndarray) -> np.ndarray:
    # obs: [n_rollout_threads, n_agents, obs_dim]
    n_threads, n_agents = obs.shape[0], obs.shape[1]
    flat = obs.reshape(n_threads, -1)
    return np.repeat(flat[:, None, :], n_agents, axis=1).astype(np.float32)


def _build_policy_inference_fn(policy: R_MAPPOPolicy, env, args):
    num_agents = int(env.n_agents)
    obs_dim = int(env.obs_dim)
    recurrent_N = int(args.recurrent_N)
    hidden_size = int(args.hidden_size)
    use_recurrent = bool(
        getattr(args, "use_recurrent_policy", False) or getattr(args, "use_naive_recurrent_policy", False)
    )

    rnn_states = None
    masks = None
    batch_cache = None
    last_t = None

    def _init_eval_states(batch_size: int):
        states = np.zeros((batch_size * num_agents, recurrent_N, hidden_size), dtype=np.float32)
        mask = np.ones((batch_size * num_agents, 1), dtype=np.float32)
        return states, mask

    def reset_state():
        nonlocal rnn_states, masks, batch_cache, last_t
        rnn_states = None
        masks = None
        batch_cache = None
        last_t = None

    def _to_time_scalar(t):
        if isinstance(t, torch.Tensor):
            if t.numel() == 0:
                return None
            return float(t.detach().reshape(-1)[0].item())
        try:
            return float(t)
        except Exception:
            return None

    def g(s, t):
        nonlocal rnn_states, masks, batch_cache, last_t

        if isinstance(s, torch.Tensor):
            s_np = s.detach().cpu().numpy().astype(np.float32)
            out_device = s.device
            out_dtype = s.dtype
        else:
            s_np = np.asarray(s, dtype=np.float32)
            out_device = torch.device(args.device)
            out_dtype = torch.float32

        if s_np.ndim == 1:
            s_np = s_np.reshape(1, -1)
        batch = int(s_np.shape[0])
        t_scalar = _to_time_scalar(t)

        if use_recurrent:
            # For recurrent eval, reset hidden state when a new episode starts
            # (time restarts from 0 or decreases), and when batch size changes.
            is_new_episode = False
            if t_scalar is not None and last_t is not None:
                if (t_scalar <= 1e-8 and last_t > 1e-8) or (t_scalar < last_t - 1e-8):
                    is_new_episode = True
            if (rnn_states is None) or (batch_cache != batch) or is_new_episode:
                rnn_states, masks = _init_eval_states(batch)
                batch_cache = batch
            rnn_states_in = rnn_states
            masks_in = masks
        else:
            # For feed-forward MAPPO, keep eval inference stateless so env rollouts can be parallelized safely.
            rnn_states_in, masks_in = _init_eval_states(batch)

        obs_agents = s_np.reshape(batch, num_agents, obs_dim)
        obs_flat = obs_agents.reshape(batch * num_agents, obs_dim)

        with torch.no_grad():
            actions_t, rnn_states_t = policy.act(
                obs_flat,
                rnn_states_in,
                masks_in,
                available_actions=None,
                deterministic=True,
            )
        actions = _t2n(actions_t).reshape(batch, num_agents, -1)
        if use_recurrent:
            rnn_states = _t2n(rnn_states_t)
            last_t = t_scalar

        joint_actions = []
        for b in range(batch):
            parts = []
            for aid, d in enumerate(env.act_dims):
                parts.append(actions[b, aid, : int(d)])
            joint_actions.append(np.concatenate(parts, axis=0))
        joint_actions = np.stack(joint_actions, axis=0).astype(np.float32)

        return torch.as_tensor(joint_actions, dtype=out_dtype, device=out_device)

    g.reset_state = reset_state
    return g


def evaluate_with_runner_strategy(env, policy: R_MAPPOPolicy, args) -> Dict[str, float]:
    Ttest = max(1, int(np.round(float(FIXED_EPISODE_SECONDS) / float(env.dt))))
    Ntest = max(1, int(args.eval_episodes))
    eval_env_workers = max(1, int(args.eval_env_workers))

    if args.algorithm_name == "rmappo" and eval_env_workers > 1:
        print(
            "[eval] rmappo uses recurrent hidden states; "
            "forcing eval_env_workers=1 to avoid parallel-state races."
        )
        eval_env_workers = 1

    prev_env_workers = int(getattr(env, "num_env_workers", 1))
    if hasattr(env, "num_env_workers"):
        env.num_env_workers = eval_env_workers

    try:
        s0 = []
        for _ in range(Ntest):
            s0.append(torch.tensor(env.reset(), dtype=torch.float32, device=env.device))
        s0 = torch.stack(s0, dim=0)

        policy_fn = _build_policy_inference_fn(policy, env, args)
        if hasattr(policy_fn, "reset_state"):
            policy_fn.reset_state()

        eval_t0 = time.perf_counter()
        with torch.no_grad():
            _, _, test_rewards, _ = env.integrate_system(T=Ttest, s0=s0, g=policy_fn)
        eval_dt = time.perf_counter() - eval_t0

        rewards_are_accumulated = bool(getattr(env, "rewards_are_accumulated", False))
        Tup = 0
        if rewards_are_accumulated:
            reward_tensor = test_rewards[..., -1]
            true_reward = reward_tensor.mean().item()
            min_reward = reward_tensor.min().item()
        else:
            reward_tensor = test_rewards[..., Tup:]
            true_reward = reward_tensor.mean().item()
            min_reward = reward_tensor.min().item()
        solved_ratio = (reward_tensor >= 0.8).float().mean().item()

        metrics = {
            "eval/true_reward": float(true_reward),
            "eval/min_reward": float(min_reward),
            "eval/solved_ratio": float(solved_ratio),
            "eval/time_sec": float(eval_dt),
        }
        metrics.update(_tensor_stats("eval/test_reward", test_rewards))
        return metrics
    finally:
        if hasattr(env, "num_env_workers"):
            env.num_env_workers = prev_env_workers


def build_args():
    p = argparse.ArgumentParser(description="MAPPO runner migrated from on-policy for ma-ctrl env adapters.")

    p.add_argument(
        "--env",
        type=str,
        default="cooperative_navigation",
        choices=[*MAMUJOCO_ENV_SPECS.keys(), *MPE_ENV_ALIASES.keys()],
    )
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--ts_grid", type=str, default="fixed", choices=["fixed", "uniform", "exp"])
    p.add_argument("--solver", type=str, default="rk4")
    p.add_argument("--consensus_weight", type=float, default=0.02)
    p.add_argument("--collect_parallel_workers", type=int, default=1)
    p.add_argument("--agent_obsk", type=int, default=1)

    p.add_argument("--episodes", type=int, default=1000, help="Total training episodes.")
    p.add_argument("--episode_length", type=int, default=25)
    p.add_argument("--n_rollout_threads", type=int, default=1)
    p.add_argument(
        "--env_step_workers",
        type=int,
        default=0,
        help="Thread pool size for parallel worker.step in rollout. 0 means n_rollout_threads.",
    )
    p.add_argument("--eval_every_episodes", type=int, default=100)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument(
        "--eval_env_workers",
        type=int,
        default=1,
        help="Parallel env workers used by evaluate_with_runner_strategy.",
    )

    # on-policy MAPPO hyperparameters
    p.add_argument("--algorithm_name", type=str, default="mappo", choices=["mappo", "rmappo"])
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--layer_N", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--critic_lr", type=float, default=5e-4)
    p.add_argument("--opti_eps", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--ppo_epoch", type=int, default=15)
    p.add_argument("--num_mini_batch", type=int, default=1)
    p.add_argument("--clip_param", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_loss_coef", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--huber_delta", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)

    p.add_argument("--recurrent_N", type=int, default=1)
    p.add_argument("--data_chunk_length", type=int, default=10)

    p.add_argument("--use_linear_lr_decay", action="store_true")
    p.add_argument("--use_clipped_value_loss", action="store_true", default=True)
    p.add_argument("--use_huber_loss", action="store_true", default=True)
    p.add_argument("--use_popart", action="store_true", default=False)
    p.add_argument("--use_valuenorm", action="store_true", default=True)
    p.add_argument("--use_max_grad_norm", action="store_true", default=True)
    p.add_argument("--use_gae", action="store_true", default=True)
    p.add_argument("--use_proper_time_limits", action="store_true", default=False)
    p.add_argument("--use_value_active_masks", action="store_true", default=True)
    p.add_argument("--use_policy_active_masks", action="store_true", default=True)
    p.add_argument("--use_feature_normalization", action="store_true", default=True)
    p.add_argument("--use_orthogonal", action="store_true", default=True)
    p.add_argument("--use_ReLU", action="store_true", default=True)
    p.add_argument("--stacked_frames", type=int, default=1)
    p.add_argument("--use_stacked_frames", action="store_true", default=False)
    p.add_argument("--gain", type=float, default=0.01)

    p.add_argument("--n_training_threads", type=int, default=1)
    p.add_argument("--save_interval_episodes", type=int, default=500)

    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="ma-ctrl")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="mappo")
    p.add_argument("--wandb_name", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_tags", type=str, default="mappo,ctde,multi-agent")

    return p.parse_args()


def _build_env(args, device):
    if args.env in MAMUJOCO_ENV_SPECS:
        scenario, agent_conf = MAMUJOCO_ENV_SPECS[args.env]
        env = MaMuJoCoEnv(
            scenario=scenario,
            agent_conf=agent_conf,
            dt=args.dt,
            device=device,
            obs_noise=args.noise,
            ts_grid=args.ts_grid,
            solver=args.solver,
            consensus_weight=args.consensus_weight,
            num_env_workers=args.collect_parallel_workers,
            agent_obsk=args.agent_obsk,
        )
        env_kind = "mamujoco"
    elif args.env in MPE_ENV_ALIASES:
        env = MPECooperativeEnv(
            mpe_env_key=args.env,
            dt=args.dt,
            device=device,
            obs_noise=args.noise,
            ts_grid=args.ts_grid,
            solver=args.solver,
            consensus_weight=args.consensus_weight,
            num_env_workers=args.collect_parallel_workers,
        )
        env_kind = "mpe"
    else:
        raise ValueError(f"Unsupported env={args.env} for MAPPO runner.")

    fixed_episode_steps = max(1, int(np.round(FIXED_EPISODE_SECONDS / args.dt)))
    requested_horizon = args.episode_length * args.dt
    fixed_horizon = fixed_episode_steps * args.dt
    if not np.isclose(requested_horizon, fixed_horizon, rtol=0.0, atol=1e-9):
        print(
            f"Overriding requested episode length ({requested_horizon:.4f}s) "
            f"with fixed {fixed_horizon:.4f}s ({fixed_episode_steps} steps)."
        )
    args.episode_length = fixed_episode_steps

    n_agents = int(env.n_agents)
    obs_dim = int(env.obs_dim)

    act_dims = [int(d) for d in env.act_dims]
    if len(set(act_dims)) != 1:
        raise ValueError(
            f"MAPPO shared policy requires homogeneous action dims across agents, got {act_dims}."
        )
    act_dim = int(act_dims[0])

    obs_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    share_obs_space = Box(low=-np.inf, high=np.inf, shape=(int(env.n),), dtype=np.float32)

    low = np.asarray(env.agent_action_lows[0], dtype=np.float32).reshape(-1)
    high = np.asarray(env.agent_action_highs[0], dtype=np.float32).reshape(-1)
    if low.shape[0] != act_dim or high.shape[0] != act_dim:
        raise RuntimeError(
            f"Action bound shape mismatch: low={low.shape}, high={high.shape}, act_dim={act_dim}"
        )
    act_space = Box(low=low, high=high, shape=(act_dim,), dtype=np.float32)

    return env, env_kind, n_agents, obs_space, share_obs_space, act_space


def _init_workers(env, env_kind: str, n_rollout_threads: int) -> List[MAPPOEpisodeWorker]:
    return [MAPPOEpisodeWorker(env, env_kind=env_kind, worker_id=i) for i in range(int(n_rollout_threads))]


def _save_checkpoint(policy: R_MAPPOPolicy, checkpoint_prefix: str) -> Tuple[str, str]:
    actor_path = checkpoint_prefix + "_actor.pt"
    critic_path = checkpoint_prefix + "_critic.pt"
    torch.save(policy.actor.state_dict(), actor_path)
    torch.save(policy.critic.state_dict(), critic_path)
    return actor_path, critic_path


def main():
    args = build_args()

    if args.n_training_threads is not None and args.n_training_threads > 0:
        torch.set_num_threads(int(args.n_training_threads))

    if args.algorithm_name == "rmappo":
        args.use_recurrent_policy = True
        args.use_naive_recurrent_policy = False
    else:
        args.use_recurrent_policy = False
        args.use_naive_recurrent_policy = False

    if args.n_rollout_threads < 1:
        raise ValueError("--n_rollout_threads must be >= 1")
    if args.env_step_workers < 0:
        raise ValueError("--env_step_workers must be >= 0")
    if args.eval_env_workers < 1:
        raise ValueError("--eval_env_workers must be >= 1")
    if args.eval_every_episodes < 1:
        raise ValueError("--eval_every_episodes must be >= 1")

    requested_device = torch.device(args.device)
    device = requested_device
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print(
            f"[WARN] Requested device '{args.device}' but CUDA is unavailable; "
            "falling back to CPU."
        )
        device = torch.device("cpu")
    args.device = str(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env, env_kind, num_agents, obs_space, share_obs_space, act_space = _build_env(args, device)

    run_name = f"{env.name}-{args.algorithm_name}-ma{num_agents}-seed{args.seed}"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_prefix = _make_indexed_output_prefix(run_name)
    checkpoint_dir = ROOT / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Env={env.name}, env_kind={env_kind}, agents={num_agents}, dt={env.dt:.3f}, ep_len={args.episode_length}")
    print(f"output_prefix={run_output_prefix}")
    print(f"checkpoint_dir={checkpoint_dir}")

    policy_args = SimpleNamespace(**vars(args))
    policy = R_MAPPOPolicy(policy_args, obs_space, share_obs_space, act_space, device=device)
    trainer = R_MAPPO(policy_args, policy, device=device)
    buffer = SharedReplayBuffer(policy_args, num_agents, obs_space, share_obs_space, act_space)

    if args.use_wandb:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
        except OSError as e:
            args.use_wandb = False
            print(f"[WARN] Local socket bind unavailable ({e}); disabling wandb logging.")

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
        except ImportError:
            print("[WARN] --use_wandb is set but wandb is not installed; continuing without wandb logging.")
        else:
            try:
                wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
                wandb_kwargs = {
                    "project": args.wandb_project,
                    "name": args.wandb_name if args.wandb_name else run_name,
                    "group": args.wandb_group if args.wandb_group else None,
                    "mode": args.wandb_mode,
                    "config": vars(args),
                    "tags": wandb_tags,
                }
                if args.wandb_entity:
                    wandb_kwargs["entity"] = args.wandb_entity

                wandb_run = wandb.init(**wandb_kwargs)
                wandb.define_metric("train/episode")
                wandb.define_metric("train/*", step_metric="train/episode")
                wandb.define_metric("policy/update")
                wandb.define_metric("policy/*", step_metric="policy/update")
                wandb.define_metric("eval/episode")
                wandb.define_metric("eval/*", step_metric="eval/episode")
                wandb_run.config.update(
                    {
                        "env_name": env.name,
                        "run_name": run_name,
                        "num_agents": num_agents,
                        "obs_dim": int(env.obs_dim),
                        "act_dims": [int(d) for d in env.act_dims],
                    },
                    allow_val_change=True,
                )
                print(f"wandb enabled: project={args.wandb_project}, run={wandb_run.name}")
            except Exception as e:
                wandb_run = None
                print(
                    f"[WARN] wandb init failed ({type(e).__name__}: {e}); "
                    "continuing without wandb logging."
                )

    workers = _init_workers(env, env_kind, args.n_rollout_threads)
    env_step_workers = int(args.env_step_workers) if int(args.env_step_workers) > 0 else int(args.n_rollout_threads)
    use_parallel_env_step = bool(args.n_rollout_threads > 1 and env_step_workers > 1)
    env_step_executor = ThreadPoolExecutor(max_workers=env_step_workers) if use_parallel_env_step else None

    total_episodes_target = int(args.episodes)
    updates = int(np.ceil(total_episodes_target / float(args.n_rollout_threads)))
    total_env_steps = 0
    episodes_done = 0
    best_eval_reward = -np.inf
    best_ckpt = None
    policy_update_idx = 0
    next_eval_episode = int(args.eval_every_episodes)

    print(
        f"Training updates={updates}, target_episodes={total_episodes_target}, "
        f"episodes_per_update={args.n_rollout_threads}, eval_every={args.eval_every_episodes}, "
        f"parallel_env_step={use_parallel_env_step}, env_step_workers={env_step_workers}, "
        f"eval_env_workers={args.eval_env_workers}"
    )

    try:
        for update in range(updates):
            if args.use_linear_lr_decay:
                policy.lr_decay(update, max(1, updates))

            obs = np.zeros((args.n_rollout_threads, num_agents, int(env.obs_dim)), dtype=np.float32)
            for wid, worker in enumerate(workers):
                rollout_seed = args.seed + update * 10000 + wid * 100
                obs[wid] = worker.reset(seed=rollout_seed)

            share_obs = _build_share_obs(obs)
            buffer.step = 0
            buffer.share_obs[0] = share_obs.copy()
            buffer.obs[0] = obs.copy()

            rnn_states = np.zeros(
                (args.n_rollout_threads, num_agents, args.recurrent_N, args.hidden_size),
                dtype=np.float32,
            )
            rnn_states_critic = np.zeros_like(rnn_states)
            masks = np.ones((args.n_rollout_threads, num_agents, 1), dtype=np.float32)
            active_masks = np.ones_like(masks)

            thread_returns = np.zeros((args.n_rollout_threads,), dtype=np.float32)
            thread_alive = np.ones((args.n_rollout_threads,), dtype=bool)

            rollout_t0 = time.perf_counter()
            for step in range(args.episode_length):
                trainer.prep_rollout()

                with torch.no_grad():
                    values_t, actions_t, action_log_probs_t, rnn_states_t, rnn_states_critic_t = policy.get_actions(
                        share_obs.reshape(-1, share_obs.shape[-1]),
                        obs.reshape(-1, obs.shape[-1]),
                        rnn_states.reshape(-1, args.recurrent_N, args.hidden_size),
                        rnn_states_critic.reshape(-1, args.recurrent_N, args.hidden_size),
                        masks.reshape(-1, 1),
                    )

                values = _t2n(values_t).reshape(args.n_rollout_threads, num_agents, -1).astype(np.float32)
                actions = _t2n(actions_t).reshape(args.n_rollout_threads, num_agents, -1).astype(np.float32)
                action_log_probs = (
                    _t2n(action_log_probs_t).reshape(args.n_rollout_threads, num_agents, -1).astype(np.float32)
                )
                rnn_states_next = (
                    _t2n(rnn_states_t)
                    .reshape(args.n_rollout_threads, num_agents, args.recurrent_N, args.hidden_size)
                    .astype(np.float32)
                )
                rnn_states_critic_next = (
                    _t2n(rnn_states_critic_t)
                    .reshape(args.n_rollout_threads, num_agents, args.recurrent_N, args.hidden_size)
                    .astype(np.float32)
                )

                rewards = np.zeros((args.n_rollout_threads, num_agents, 1), dtype=np.float32)
                dones = np.zeros((args.n_rollout_threads, num_agents), dtype=bool)
                next_obs = obs.copy()

                if env_step_executor is None:
                    for wid, worker in enumerate(workers):
                        if thread_alive[wid]:
                            obs_i, team_rew_i, done_i, _ = worker.step(actions[wid])
                            next_obs[wid] = obs_i
                            rewards[wid, :, 0] = float(team_rew_i)
                            thread_returns[wid] += float(team_rew_i)
                            dones[wid, :] = bool(done_i)
                            if done_i:
                                thread_alive[wid] = False
                        else:
                            dones[wid, :] = True
                else:
                    future_pairs = []
                    for wid, worker in enumerate(workers):
                        if thread_alive[wid]:
                            future_pairs.append((wid, env_step_executor.submit(worker.step, actions[wid])))
                        else:
                            dones[wid, :] = True
                    for wid, fut in future_pairs:
                        obs_i, team_rew_i, done_i, _ = fut.result()
                        next_obs[wid] = obs_i
                        rewards[wid, :, 0] = float(team_rew_i)
                        thread_returns[wid] += float(team_rew_i)
                        dones[wid, :] = bool(done_i)
                        if done_i:
                            thread_alive[wid] = False

                rnn_states_next[dones == True] = 0.0
                rnn_states_critic_next[dones == True] = 0.0

                masks_next = np.ones((args.n_rollout_threads, num_agents, 1), dtype=np.float32)
                masks_next[dones == True] = 0.0
                active_masks_next = np.ones_like(masks_next)
                active_masks_next[dones == True] = 0.0

                share_obs_next = _build_share_obs(next_obs)

                buffer.insert(
                    share_obs_next,
                    next_obs,
                    rnn_states_next,
                    rnn_states_critic_next,
                    actions,
                    action_log_probs,
                    values,
                    rewards,
                    masks_next,
                    active_masks=active_masks_next,
                )

                obs = next_obs
                share_obs = share_obs_next
                rnn_states = rnn_states_next
                rnn_states_critic = rnn_states_critic_next
                masks = masks_next
                active_masks = active_masks_next

            rollout_dt = time.perf_counter() - rollout_t0

            trainer.prep_rollout()
            with torch.no_grad():
                next_values_t = policy.get_values(
                    buffer.share_obs[-1].reshape(-1, buffer.share_obs.shape[-1]),
                    buffer.rnn_states_critic[-1].reshape(-1, args.recurrent_N, args.hidden_size),
                    buffer.masks[-1].reshape(-1, 1),
                )
            next_values = _t2n(next_values_t).reshape(args.n_rollout_threads, num_agents, -1).astype(np.float32)
            buffer.compute_returns(next_values, trainer.value_normalizer)

            trainer.prep_training()
            train_info = trainer.train(buffer)
            buffer.after_update()

            policy_update_idx += 1
            episodes_done += int(args.n_rollout_threads)
            total_env_steps += int(args.n_rollout_threads * args.episode_length)

            mean_ep_return = float(thread_returns.mean())
            min_ep_return = float(thread_returns.min())
            max_ep_return = float(thread_returns.max())

            print(
                f"Update {update + 1}/{updates} | episodes={episodes_done}/{total_episodes_target} "
                f"| mean_ep_return={mean_ep_return:.4f} min={min_ep_return:.4f} max={max_ep_return:.4f} "
                f"| rollout_sec={rollout_dt:.3f}"
            )

            train_metrics = {
                "train/episode": int(min(episodes_done, total_episodes_target)),
                "train/episodes_done": int(episodes_done),
                "train/total_env_steps": int(total_env_steps),
                "train/mean_episode_return": mean_ep_return,
                "train/min_episode_return": min_ep_return,
                "train/max_episode_return": max_ep_return,
                "train/rollout_time_sec": float(rollout_dt),
                "train/episode_length_steps": int(args.episode_length),
            }
            policy_metrics = {
                "policy/update": int(policy_update_idx),
                "policy/value_loss": _to_scalar(train_info.get("value_loss", 0.0)),
                "policy/policy_loss": _to_scalar(train_info.get("policy_loss", 0.0)),
                "policy/dist_entropy": _to_scalar(train_info.get("dist_entropy", 0.0)),
                "policy/actor_grad_norm": _to_scalar(train_info.get("actor_grad_norm", 0.0)),
                "policy/critic_grad_norm": _to_scalar(train_info.get("critic_grad_norm", 0.0)),
                "policy/ratio": _to_scalar(train_info.get("ratio", 0.0)),
            }
            _wandb_log(wandb_run, train_metrics)
            _wandb_log(wandb_run, policy_metrics)

            if args.save_interval_episodes > 0 and episodes_done % args.save_interval_episodes == 0:
                ckpt_prefix = str(checkpoint_dir / f"mappo_seed{args.seed}_run{run_timestamp}_ep{episodes_done}")
                actor_path, critic_path = _save_checkpoint(policy, ckpt_prefix)
                print(f"Checkpoint saved: actor={actor_path}, critic={critic_path}")

            episodes_for_schedule = int(min(episodes_done, total_episodes_target))
            while episodes_for_schedule >= next_eval_episode:
                eval_metrics = evaluate_with_runner_strategy(env, policy, args)
                eval_ep_mark = int(next_eval_episode)
                eval_metrics["eval/episode"] = eval_ep_mark

                print(
                    f"Eval @ episode={eval_ep_mark}: "
                    f"true_reward={eval_metrics['eval/true_reward']:.4f}, "
                    f"min_reward={eval_metrics['eval/min_reward']:.4f}, "
                    f"solved_ratio={eval_metrics['eval/solved_ratio']:.4f}"
                )

                if np.isfinite(eval_metrics["eval/true_reward"]) and eval_metrics["eval/true_reward"] > best_eval_reward:
                    best_eval_reward = float(eval_metrics["eval/true_reward"])
                    best_prefix = str(
                        checkpoint_dir
                        / f"best_mappo_seed{args.seed}_run{run_timestamp}_env{env.name.replace('/', '-')}_ep{eval_ep_mark}"
                    )
                    actor_path, critic_path = _save_checkpoint(policy, best_prefix)
                    best_ckpt = (actor_path, critic_path)
                    print(
                        f"[best-checkpoint] updated: reward={best_eval_reward:.6f}, "
                        f"actor={actor_path}, critic={critic_path}"
                    )

                _wandb_log(wandb_run, eval_metrics)
                _wandb_log(
                    wandb_run,
                    {
                        "train/episode": int(min(episodes_done, total_episodes_target)),
                        "train/best_eval_true_reward": float(best_eval_reward),
                    },
                )
                next_eval_episode += int(args.eval_every_episodes)

    finally:
        for worker in workers:
            worker.close()
        if env_step_executor is not None:
            env_step_executor.shutdown(wait=True, cancel_futures=False)
        if wandb_run is not None:
            wandb_run.finish()

    print("Training finished.")
    print(f"episodes_done={episodes_done}, total_env_steps={total_env_steps}, best_eval_true_reward={best_eval_reward:.6f}")
    if best_ckpt is not None:
        print(f"best_actor={best_ckpt[0]}")
        print(f"best_critic={best_ckpt[1]}")


if __name__ == "__main__":
    main()
