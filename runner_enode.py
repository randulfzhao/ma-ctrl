import os
import argparse
from typing import List, Dict
from pathlib import Path
from datetime import datetime
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import numpy as np
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_default_dtype(torch.float32)

ROOT = Path(__file__).resolve().parent
ODERL_DIR = ROOT / "oderl-main"
if str(ODERL_DIR) not in sys.path:
    sys.path.insert(0, str(ODERL_DIR))

import envs
import ctrl.ctrl as base
from ctrl import utils
from ctrl.policy import Policy
from envs.base_env import BaseEnv

FIXED_EPISODE_SECONDS = 2.5
MAMUJOCO_ENV_SPECS = {
    "ant2x4": ("Ant", "2x4"),
    "ant2x4d": ("Ant", "2x4d"),
    "ant4x2": ("Ant", "4x2"),
    "walker": ("Walker2d", "2x3"),
    "walker2x3": ("Walker2d", "2x3"),
    "cheetah": ("HalfCheetah", "6x1"),
    "cheetah6x1": ("HalfCheetah", "6x1"),
    "swimmer": ("Swimmer", "2x1"),
    "swimmer2x1": ("Swimmer", "2x1"),
}
MPE_ENV_SPECS: Dict[str, Dict[str, object]] = {
    "cooperative_navigation": {
        "scenario": "simple_spread_v3",
        "kwargs": {
            "N": 3,
            "local_ratio": 0.0,
        },
        "controlled_role": "all",
    },
    "cooperative_predator_prey": {
        "scenario": "simple_tag_v3",
        "kwargs": {
            "num_good": 1,
            "num_adversaries": 3,
            "num_obstacles": 2,
        },
        "controlled_role": "adversary",
    },
}
MPE_ENV_ALIASES = {
    "cooperative_navigation": "cooperative_navigation",
    "coop_navigation": "cooperative_navigation",
    "navigation": "cooperative_navigation",
    "cooperative_predator_prey": "cooperative_predator_prey",
    "coop_predator_prey": "cooperative_predator_prey",
    "predator_prey": "cooperative_predator_prey",
    "cooperative_predactor_pray": "cooperative_predator_prey",
}


class LocalPolicyEnv:
    """Minimal env spec used to initialize one independent actor per agent."""

    def __init__(self, obs_dim: int, act_low: np.ndarray, act_high: np.ndarray, device):
        self.n = int(obs_dim)
        self.m = int(np.asarray(act_low).reshape(-1).shape[0])
        act_low = np.asarray(act_low, dtype=np.float32).reshape(-1)
        act_high = np.asarray(act_high, dtype=np.float32).reshape(-1)
        max_abs = float(np.max(np.abs(np.concatenate([act_low, act_high]))))
        self.act_rng = max(1e-6, max_abs)
        self.ac_lb = torch.tensor(act_low, dtype=torch.float32, device=device)
        self.ac_ub = torch.tensor(act_high, dtype=torch.float32, device=device)


class AgentPolicyController(nn.Module):
    """One decentralized actor controller for a single agent."""

    def __init__(self, agent_idx: int, local_env: LocalPolicyEnv, nl_g: int, nn_g: int, act_g: str):
        super().__init__()
        self.agent_idx = int(agent_idx)
        self.env = local_env
        self._g = Policy(local_env, nl=nl_g, nn=nn_g, act=act_g)

    def forward(self, s, t):
        return self._g(s, t)


class CTDEJointPolicy(nn.Module):
    """Joint policy wrapper with independent per-agent actors (decentralized execution)."""

    def __init__(self, env: BaseEnv, agent_ctrls: List[AgentPolicyController], obs_slices: List[slice], act_dims: List[int]):
        super().__init__()
        if len(agent_ctrls) != len(obs_slices):
            raise ValueError("agent_ctrls and obs_slices length mismatch")
        self.env = env
        self.agent_ctrls = nn.ModuleList(agent_ctrls)
        self.obs_slices = list(obs_slices)
        self.act_dims = [int(d) for d in act_dims]
        self.n_agents = len(agent_ctrls)

    def forward(self, s, t):
        acts = []
        for i, agent_ctrl in enumerate(self.agent_ctrls):
            si = s[..., self.obs_slices[i]]
            ai = agent_ctrl(si, t)
            if ai.shape[-1] != self.act_dims[i]:
                raise RuntimeError(
                    f"Agent {i} action dim mismatch: expected {self.act_dims[i]}, got {ai.shape[-1]}"
                )
            acts.append(ai)
        return torch.cat(acts, dim=-1)


def _infer_obs_dims(env: BaseEnv, n_agents: int) -> List[int]:
    if hasattr(env, "obs_dim"):
        return [int(getattr(env, "obs_dim"))] * n_agents
    if hasattr(env, "n_i"):
        return [int(getattr(env, "n_i"))] * n_agents
    if int(env.n) % int(n_agents) != 0:
        raise ValueError(f"Cannot evenly split state dim n={env.n} across n_agents={n_agents}.")
    return [int(env.n) // int(n_agents)] * n_agents


def _infer_action_specs(env: BaseEnv, n_agents: int):
    if hasattr(env, "act_dims"):
        act_dims = [int(d) for d in getattr(env, "act_dims")]
    elif hasattr(env, "m_i"):
        act_dims = [int(getattr(env, "m_i"))] * n_agents
    else:
        if int(env.m) % int(n_agents) != 0:
            raise ValueError(f"Cannot evenly split action dim m={env.m} across n_agents={n_agents}.")
        act_dims = [int(env.m) // int(n_agents)] * n_agents
    if len(act_dims) != n_agents:
        raise ValueError(f"Action dim list length {len(act_dims)} does not match n_agents={n_agents}.")
    if int(sum(act_dims)) != int(env.m):
        raise ValueError(f"Action dims sum {sum(act_dims)} does not match env.m={env.m}.")

    if hasattr(env, "agent_action_lows") and hasattr(env, "agent_action_highs"):
        lows = [np.asarray(x, dtype=np.float32).reshape(-1) for x in getattr(env, "agent_action_lows")]
        highs = [np.asarray(x, dtype=np.float32).reshape(-1) for x in getattr(env, "agent_action_highs")]
    else:
        lb = env.ac_lb.detach().cpu().numpy().astype(np.float32).reshape(-1)
        ub = env.ac_ub.detach().cpu().numpy().astype(np.float32).reshape(-1)
        lows, highs = [], []
        offset = 0
        for d in act_dims:
            lows.append(lb[offset : offset + d])
            highs.append(ub[offset : offset + d])
            offset += d
    return act_dims, lows, highs


def _build_obs_slices(obs_dims: List[int], total_n: int) -> List[slice]:
    obs_slices = []
    offset = 0
    for d in obs_dims:
        obs_slices.append(slice(offset, offset + int(d)))
        offset += int(d)
    if int(offset) != int(total_n):
        raise ValueError(f"Obs dims sum {offset} does not match total_n={total_n}.")
    return obs_slices


def build_ctde_multi_controller_policy(
    env: BaseEnv,
    n_agents: int,
    device,
    nl_g: int,
    nn_g: int,
    act_g: str,
):
    obs_dims = _infer_obs_dims(env, n_agents)
    obs_slices = _build_obs_slices(obs_dims, env.n)
    act_dims, act_lows, act_highs = _infer_action_specs(env, n_agents)

    agent_ctrls: List[AgentPolicyController] = []
    for i in range(n_agents):
        local_env = LocalPolicyEnv(
            obs_dim=obs_dims[i],
            act_low=act_lows[i],
            act_high=act_highs[i],
            device=device,
        )
        agent_ctrls.append(
            AgentPolicyController(
                agent_idx=i,
                local_env=local_env,
                nl_g=int(nl_g),
                nn_g=int(nn_g),
                act_g=str(act_g),
            )
        )

    joint_policy = CTDEJointPolicy(env, agent_ctrls, obs_slices=obs_slices, act_dims=act_dims).to(device)
    return joint_policy, agent_ctrls, obs_dims, act_dims


def _make_indexed_output_prefix(run_name: str) -> str:
    now = datetime.now()
    date_dir = Path("output") / now.strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    time_tag = now.strftime("%H%M%S")
    next_idx = 1
    for child in date_dir.iterdir():
        if not child.is_dir():
            continue
        idx_token = child.name.split("_", 1)[0]
        if idx_token.isdigit():
            next_idx = max(next_idx, int(idx_token) + 1)
    while True:
        run_dir = date_dir / f"{next_idx:03d}_{time_tag}_{run_name}"
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            next_idx += 1
    return str(run_dir / run_name)


class MaMuJoCoEnv(BaseEnv):
    """MaMuJoCo adapter so the ENODE pipeline can train on PettingZoo rollouts."""

    def __init__(
        self,
        scenario: str,
        agent_conf: str,
        dt: float,
        device,
        obs_noise: float,
        ts_grid: str,
        solver: str,
        consensus_weight: float = 0.0,
        ac_rew_const: float = 0.01,
        num_env_workers: int = 1,
        agent_obsk: int = 1,
    ):
        try:
            from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1 as mamujoco
        except Exception as e:
            raise ImportError(
                "MaMuJoCo not available. Please run with conda env 'mbrl' (gymnasium_robotics installed)."
            ) from e

        self.mamujoco = mamujoco
        self.scenario = scenario
        self.agent_conf = agent_conf
        self.agent_obsk = int(agent_obsk)
        self.consensus_weight = consensus_weight
        self.num_env_workers = max(1, int(num_env_workers))
        self.use_env_rewards = True
        self.rewards_are_accumulated = True
        self.use_solved_threshold = False

        env = self._make_env()
        obs, _ = env.reset(seed=0)
        self.agent_ids = list(env.agents)
        self.n_agents = len(self.agent_ids)
        self.agent_obs_dims = [int(np.asarray(obs[a]).shape[0]) for a in self.agent_ids]
        # Some MaMuJoCo configs (e.g., HalfCheetah-6x1) have heterogeneous local
        # observation sizes across agents. Pad shorter vectors to keep a stable joint
        # state shape compatible with the rest of the ENODE pipeline.
        self.obs_dim = int(max(self.agent_obs_dims))
        self.act_dims = [int(np.prod(env.action_space(a).shape)) for a in self.agent_ids]
        self.m_i = self.act_dims[0]
        self.n_i = self.obs_dim
        action_high = float(np.max(np.abs(env.action_space(self.agent_ids[0]).high)))
        self.agent_action_lows = [np.asarray(env.action_space(a).low, dtype=np.float32) for a in self.agent_ids]
        self.agent_action_highs = [np.asarray(env.action_space(a).high, dtype=np.float32) for a in self.agent_ids]
        if len(set(self.agent_obs_dims)) > 1:
            print(
                f"[MaMuJoCoEnv] heterogeneous obs dims {self.agent_obs_dims}; "
                f"padding to {self.obs_dim} per agent."
            )
        env.close()

        n = self.n_agents * self.obs_dim
        m = int(sum(self.act_dims))
        names = [f"agent{i}.obs{j}" for i in range(self.n_agents) for j in range(self.obs_dim)]
        names += [f"agent{i}.act{j}" for i, d in enumerate(self.act_dims) for j in range(d)]
        super().__init__(
            dt=dt,
            n=n,
            m=m,
            act_rng=action_high,
            obs_trans=False,
            name=f"mamujoco-{self.scenario.lower()}-{self.agent_conf}",
            state_actions_names=names,
            device=device,
            solver=solver,
            obs_noise=obs_noise,
            ts_grid=ts_grid,
            ac_rew_const=ac_rew_const,
            vel_rew_const=0.0,
        )
        self.N0 = 10
        self.Nexpseq = 2
        self.reward_range = [-10.0, 10.0]
        self.reset()

    def __getstate__(self):
        state = super().__getstate__()
        if 'mamujoco' in state:
            del state['mamujoco']
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1 as mamujoco
        self.mamujoco = mamujoco

    def _make_env(self):
        kwargs = {
            "scenario": self.scenario,
            "agent_conf": self.agent_conf,
            "agent_obsk": self.agent_obsk,
            "render_mode": None,
        }
        # Avoid early-termination artifacts for environments that support this argument.
        if self.scenario in {"Ant", "Walker2d", "Hopper", "Humanoid", "HumanoidStandup"}:
            kwargs["terminate_when_unhealthy"] = False
        return self.mamujoco.parallel_env(**kwargs)

    def _obs_dict_to_joint(self, obs_dict):
        parts = []
        for aid in self.agent_ids:
            obs = np.asarray(obs_dict[aid], dtype=np.float32).reshape(-1)
            if obs.shape[0] < self.obs_dim:
                obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]), mode='constant')
            elif obs.shape[0] > self.obs_dim:
                obs = obs[:self.obs_dim]
            parts.append(obs)
        return np.concatenate(parts, axis=0)

    def _split_joint_action(self, joint_action):
        acts = {}
        offset = 0
        for i, aid in enumerate(self.agent_ids):
            d = self.act_dims[i]
            raw = joint_action[offset : offset + d]
            clipped = np.clip(raw, self.agent_action_lows[i], self.agent_action_highs[i]).astype(np.float32)
            acts[aid] = clipped
            offset += d
        return acts

    def torch_transform_states(self, state):
        return state

    def obs2state(self, state):
        return state

    def reset(self):
        env = self._make_env()
        obs, _ = env.reset()
        self.state = self._obs_dict_to_joint(obs)
        env.close()
        return self.state.copy()

    def torch_rhs(self, state, action):
        # True rhs is unknown for black-box MuJoCo stepping; integrate_system is overridden.
        return torch.zeros_like(state)

    def diff_obs_reward_(self, s):
        sj = s.reshape(*s.shape[:-1], self.n_agents, self.obs_dim)
        progress = sj[..., 0].mean(dim=-1)
        centered = sj - sj.mean(dim=-2, keepdim=True)
        disagreement = (centered**2).mean(dim=(-1, -2))
        posture_pen = 1e-3 * (sj[..., 1:] ** 2).mean(dim=(-1, -2))
        return progress - self.consensus_weight * disagreement - posture_pen

    def diff_ac_reward_(self, a):
        return -self.ac_rew_const * torch.sum(a**2, dim=-1)

    def integrate_system(self, T, g, s0=None, N=1, return_states=False, reset_seeds=None):
        with torch.no_grad():
            if s0 is not None:
                N = int(s0.shape[0])
            if reset_seeds is None:
                episode_seeds = [None] * int(N)
            else:
                episode_seeds = [int(seed) for seed in reset_seeds]
                if len(episode_seeds) != int(N):
                    raise ValueError(
                        f"reset_seeds length mismatch: expected {int(N)}, got {len(episode_seeds)}"
                    )
            ts_single = self.dt * torch.arange(T, dtype=torch.float32, device=self.device)
            sts, ats, rts = [], [], []

            def _rollout_one_episode(reset_seed=None):
                env = self._make_env()
                if reset_seed is None:
                    obs, _ = env.reset()
                else:
                    obs, _ = env.reset(seed=int(reset_seed))
                st_ep, at_ep, rt_ep = [], [], []
                ep_return = 0.0
                for t_idx in range(T):
                    joint_obs_np = self._obs_dict_to_joint(obs)
                    joint_obs = torch.tensor(joint_obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                    t_tensor = ts_single[t_idx]
                    action = g(joint_obs, t_tensor)
                    if isinstance(action, np.ndarray):
                        joint_action = action.reshape(-1).astype(np.float32)
                    else:
                        joint_action = action.detach().reshape(-1).cpu().numpy().astype(np.float32)
                    act_dict = self._split_joint_action(joint_action)
                    clipped_joint = np.concatenate(
                        [np.asarray(act_dict[aid], dtype=np.float32).reshape(-1) for aid in self.agent_ids],
                        axis=0,
                    )
                    st_ep.append(joint_obs.squeeze(0).cpu())
                    # Record the action actually executed in env (post-clipping),
                    # keeping action semantics consistent with MPE rollouts.
                    at_ep.append(torch.tensor(clipped_joint, dtype=torch.float32))
                    obs, rewards, terms, truncs, _ = env.step(act_dict)
                    reward_vec = np.asarray([rewards[a] for a in self.agent_ids], dtype=np.float32)
                    if np.allclose(reward_vec, reward_vec[0], atol=1e-6):
                        team_reward = float(reward_vec[0])
                    else:
                        team_reward = float(reward_vec.mean())
                    ep_return += team_reward
                    rt_ep.append(ep_return)
                    done = all(terms.values()) or all(truncs.values())
                    if done:
                        break
                env.close()
                return st_ep, at_ep, rt_ep

            workers = max(1, min(self.num_env_workers, N))
            if workers == 1:
                rollouts = [_rollout_one_episode(episode_seeds[i]) for i in range(N)]
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(_rollout_one_episode, episode_seeds[i]) for i in range(N)]
                    rollouts = [f.result() for f in futures]

            for st_ep, at_ep, rt_ep in rollouts:
                if len(st_ep) == 0:
                    st_ep.append(torch.zeros(self.n, dtype=torch.float32))
                    at_ep.append(torch.zeros(self.m, dtype=torch.float32))
                    rt_ep.append(0.0)
                while len(st_ep) < T:
                    st_ep.append(st_ep[-1].clone())
                    at_ep.append(torch.zeros_like(at_ep[-1]))
                    rt_ep.append(rt_ep[-1])

                sts.append(torch.stack(st_ep[:T]))
                ats.append(torch.stack(at_ep[:T]))
                rts.append(torch.tensor(rt_ep[:T], dtype=torch.float32))

            st = torch.stack(sts).to(self.device)
            at = torch.stack(ats).to(self.device)
            rt = torch.stack(rts).to(self.device)
            ts = torch.stack([ts_single] * N)
            returns = [st, at, rt, ts]
            if return_states:
                returns.append(st)
            return returns

    def render(self, mode='human', **kwargs):
        return None


class MPECooperativeEnv(BaseEnv):
    """PettingZoo MPE adapter for cooperative_navigation and cooperative_predator_prey."""

    def __init__(
        self,
        mpe_env_key: str,
        dt: float,
        device,
        obs_noise: float,
        ts_grid: str,
        solver: str,
        consensus_weight: float = 0.0,
        ac_rew_const: float = 0.01,
        num_env_workers: int = 1,
    ):
        canonical_key = MPE_ENV_ALIASES.get(str(mpe_env_key), str(mpe_env_key))
        if canonical_key not in MPE_ENV_SPECS:
            raise ValueError(f"Unsupported MPE env key: {mpe_env_key}")
        self.mpe_env_key = canonical_key
        self.consensus_weight = float(consensus_weight)
        self.num_env_workers = max(1, int(num_env_workers))
        self.use_env_rewards = True
        self.rewards_are_accumulated = True
        self.use_solved_threshold = False
        self.max_cycles = max(1, int(np.round(FIXED_EPISODE_SECONDS / dt)))

        spec = MPE_ENV_SPECS[self.mpe_env_key]
        self.mpe_scenario = str(spec["scenario"])
        self.mpe_kwargs = dict(spec.get("kwargs", {}))
        self.controlled_role = str(spec.get("controlled_role", "all"))
        self.continuous_actions = True

        env = self._make_env()
        obs, _ = env.reset(seed=0)
        self.all_agent_ids = list(env.agents)
        if self.controlled_role == "adversary":
            self.agent_ids = [aid for aid in self.all_agent_ids if aid.startswith("adversary_")]
            if len(self.agent_ids) == 0:
                raise RuntimeError("Expected adversary agents but none were found.")
        else:
            self.agent_ids = list(self.all_agent_ids)
        self.reward_agent_ids = list(self.agent_ids)
        self.n_agents = len(self.agent_ids)
        self.agent_obs_dims = [int(np.asarray(obs[aid]).reshape(-1).shape[0]) for aid in self.agent_ids]
        self.obs_dim = int(max(self.agent_obs_dims))

        self.act_dims = []
        self.agent_action_lows = []
        self.agent_action_highs = []
        for aid in self.agent_ids:
            space = env.action_space(aid)
            if hasattr(space, "shape") and space.shape is not None and int(np.prod(space.shape)) > 0:
                d = int(np.prod(space.shape))
                low = np.asarray(space.low, dtype=np.float32).reshape(-1)
                high = np.asarray(space.high, dtype=np.float32).reshape(-1)
            elif hasattr(space, "n"):
                d = int(space.n)
                low = np.zeros((d,), dtype=np.float32)
                high = np.ones((d,), dtype=np.float32)
            else:
                raise TypeError(f"Unsupported action space for agent {aid}: {space}")
            self.act_dims.append(d)
            self.agent_action_lows.append(low)
            self.agent_action_highs.append(high)
        env.close()

        if len(set(self.agent_obs_dims)) > 1:
            print(
                f"[MPECooperativeEnv] heterogeneous obs dims {self.agent_obs_dims}; "
                f"padding to {self.obs_dim} per agent."
            )

        self.n_i = self.obs_dim
        self.m_i = self.act_dims[0]
        self.joint_action_low = np.concatenate(self.agent_action_lows, axis=0).astype(np.float32)
        self.joint_action_high = np.concatenate(self.agent_action_highs, axis=0).astype(np.float32)
        action_high = float(np.max(np.abs(np.concatenate([self.joint_action_low, self.joint_action_high], axis=0))))

        n = self.n_agents * self.obs_dim
        m = int(sum(self.act_dims))
        names = [f"agent{i}.obs{j}" for i in range(self.n_agents) for j in range(self.obs_dim)]
        names += [f"agent{i}.act{j}" for i, d in enumerate(self.act_dims) for j in range(d)]
        super().__init__(
            dt=dt,
            n=n,
            m=m,
            act_rng=action_high,
            obs_trans=False,
            name=f"mpe-{self.mpe_env_key}",
            state_actions_names=names,
            device=device,
            solver=solver,
            obs_noise=obs_noise,
            ts_grid=ts_grid,
            ac_rew_const=ac_rew_const,
            vel_rew_const=0.0,
        )
        # Use true action bounds from MPE spaces (e.g. [0,1] for continuous MPE).
        self.ac_lb = torch.tensor(self.joint_action_low, dtype=torch.float32, device=device)
        self.ac_ub = torch.tensor(self.joint_action_high, dtype=torch.float32, device=device)
        self.N0 = 10
        self.Nexpseq = 2
        self.reward_range = [-10.0, 10.0]
        self.reset()

    def _make_env(self):
        kwargs = dict(self.mpe_kwargs)
        kwargs.update(
            {
                "max_cycles": int(self.max_cycles),
                "continuous_actions": bool(self.continuous_actions),
                "render_mode": None,
            }
        )
        if self.mpe_scenario == "simple_spread_v3":
            try:
                from mpe2 import simple_spread_v3
            except Exception:
                from pettingzoo.mpe import simple_spread_v3

            return simple_spread_v3.parallel_env(**kwargs)
        if self.mpe_scenario == "simple_tag_v3":
            try:
                from mpe2 import simple_tag_v3
            except Exception:
                from pettingzoo.mpe import simple_tag_v3

            return simple_tag_v3.parallel_env(**kwargs)
        raise ValueError(f"Unsupported MPE scenario: {self.mpe_scenario}")

    def _obs_dict_to_joint(self, obs_dict):
        parts = []
        for i, aid in enumerate(self.agent_ids):
            default_dim = self.agent_obs_dims[i]
            raw_obs = obs_dict.get(aid, np.zeros((default_dim,), dtype=np.float32))
            obs = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
            if obs.shape[0] < self.obs_dim:
                obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]), mode="constant")
            elif obs.shape[0] > self.obs_dim:
                obs = obs[: self.obs_dim]
            parts.append(obs)
        return np.concatenate(parts, axis=0)

    def _default_env_action(self, action_space):
        if hasattr(action_space, "shape") and action_space.shape is not None and int(np.prod(action_space.shape)) > 0:
            zeros = np.zeros(action_space.shape, dtype=np.float32)
            if hasattr(action_space, "low") and hasattr(action_space, "high"):
                zeros = np.clip(zeros, action_space.low, action_space.high)
            return zeros.astype(np.float32)
        if hasattr(action_space, "n"):
            return 0
        raise TypeError(f"Unsupported action space type: {action_space}")

    def _build_action_dict(self, joint_action: np.ndarray, env):
        joint_action = np.asarray(joint_action, dtype=np.float32).reshape(-1)
        if joint_action.shape[0] != self.m:
            raise RuntimeError(f"Joint action dim mismatch: expected {self.m}, got {joint_action.shape[0]}")
        clipped_joint = np.clip(joint_action, self.joint_action_low, self.joint_action_high).astype(np.float32)
        actions = {}
        active_agents = set(env.agents)
        offset = 0
        for i, aid in enumerate(self.agent_ids):
            d = self.act_dims[i]
            local_act = clipped_joint[offset : offset + d]
            offset += d
            if aid not in active_agents:
                continue
            space = env.action_space(aid)
            if hasattr(space, "n"):
                actions[aid] = int(np.argmax(local_act))
            else:
                actions[aid] = local_act
        for aid in env.agents:
            if aid not in actions:
                actions[aid] = self._default_env_action(env.action_space(aid))
        return actions, clipped_joint

    def _team_reward(self, rewards: dict) -> float:
        reward_keys = [aid for aid in self.reward_agent_ids if aid in rewards]
        if len(reward_keys) == 0:
            reward_keys = list(rewards.keys())
        if len(reward_keys) == 0:
            return 0.0
        reward_vec = np.asarray([rewards[k] for k in reward_keys], dtype=np.float32)
        if reward_vec.shape[0] == 1 or np.allclose(reward_vec, reward_vec[0], atol=1e-6):
            return float(reward_vec[0])
        return float(reward_vec.mean())

    def torch_transform_states(self, state):
        return state

    def obs2state(self, state):
        return state

    def reset(self):
        env = self._make_env()
        obs, _ = env.reset()
        self.state = self._obs_dict_to_joint(obs)
        env.close()
        return self.state.copy()

    def torch_rhs(self, state, action):
        # True rhs is unknown for black-box environment stepping.
        return torch.zeros_like(state)

    def diff_obs_reward_(self, s):
        sj = s.reshape(*s.shape[:-1], self.n_agents, self.obs_dim)
        centered = sj - sj.mean(dim=-2, keepdim=True)
        disagreement = (centered**2).mean(dim=(-1, -2))
        state_pen = 1e-3 * (sj**2).mean(dim=(-1, -2))
        return -self.consensus_weight * disagreement - state_pen

    def diff_ac_reward_(self, a):
        return -self.ac_rew_const * torch.sum(a**2, dim=-1)

    def integrate_system(self, T, g, s0=None, N=1, return_states=False, reset_seeds=None):
        with torch.no_grad():
            T = int(T)
            if s0 is not None:
                N = int(s0.shape[0])
            N = max(1, int(N))
            if reset_seeds is None:
                episode_seeds = [None] * int(N)
            else:
                episode_seeds = [int(seed) for seed in reset_seeds]
                if len(episode_seeds) != int(N):
                    raise ValueError(
                        f"reset_seeds length mismatch: expected {int(N)}, got {len(episode_seeds)}"
                    )
            ts_single = self.dt * torch.arange(T, dtype=torch.float32, device=self.device)

            envs = [self._make_env() for _ in range(N)]
            obs_dicts = []
            for i, env in enumerate(envs):
                seed_i = episode_seeds[i]
                if seed_i is None:
                    obs, _ = env.reset()
                else:
                    obs, _ = env.reset(seed=int(seed_i))
                obs_dicts.append(obs)
            alive = [True] * N
            ep_returns = [0.0] * N
            sts = []
            ats = []
            rts = []

            for t_idx in range(T):
                obs_batch_np = np.stack([self._obs_dict_to_joint(obs_dicts[i]) for i in range(N)], axis=0)
                obs_batch = torch.tensor(obs_batch_np, dtype=torch.float32, device=self.device)
                t_tensor = ts_single[t_idx]
                action_batch = g(obs_batch, t_tensor)
                if isinstance(action_batch, np.ndarray):
                    action_batch_np = np.asarray(action_batch, dtype=np.float32)
                else:
                    action_batch_np = action_batch.detach().cpu().numpy().astype(np.float32)
                if action_batch_np.ndim == 1:
                    action_batch_np = action_batch_np.reshape(1, -1)
                if action_batch_np.shape[0] != N:
                    if action_batch_np.shape[0] == 1:
                        action_batch_np = np.repeat(action_batch_np, N, axis=0)
                    else:
                        raise RuntimeError(
                            f"Policy batch dim mismatch: expected {N}, got {action_batch_np.shape[0]}"
                        )
                if action_batch_np.shape[1] != self.m:
                    raise RuntimeError(
                        f"Policy action dim mismatch: expected {self.m}, got {action_batch_np.shape[1]}"
                    )

                step_actions = []
                step_rewards = []
                for i, env in enumerate(envs):
                    if not alive[i]:
                        step_actions.append(torch.zeros(self.m, dtype=torch.float32))
                        step_rewards.append(ep_returns[i])
                        continue
                    env_actions, clipped_joint = self._build_action_dict(action_batch_np[i], env)
                    next_obs, rewards, terms, truncs, _ = env.step(env_actions)
                    team_reward = self._team_reward(rewards)
                    ep_returns[i] += float(team_reward)
                    step_rewards.append(ep_returns[i])
                    step_actions.append(torch.tensor(clipped_joint, dtype=torch.float32))
                    done = (
                        len(env.agents) == 0
                        or (len(terms) > 0 and all(terms.values()))
                        or (len(truncs) > 0 and all(truncs.values()))
                    )
                    if done:
                        alive[i] = False
                    else:
                        obs_dicts[i] = next_obs

                sts.append(torch.tensor(obs_batch_np, dtype=torch.float32))
                ats.append(torch.stack(step_actions))
                rts.append(torch.tensor(step_rewards, dtype=torch.float32))

            for env in envs:
                env.close()

            st = torch.stack(sts, dim=1).to(self.device)
            at = torch.stack(ats, dim=1).to(self.device)
            rt = torch.stack(rts, dim=1).to(self.device)
            ts = torch.stack([ts_single] * N)
            returns = [st, at, rt, ts]
            if return_states:
                returns.append(st)
            return returns

    def render(self, mode='human', **kwargs):
        return None


class CooperativeCoupledEnv(BaseEnv):
    """Cooperative multi-agent wrapper with state coupling.

    It lifts a single-agent ODE environment into a joint environment so the
    original ENODE pipeline can be reused unchanged.
    """

    def __init__(
        self,
        base_env_cls,
        n_agents: int,
        dt: float,
        device,
        obs_trans: bool,
        obs_noise: float,
        ts_grid: str,
        solver: str,
        coupling_strength: float = 0.2,
        consensus_weight: float = 0.1,
    ):
        self.n_agents = n_agents
        self.coupling_strength = coupling_strength
        self.consensus_weight = consensus_weight

        self.agent_envs: List[BaseEnv] = [
            base_env_cls(
                dt=dt,
                obs_trans=obs_trans,
                device=device,
                obs_noise=obs_noise,
                ts_grid=ts_grid,
                solver=solver,
            )
            for _ in range(n_agents)
        ]
        self.base_env = self.agent_envs[0]

        n_i = self.base_env.n
        m_i = self.base_env.m
        act_rng = float(self.base_env.act_rng)

        state_action_names = []
        for i in range(n_agents):
            state_action_names.extend([f"agent{i}.{name}" for name in self.base_env.state_actions_names[:-1]])
        state_action_names.extend([f"agent{i}.action" for i in range(n_agents)])

        super().__init__(
            dt=dt,
            n=n_agents * n_i,
            m=n_agents * m_i,
            act_rng=act_rng,
            obs_trans=obs_trans,
            name=f"coop-{n_agents}x-{self.base_env.name}",
            state_actions_names=state_action_names,
            device=device,
            solver=solver,
            obs_noise=obs_noise,
            ts_grid=ts_grid,
            ac_rew_const=self.base_env.ac_rew_const,
            vel_rew_const=self.base_env.vel_rew_const,
        )

        # Keep default data-collection behavior roughly proportional to team size.
        self.N0 = self.base_env.N0
        self.Nexpseq = self.base_env.Nexpseq
        self.n_i = n_i
        self.m_i = m_i
        self.reward_range = [
            self.base_env.reward_range[0] - consensus_weight,
            self.base_env.reward_range[1],
        ]

        self.reset()

    def _reshape_joint(self, tensor, per_agent_dim):
        *prefix, _ = tensor.shape
        return tensor.reshape(*prefix, self.n_agents, per_agent_dim)

    def torch_transform_states(self, state):
        s = self._reshape_joint(state, self.n_i)
        parts = [self.agent_envs[i].torch_transform_states(s[..., i, :]) for i in range(self.n_agents)]
        return torch.cat(parts, dim=-1)

    def obs2state(self, obs):
        s = self._reshape_joint(obs, self.n_i)
        parts = [self.agent_envs[i].obs2state(s[..., i, :]) for i in range(self.n_agents)]
        return torch.cat(parts, dim=-1)

    def reset(self):
        states = [torch.tensor(agent.reset(), dtype=torch.float32) for agent in self.agent_envs]
        self.state = torch.cat(states, dim=0).cpu().numpy()
        return self.state.copy()

    def torch_rhs(self, state, action):
        s = self._reshape_joint(state, self.n_i)
        a = self._reshape_joint(action, self.m_i)

        ds_local = [self.agent_envs[i].torch_rhs(s[..., i, :], a[..., i, :]) for i in range(self.n_agents)]
        ds_local = torch.stack(ds_local, dim=-2)

        # Diffusive state coupling: each agent is softly attracted to team mean state.
        mean_state = s.mean(dim=-2, keepdim=True)
        ds_coupled = ds_local + self.coupling_strength * (mean_state - s)

        return ds_coupled.reshape(*state.shape[:-1], self.n)

    def diff_obs_reward_(self, s):
        sj = self._reshape_joint(s, self.n_i)
        per_agent = [self.agent_envs[i].diff_obs_reward_(sj[..., i, :]) for i in range(self.n_agents)]
        per_agent = torch.stack(per_agent, dim=-1)
        team_reward = per_agent.mean(dim=-1)

        # Cooperative term: penalize disagreement in states (state consensus).
        centered = sj - sj.mean(dim=-2, keepdim=True)
        disagreement = (centered**2).mean(dim=(-1, -2))
        return team_reward - self.consensus_weight * disagreement

    def diff_ac_reward_(self, a):
        aj = self._reshape_joint(a, self.m_i)
        per_agent = [self.agent_envs[i].diff_ac_reward_(aj[..., i, :]) for i in range(self.n_agents)]
        per_agent = torch.stack(per_agent, dim=-1)
        return per_agent.mean(dim=-1)

    def render(self, mode='human', **kwargs):
        # For debugging: render only the first agent env.
        return self.agent_envs[0].render(mode=mode, **kwargs)


def build_args():
    p = argparse.ArgumentParser(description='Cooperative multi-agent ENODE runner (state-coupled).')
    p.add_argument(
        '--env',
        default='cartpole',
        choices=['pendulum', 'cartpole', 'acrobot', *MAMUJOCO_ENV_SPECS.keys(), *MPE_ENV_ALIASES.keys()],
    )
    p.add_argument('--n_agents', type=int, default=3)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--noise', type=float, default=0.0)
    p.add_argument('--ts_grid', default='fixed', choices=['fixed', 'uniform', 'exp'])
    p.add_argument('--solver', default='rk4')
    p.add_argument('--rounds', type=int, default=50)
    p.add_argument('--episodes', type=int, default=None, help='Approximate number of episodes to train.')
    p.add_argument('--h_train', type=float, default=2.0)
    p.add_argument('--h_data', type=float, default=5.0)
    p.add_argument('--L', type=int, default=30)
    p.add_argument('--n_ens', type=int, default=5)
    p.add_argument('--coupling_strength', type=float, default=0.2)
    p.add_argument('--consensus_weight', type=float, default=0.1)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--obs_trans', action='store_true', help='Use transformed observations if base env supports it.')
    p.add_argument('--episode_length', type=int, default=50)
    p.add_argument('--replay_buffer_size', type=int, default=104)
    p.add_argument('--dyn_max_episodes', type=int, default=200, help='Hard cap on episodes used to train dynamics.')
    p.add_argument('--discount_rho', type=float, default=0.95)
    p.add_argument('--soft_update_tau', type=float, default=0.001)
    p.add_argument('--actor_lr', type=float, default=1e-4)
    p.add_argument('--critic_lr', type=float, default=1e-3)
    p.add_argument('--dyn_lr', type=float, default=1e-3)
    p.add_argument('--dyn_grad_clip', type=float, default=10.0, help='Max grad norm for dynamics model; <=0 disables clipping.')
    p.add_argument('--dyn_batch_size', type=int, default=1024, help='Mini-batch size for each dynamics gradient step.')
    p.add_argument('--dyn_update_steps', type=int, default=1000, help='Number of gradient descent steps per dynamics update.')
    p.add_argument('--dyn_window_steps', type=int, default=5, help='Raw (non-interpolated) steps per sampled dynamics chain.')
    p.add_argument('--rew_lr', type=float, default=1e-3)
    p.add_argument('--policy_batch_size', type=int, default=256)
    p.add_argument('--critic_updates', type=int, default=4)
    p.add_argument('--exploration_steps', type=int, default=1000)
    p.add_argument('--dyn_nrep', type=int, default=3, help='Parallel trajectory replicates used in each dynamics step.')
    p.add_argument('--collect_parallel_workers', type=int, default=1, help='Parallel workers for episode collection.')
    p.add_argument('--torch_num_threads', type=int, default=1, help='PyTorch intra-op threads.')
    p.add_argument('--torch_num_interop_threads', type=int, default=1, help='PyTorch inter-op threads.')
    p.add_argument('--model_save_interval', type=int, default=1000)
    p.add_argument(
        '--new_episodes_per_round',
        type=int,
        default=None,
        help='Number of new trajectories to add per round. If set, overrides env.Nexpseq with value-1.',
    )
    p.add_argument('--seed', type=int, default=111)
    p.add_argument('--use_wandb', action='store_true')
    p.add_argument('--wandb_project', type=str, default='ma-ctrl')
    p.add_argument('--wandb_entity', type=str, default='')
    p.add_argument('--wandb_group', type=str, default='mamujoco-enode')
    p.add_argument('--wandb_name', type=str, default='')
    p.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    p.add_argument('--wandb_tags', type=str, default='enode,ctde,multi-agent')
    return p.parse_args()


def main():
    args = build_args()
    if args.torch_num_threads is not None and args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    if args.torch_num_interop_threads is not None and args.torch_num_interop_threads > 0:
        torch.set_num_interop_threads(int(args.torch_num_interop_threads))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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
        )
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
    else:
        env_map = {
            'pendulum': envs.CTPendulum,
            'cartpole': envs.CTCartpole,
            'acrobot': envs.CTAcrobot,
        }
        env_cls = env_map[args.env]
        env = CooperativeCoupledEnv(
            base_env_cls=env_cls,
            n_agents=args.n_agents,
            dt=args.dt,
            device=device,
            obs_trans=args.obs_trans,
            obs_noise=args.noise,
            ts_grid=args.ts_grid,
            solver=args.solver,
            coupling_strength=args.coupling_strength,
            consensus_weight=args.consensus_weight,
        )
        env.use_env_rewards = False

    effective_agents = int(getattr(env, 'n_agents', args.n_agents))

    # Use a fixed episode duration across training/eval and snap to integer steps.
    fixed_episode_steps = max(1, int(np.round(FIXED_EPISODE_SECONDS / args.dt)))
    horizon = fixed_episode_steps * args.dt
    requested_horizon = args.episode_length * args.dt
    if not np.isclose(requested_horizon, horizon, rtol=0.0, atol=1e-9):
        print(
            f"Overriding requested episode length ({requested_horizon:.4f}s) "
            f"with fixed {horizon:.4f}s ({fixed_episode_steps} steps)."
        )
    args.episode_length = fixed_episode_steps
    args.h_data = horizon
    args.h_train = horizon

    # Align initial exploration budget in environment steps.
    env.N0 = max(1, int(np.ceil(args.exploration_steps / fixed_episode_steps)))
    if args.new_episodes_per_round is not None:
        if args.new_episodes_per_round < 1:
            raise ValueError('--new_episodes_per_round must be >= 1')
        env.Nexpseq = int(args.new_episodes_per_round) - 1

    D = utils.collect_data(env, H=args.h_data, N=env.N0)

    ctrl = base.CTRL(
        env,
        dynamics='enode',
        n_ens=args.n_ens,
        learn_sigma=False,
        nl_f=3,
        nn_f=200,
        act_f='elu',
        dropout_f=0.05,
        nl_g=2,
        nn_g=200,
        act_g='relu',
        nl_V=2,
        nn_V=200,
        act_V='tanh',
    ).to(device)
    # Use user-selected ODE solver for model rollouts; adaptive solvers can OOM in long backprop traces.
    ctrl.set_solver(args.solver)
    ctde_multi_controller = False
    ctde_obs_dims = []
    ctde_act_dims = []
    if effective_agents >= 1:
        joint_policy, agent_ctrls, ctde_obs_dims, ctde_act_dims = build_ctde_multi_controller_policy(
            env=env,
            n_agents=effective_agents,
            device=device,
            nl_g=ctrl.kwargs['nl_g'],
            nn_g=ctrl.kwargs['nn_g'],
            act_g=ctrl.kwargs['act_g'],
        )
        ctrl._g = joint_policy
        ctrl = ctrl.to(device)
        ctde_multi_controller = True
        print(
            f"CTDE decentralized actors enabled: controllers={len(agent_ctrls)} "
            f"obs_dims={ctde_obs_dims} act_dims={ctde_act_dims}"
        )

    print(
        f'Env={env.name}, agents={effective_agents}, dt={env.dt:.3f}, '
        f'noise={env.obs_noise:.3f}, ts_grid={env.ts_grid}, device={device}'
    )

    if args.episodes is not None:
        rounds = max(1, int(np.ceil((args.episodes - env.N0) / (env.Nexpseq + 1))))
    else:
        rounds = args.rounds
    # Convert "save every K steps" to "save every K episodes" approximation.
    model_save_interval_rounds = max(1, int(np.ceil(args.model_save_interval / (env.Nexpseq + 1))))
    # Continuous discount conversion: gamma(t)=exp(-t/tau), so tau = -dt/log(rho)
    policy_tau = float(-args.dt / np.log(max(min(args.discount_rho, 1.0 - 1e-8), 1e-8)))

    run_name = (
        f"{ctrl.name}-ma{effective_agents}-ctde{int(ctde_multi_controller)}-k{args.coupling_strength:.2f}-"
        f"cw{args.consensus_weight:.2f}-seed{args.seed}"
    )
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = ROOT / "checkpoint"
    run_output_prefix = _make_indexed_output_prefix(run_name)
    eval_metrics_path = run_output_prefix + "-eval.jsonl"
    print(f"output_prefix={run_output_prefix}")
    print(f"eval_metrics_path={eval_metrics_path}")
    print(f"best_checkpoint_dir={checkpoint_dir}")
    runtime_device_info = {
        'runtime_device': str(device),
        'runtime_device_type': str(device.type),
        'runtime_cuda_available': bool(torch.cuda.is_available()),
        'runtime_cuda_device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        'runtime_cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
    }
    if device.type == 'cuda' and torch.cuda.is_available():
        cuda_idx = int(device.index) if device.index is not None else 0
        runtime_device_info['runtime_cuda_device_index'] = cuda_idx
        runtime_device_info['runtime_cuda_device_name'] = torch.cuda.get_device_name(cuda_idx)
    print(
        'Runtime device info: '
        f"device={runtime_device_info['runtime_device']} "
        f"type={runtime_device_info['runtime_device_type']} "
        f"cuda_available={runtime_device_info['runtime_cuda_available']} "
        f"cuda_count={runtime_device_info['runtime_cuda_device_count']} "
        f"cuda_visible_devices='{runtime_device_info['runtime_cuda_visible_devices']}' "
        f"cuda_name={runtime_device_info.get('runtime_cuda_device_name', 'n/a')}"
    )
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
        except ImportError as e:
            raise ImportError("wandb is not installed. Install with `pip install wandb`.") from e
        wandb_tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]
        wandb_kwargs = {
            'project': args.wandb_project,
            'name': args.wandb_name if args.wandb_name else run_name,
            'group': args.wandb_group if args.wandb_group else None,
            'mode': args.wandb_mode,
            'config': vars(args),
            'tags': wandb_tags,
        }
        if args.wandb_entity:
            wandb_kwargs['entity'] = args.wandb_entity
        wandb_run = wandb.init(**wandb_kwargs)
        wandb.define_metric('train/round')
        wandb.define_metric('train/*', step_metric='train/round')
        wandb.define_metric('train/dataset/*', step_metric='train/round')
        wandb.define_metric('train/new_data/*', step_metric='train/round')
        wandb.define_metric('train/time_*', step_metric='train/round')
        wandb.define_metric('policy/global_step')
        wandb.define_metric('policy/*', step_metric='policy/global_step')
        wandb.define_metric('dynamics/global_step')
        wandb.define_metric('dynamics/*', step_metric='dynamics/global_step')
        wandb.define_metric('eval/round')
        wandb.define_metric('eval/*', step_metric='eval/round')
        wandb.define_metric('eval/test_reward_*', step_metric='eval/round')
        wandb_run.config.update(
            {
                'effective_agents': effective_agents,
                'env_name': env.name,
                'run_name': run_name,
                'ctde_multi_controller': int(ctde_multi_controller),
                'ctde_obs_dims': ctde_obs_dims,
                'ctde_act_dims': ctde_act_dims,
                **runtime_device_info,
            },
            allow_val_change=True,
        )
        for key, value in runtime_device_info.items():
            wandb_run.summary[key] = value
        print(f"wandb enabled: project={args.wandb_project}, run={wandb_run.name}")
    print(f"Training rounds={rounds} (target episodes={args.episodes})")
    print(
        f"seed={args.seed} ep_len={args.episode_length} rho={args.discount_rho} "
        f"actor_lr={args.actor_lr} critic_lr={args.critic_lr} dyn_lr={args.dyn_lr} "
        f"dyn_grad_clip={args.dyn_grad_clip} dyn_batch_size={args.dyn_batch_size} "
        f"dyn_update_steps={args.dyn_update_steps} dyn_window_steps={args.dyn_window_steps} "
        f"dyn_max_episodes={args.dyn_max_episodes} "
        f"explore_steps={args.exploration_steps} "
        f"collect_workers={args.collect_parallel_workers} dyn_nrep={args.dyn_nrep} "
        f"new_eps_per_round={env.Nexpseq + 1} init_eps={env.N0} "
        f"save_steps={args.model_save_interval}"
    )
    if args.rew_lr != args.dyn_lr:
        print("Note: reward model is not separated in this codebase; rew_lr is logged but not used.")

    try:
        utils.plot_model(ctrl, D, L=args.L, H=args.h_train, rep_buf=min(10, D.N), fname=run_output_prefix + '-train.png')
        utils.plot_test(ctrl, D, L=args.L, H=FIXED_EPISODE_SECONDS, N=min(5, max(3, D.N)), fname=run_output_prefix + '-test.png')
        utils.train_loop(
            ctrl, D, run_output_prefix, rounds, L=args.L, H=args.h_train,
            policy_tau=policy_tau, actor_lr=args.actor_lr,
            critic_lr=args.critic_lr, soft_tau=args.soft_update_tau, dyn_lr=args.dyn_lr,
            dyn_grad_clip=args.dyn_grad_clip,
            dyn_batch_size=args.dyn_batch_size,
            dyn_update_steps=args.dyn_update_steps,
            dyn_window_steps=args.dyn_window_steps,
            dyn_max_episodes=args.dyn_max_episodes,
            dyn_nrep=args.dyn_nrep,
            dyn_rep_buf=args.replay_buffer_size, model_save_interval_rounds=model_save_interval_rounds,
            dyn_save_every=model_save_interval_rounds, policy_save_every=model_save_interval_rounds,
            use_env_rewards=getattr(env, 'use_env_rewards', False),
            policy_batch_size=args.policy_batch_size, critic_updates=args.critic_updates,
            eval_horizon_sec=FIXED_EPISODE_SECONDS,
            eval_metrics_path=eval_metrics_path,
            save_best_checkpoint=True,
            checkpoint_dir=str(checkpoint_dir),
            run_timestamp=run_timestamp,
            checkpoint_algo=ctrl.dynamics,
            checkpoint_env_name=env.name,
            seed=args.seed,
            wandb_run=wandb_run
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    main()
