import os
import argparse
from typing import List
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
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
from envs.base_env import BaseEnv

FIXED_EPISODE_SECONDS = 2.5


class MaMuJoCoAnt2x4Env(BaseEnv):
    """Ant 2x4 adapter so the ENODE pipeline can train on MaMuJoCo rollouts."""

    def __init__(
        self,
        dt: float,
        device,
        obs_noise: float,
        ts_grid: str,
        solver: str,
        consensus_weight: float = 0.0,
        ac_rew_const: float = 0.01,
        num_env_workers: int = 1,
    ):
        try:
            from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1 as mamujoco
        except Exception as e:
            raise ImportError(
                "MaMuJoCo not available. Please run with conda env 'mbrl' (gymnasium_robotics installed)."
            ) from e

        self.mamujoco = mamujoco
        self.scenario = "Ant"
        self.agent_conf = "2x4"
        self.agent_obsk = 1
        self.consensus_weight = consensus_weight
        self.num_env_workers = max(1, int(num_env_workers))
        self.use_env_rewards = True
        self.rewards_are_accumulated = True
        self.use_solved_threshold = False

        env = self._make_env()
        obs, _ = env.reset(seed=0)
        self.agent_ids = list(env.agents)
        self.n_agents = len(self.agent_ids)
        self.obs_dim = int(np.asarray(obs[self.agent_ids[0]]).shape[0])
        self.act_dims = [int(np.prod(env.action_space(a).shape)) for a in self.agent_ids]
        self.m_i = self.act_dims[0]
        self.n_i = self.obs_dim
        action_high = float(np.max(np.abs(env.action_space(self.agent_ids[0]).high)))
        self.agent_action_lows = [np.asarray(env.action_space(a).low, dtype=np.float32) for a in self.agent_ids]
        self.agent_action_highs = [np.asarray(env.action_space(a).high, dtype=np.float32) for a in self.agent_ids]
        env.close()

        n = self.n_agents * self.obs_dim
        m = int(sum(self.act_dims))
        names = [f"agent{i}.obs{j}" for i in range(self.n_agents) for j in range(self.obs_dim)]
        names += [f"agent{i}.act{j}" for i in range(self.n_agents) for j in range(self.m_i)]
        super().__init__(
            dt=dt,
            n=n,
            m=m,
            act_rng=action_high,
            obs_trans=False,
            name="mamujoco-ant2x4",
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
        return self.mamujoco.parallel_env(
            scenario=self.scenario,
            agent_conf=self.agent_conf,
            agent_obsk=self.agent_obsk,
            render_mode=None,
            terminate_when_unhealthy=False,
        )

    def _obs_dict_to_joint(self, obs_dict):
        return np.concatenate([np.asarray(obs_dict[a], dtype=np.float32) for a in self.agent_ids], axis=0)

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

    def integrate_system(self, T, g, s0=None, N=1, return_states=False):
        with torch.no_grad():
            if s0 is not None:
                N = int(s0.shape[0])
            ts_single = self.dt * torch.arange(T, dtype=torch.float32, device=self.device)
            sts, ats, rts = [], [], []

            def _rollout_one_episode():
                env = self._make_env()
                obs, _ = env.reset()
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
                    st_ep.append(joint_obs.squeeze(0).cpu())
                    at_ep.append(torch.tensor(joint_action, dtype=torch.float32))
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
                rollouts = [_rollout_one_episode() for _ in range(N)]
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(_rollout_one_episode) for _ in range(N)]
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
    p.add_argument('--env', default='cartpole', choices=['pendulum', 'cartpole', 'acrobot', 'ant2x4'])
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
    p.add_argument('--wandb_group', type=str, default='ant2x4-enode')
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

    if args.env == 'ant2x4':
        env = MaMuJoCoAnt2x4Env(
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
        f"{ctrl.name}-ma{effective_agents}-k{args.coupling_strength:.2f}-"
        f"cw{args.consensus_weight:.2f}-seed{args.seed}"
    )
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
        utils.plot_model(ctrl, D, L=args.L, H=args.h_train, rep_buf=min(10, D.N), fname=run_name + '-train.png')
        utils.plot_test(ctrl, D, L=args.L, H=FIXED_EPISODE_SECONDS, N=min(5, max(3, D.N)), fname=run_name + '-test.png')
        utils.train_loop(
            ctrl, D, run_name, rounds, L=args.L, H=args.h_train,
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
            wandb_run=wandb_run
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    main()
