import os
import argparse
from typing import List

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_default_dtype(torch.float32)

import envs
import ctrl.ctrl as base
from ctrl import utils
from envs.base_env import BaseEnv


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
    p.add_argument('--env', default='cartpole', choices=['pendulum', 'cartpole', 'acrobot'])
    p.add_argument('--n_agents', type=int, default=3)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--noise', type=float, default=0.0)
    p.add_argument('--ts_grid', default='fixed', choices=['fixed', 'uniform', 'exp'])
    p.add_argument('--solver', default='dopri5')
    p.add_argument('--rounds', type=int, default=50)
    p.add_argument('--h_train', type=float, default=2.0)
    p.add_argument('--h_data', type=float, default=5.0)
    p.add_argument('--L', type=int, default=30)
    p.add_argument('--n_ens', type=int, default=5)
    p.add_argument('--dyn_grad_clip', type=float, default=10.0, help='Max grad norm for dynamics model; <=0 disables clipping.')
    p.add_argument('--dyn_max_episodes', type=int, default=200, help='Hard cap on episodes used to train dynamics.')
    p.add_argument('--coupling_strength', type=float, default=0.2)
    p.add_argument('--consensus_weight', type=float, default=0.1)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--obs_trans', action='store_true', help='Use transformed observations if base env supports it.')
    return p.parse_args()


def main():
    args = build_args()
    device = torch.device(args.device)

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

    print(
        f'Env={env.name}, agents={args.n_agents}, dt={env.dt:.3f}, '
        f'noise={env.obs_noise:.3f}, ts_grid={env.ts_grid}, device={device}'
    )

    run_name = (
        f"{ctrl.name}-ma{args.n_agents}-k{args.coupling_strength:.2f}-"
        f"cw{args.consensus_weight:.2f}"
    )

    utils.plot_model(ctrl, D, L=args.L, H=args.h_train, rep_buf=min(10, D.N), fname=run_name + '-train.png')
    utils.plot_test(ctrl, D, L=args.L, H=max(args.h_train, 2.5), N=min(5, max(3, D.N)), fname=run_name + '-test.png')
    utils.train_loop(
        ctrl, D, run_name, args.rounds, L=args.L, H=args.h_train,
        dyn_grad_clip=args.dyn_grad_clip, dyn_max_episodes=args.dyn_max_episodes
    )


if __name__ == '__main__':
    main()
