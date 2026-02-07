
import torch
import torch.nn as nn
from utils import BNN

tanh_ = torch.nn.Tanh()
def final_activation(env,a):
    if getattr(env, "ac_lb", None) is None or getattr(env, "ac_ub", None) is None:
        return tanh_(a) * env.act_rng
    ac_lb = env.ac_lb.to(a.device, dtype=a.dtype)
    ac_ub = env.ac_ub.to(a.device, dtype=a.dtype)
    act_mid = (ac_lb + ac_ub) / 2.0
    act_amp = (ac_ub - ac_lb) / 2.0
    while act_mid.ndim < a.ndim:
        act_mid = act_mid.unsqueeze(0)
        act_amp = act_amp.unsqueeze(0)
    return act_mid + tanh_(a) * act_amp

class Policy(nn.Module):
    def __init__(self, env, nl=2, nn=100, act='relu'):
        super().__init__()
        self.env   = env
        self.act   = act
        self.multi_agent_decentralized = (
            hasattr(env, "n_agents") and hasattr(env, "n_i") and hasattr(env, "m_i") and
            int(env.n) == int(env.n_agents) * int(env.n_i) and
            int(env.m) == int(env.n_agents) * int(env.m_i)
        )
        if self.multi_agent_decentralized:
            self.n_agents = int(env.n_agents)
            self.n_i = int(env.n_i)
            self.m_i = int(env.m_i)
            # Shared actor for decentralized execution:
            # each agent action depends only on its own local observation.
            self._g = BNN(self.n_i, self.m_i, n_hid_layers=nl, act=act, n_hidden=nn, dropout=0.0, bnn=False)
        else:
            self._g = BNN(env.n, env.m, n_hid_layers=nl, act=act, n_hidden=nn, dropout=0.0, bnn=False)
        self.reset_parameters()
    
    def reset_parameters(self,w=0.1):
        self._g.reset_parameters(w)
    
    def forward(self,s,t):
        if self.multi_agent_decentralized:
            sj = s.reshape(*s.shape[:-1], self.n_agents, self.n_i)
            aj = self._g(sj)
            a = aj.reshape(*s.shape[:-1], self.env.m)
        else:
            a = self._g(s)
        return final_activation(self.env, a)
