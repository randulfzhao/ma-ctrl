import torch, copy, numpy as np, time
from typing import Optional

from utils.utils import KernelInterpolation, K

class Dataset:
    def __init__(self, env, D, ts):
        D = Dataset.__compute_rewards_if_needed(env,D) # N,T,n+m+1
        self.env = env
        self.n   = env.n
        self.m   = env.m
        self.D   = D
        self.ts  = ts
        # Cache of fixed-length raw windows + precomputed interpolation factors.
        self._window_cache = {}

    def _invalidate_window_cache(self):
        self._window_cache = {}

    @property
    def device(self):
        return self.env.device

    @property
    def shape(self):
        return self.D.shape

    @property
    def dt(self):
        return self.env.dt

    @property
    def N(self):
        return self.shape[0]

    @property
    def T(self):
        return self.shape[1]
    
    @property
    def s(self):
        return self.D[:,:,:self.n]

    @property
    def a(self):
        return self.D[:,:,self.n:self.n+self.m]

    @property
    def sa(self):
        return self.D[:,:,:self.n+self.m]

    @property
    def r(self):
        return self.D[:,:,-1:]

    def clone(self):
        return copy.deepcopy(self)

    def add_experience(self, Dnew, ts):
        assert (len(Dnew.shape)==3), 'New experience must be a 3D torch tensor' # N,T,nm
        Dnew = self.__compute_rewards_if_needed(self.env,Dnew) # N,T,n+m+1
        Dnew,ts = Dnew.to(self.device),ts.to(self.device)
        self.D  = torch.cat([self.D,Dnew])
        self.ts = torch.cat([self.ts,ts])
        self._invalidate_window_cache()
            
    def crop_last(self,N=1):
        self.D  = self.D[:-N]
        self.ts = self.ts[:-N]
        self._invalidate_window_cache()

    def keep_last(self, N):
        N = int(max(0, N))
        if N == 0:
            self.D = self.D[:0]
            self.ts = self.ts[:0]
            self._invalidate_window_cache()
            return
        if self.N <= N:
            return
        self.D = self.D[-N:]
        self.ts = self.ts[-N:]
        self._invalidate_window_cache()

    @staticmethod
    def __compute_rewards_if_needed(env,D):
        ''' returns (s,a,r) '''
        assert (len(D.shape)==3), 'Dataset must be a 3D torch tensor' # N,T,nm
        if D.shape[-1]==env.n+env.m:
            [N,T,nm] = D.shape
            with torch.no_grad():
                s_ = D[:,:,:env.n].view([-1,env.n])
                a_ = D[:,:,env.n:].view([-1,env.m])
                rewards = env.diff_reward(s_,a_).view([N,T,1])
                D = torch.cat([D,rewards],2) # N,T,n+m+1
        return D
    
    def to(self,device):
        self.D = self.D.to(device)
        self.ts = self.ts.to(device)
        self._invalidate_window_cache()
        return self
    
    def prepare_parallel_window_cache(self, H, cont, sf=1.0, ell=0.5, eps=1e-5, force=False):
        """Build fixed-length raw windows and interpolation factors in parallel.

        The cache is built from *raw* trajectory steps only. Interpolation is
        precomputed once (batched linear solves) so train-time sampling does not
        rebuild interpolation operators.
        """
        T = int(H/self.dt)
        if T < 1:
            raise ValueError(f'Window length must be >=1, got H={H}.')
        if T > self.T:
            raise ValueError(f'Window length {T} exceeds trajectory length {self.T}.')

        key = (int(T), bool(cont), float(sf), float(ell), float(eps))
        if (not force) and (key in self._window_cache):
            return self._window_cache[key]

        # Raw, non-interpolated windows: [Nseq, Nstart, T, n+m+1]
        full_windows = self.D.unfold(1, T, 1).permute(0, 1, 3, 2).contiguous()
        ts_windows = self.ts.unfold(1, T, 1).contiguous()  # [Nseq, Nstart, T]
        nseq, nstart = full_windows.shape[0], full_windows.shape[1]

        windows = full_windows.reshape(-1, T, full_windows.shape[-1]).contiguous()
        ts_flat = ts_windows.reshape(-1, T).contiguous()
        st_flat = windows[:, :, :self.n].contiguous()
        at_flat = windows[:, :, self.n:self.n+self.m].contiguous()
        rt_flat = windows[:, :, -1:].contiguous()

        seq_idx = torch.arange(nseq, device=self.device, dtype=torch.long).unsqueeze(1).expand(nseq, nstart).reshape(-1)
        start_idx = torch.arange(nstart, device=self.device, dtype=torch.long).unsqueeze(0).expand(nseq, nstart).reshape(-1)

        cache = {
            'T': int(T),
            'num_starts': int(nstart),
            'st': st_flat,
            'at': at_flat,
            'rt': rt_flat,
            'ts': ts_flat,
            'seq_idx': seq_idx,
            'start_idx': start_idx,
            'sf': float(sf),
            'ell': float(ell),
            'eps': float(eps),
        }

        if cont:
            # Batched interpolation precompute for all windows in parallel.
            X = ts_flat.unsqueeze(-1).to(torch.float32)  # [Nw, T, 1]
            sf_t = torch.tensor(float(sf), device=X.device, dtype=X.dtype)
            ell_t = torch.tensor(float(ell), device=X.device, dtype=X.dtype)
            K_xx = K(X, X, ell=ell_t, sf=sf_t, eps=float(eps))  # [Nw, T, T]

            # Action interpolation factors used by forward simulation.
            action_alpha = torch.linalg.solve(K_xx, at_flat.to(X.dtype)).to(at_flat.dtype)
            # State interpolation factors are also cached for downstream use/debugging.
            state_alpha = torch.linalg.solve(K_xx, st_flat.to(X.dtype)).to(st_flat.dtype)
            cache['action_alpha'] = action_alpha.contiguous()
            cache['state_alpha'] = state_alpha.contiguous()

        self._window_cache[key] = cache
        return cache

    def extract_data(
        self,
        H,
        cont,
        nrep=1,
        idx=None,
        use_window_cache=False,
        cache_sf=1.0,
        cache_ell=0.5,
        cache_eps=1e-5,
    ):
        ''' extracts sequences randomly subsequenced from the dataset
                H  - in second
                cont - boolean denoting whether the system is continuous
            returns
                g  - policy or None
                st - [N,T,n]
                at - [N,T,m]
                rt - [N,T,1]
        '''
        idx = np.arange(0, self.N, dtype=np.int64) if idx is None else np.asarray(list(idx), dtype=np.int64)
        if idx.size == 0:
            raise ValueError('No sequence indices were provided for extract_data.')
        nrep = int(max(1, nrep))
        idx = np.tile(idx, nrep)
        # Support Python-style negative indexing used by plotting utilities.
        idx = np.where(idx < 0, idx + self.N, idx)
        if idx.min() < 0 or idx.max() >= self.N:
            raise IndexError('Sampled sequence index is out of range.')

        T = int(H/self.dt) # convert sec to # data points
        if T > self.T:
            raise ValueError(f'Requested window length {T} exceeds trajectory length {self.T}.')

        if use_window_cache and cont:
            cache = self.prepare_parallel_window_cache(
                H=H, cont=cont, sf=cache_sf, ell=cache_ell, eps=cache_eps
            )
            starts = np.random.randint(0, cache['num_starts'], size=len(idx)).astype(np.int64, copy=False)
            win_idx_np = idx * cache['num_starts'] + starts
            win_idx = torch.as_tensor(win_idx_np, dtype=torch.long, device=self.device)

            st = cache['st'][win_idx]
            at = cache['at'][win_idx]
            rt = cache['rt'][win_idx]
            ts = cache['ts'][win_idx]
            g = self.__extract_policy(
                idx,
                at=at,
                cont=cont,
                ts=ts,
                T=T,
                action_alpha=cache.get('action_alpha', None)[win_idx],
                state_alpha=cache.get('state_alpha', None)[win_idx],
                sf=cache['sf'],
                ell=cache['ell'],
                eps=cache['eps'],
            )
            return g, st, at, rt, ts

        t0s = torch.tensor(np.random.randint(0,1+self.T-T,len(idx)),dtype=torch.int64).to(self.device)
        idx_list = idx.tolist()
        st_at_rt = torch.stack([self.D[seq_idx_,t0:t0+T] for t0,seq_idx_ in zip(t0s,idx_list)])
        st,at,rt = st_at_rt[:,:,:self.n], st_at_rt[:,:,self.n:self.n+self.m], st_at_rt[:,:,-1:]
        ts = torch.stack([self.ts[seq_idx_,t0:t0+T] for t0,seq_idx_ in zip(t0s,idx_list)])
        g = self.__extract_policy(idx_list, at=at, cont=cont, ts=ts, T=T)
        return g, st, at, rt, ts
    
    def __extract_policy(
        self,
        idx,
        at,
        cont,
        ts,
        T,
        action_alpha: Optional[torch.Tensor] = None,
        state_alpha: Optional[torch.Tensor] = None,
        sf: float = 1.0,
        ell: float = 0.5,
        eps: float = 1e-5,
    ):
        if cont:
            if action_alpha is not None:
                return CachedKernelInterpolatePolicy(
                    ts=ts,
                    action_alpha=action_alpha,
                    state_alpha=state_alpha,
                    sf=sf,
                    ell=ell,
                    eps=eps,
                )
            return KernelInterpolatePolicy(at, ts, sf=sf, ell=ell, eps=eps)
        return DiscreteActions(at, ts)

class DiscreteActions:
    def __init__(self, at, ts):
        if len(at.shape) != 3:
            raise ValueError('Actions must be 3D!\n')
        self.at = at.to(at.device) # N,T,m
        self.ts = ts.to(at.device) # N,T
        self.N  = self.ts.shape[0]
        self.max_idx  = self.at.shape[1]-1
        self._call_time_accum = 0.0
        self._timing_enabled = False

    def reset_timing(self):
        self._call_time_accum = 0.0
        self._timing_enabled = True

    def get_timing(self):
        return float(self._call_time_accum)

    def __call__(self,s,t):
        t0 = time.perf_counter() if self._timing_enabled else None
        # t = t.item() if isinstance(t,torch.Tensor) else t
        if t[0].item()>self.ts[0,-1].item(): # actions outside the defined range
            actions = self.at[:,-1]
        else:
            before_idx = [(t[i]+1e-5>self.ts[i]).sum().item()-1 for i in range(self.N)]
            before_idx = [min(item,self.max_idx) for item in before_idx]
            actions = self.at[np.arange(self.N),before_idx]
        if actions.isnan().sum()>0:
            raise ValueError('Action interpolation is wrong!')
        if s.ndim==2:
            out = actions
        elif s.ndim==3:
            out = torch.stack([actions]*s.shape[0])
        elif s.ndim==4:
            tmp = torch.stack([actions]*s.shape[1])
            out = torch.stack([tmp]*s.shape[0])
        else:
            raise ValueError(f'Unsupported state rank for actions: {s.ndim}')
        if self._timing_enabled:
            self._call_time_accum += (time.perf_counter() - t0)
        return out

class CachedKernelInterpolatePolicy:
    """Interpolation policy with precomputed batched kernel solves."""

    def __init__(self, ts, action_alpha, state_alpha=None, sf=1.0, ell=0.5, eps=1e-5):
        if ts.ndim != 2:
            raise ValueError(f'ts must be 2D [N,T], got {tuple(ts.shape)}')
        if action_alpha.ndim != 3:
            raise ValueError(f'action_alpha must be 3D [N,T,m], got {tuple(action_alpha.shape)}')
        self.N = int(ts.shape[0])
        self.X = ts.unsqueeze(-1).to(torch.float32)  # [N,T,1]
        self.action_alpha = action_alpha.to(torch.float32)
        self.state_alpha = None if state_alpha is None else state_alpha.to(torch.float32)
        self.sf = torch.tensor(float(sf), device=self.X.device, dtype=self.X.dtype)
        self.ell = torch.tensor(float(ell), device=self.X.device, dtype=self.X.dtype)
        self.eps = float(eps)
        self._call_time_accum = 0.0
        self._timing_enabled = False

    def reset_timing(self):
        self._call_time_accum = 0.0
        self._timing_enabled = True

    def get_timing(self):
        return float(self._call_time_accum)

    def _format_query_times(self, t):
        if isinstance(t, torch.Tensor):
            t_ = t.to(device=self.X.device, dtype=self.X.dtype).reshape(-1)
        else:
            t_ = torch.tensor([t], device=self.X.device, dtype=self.X.dtype)
        if t_.numel() == 1 and self.N > 1:
            t_ = t_.repeat(self.N)
        if t_.numel() != self.N:
            raise ValueError(f'Query times must have {self.N} elements, got {t_.numel()}.')
        return t_.unsqueeze(-1).unsqueeze(-1)  # [N,1,1]

    def _interpolate(self, alpha, t):
        q = self._format_query_times(t)
        kxX = K(q, self.X, ell=self.ell, sf=self.sf, eps=self.eps)  # [N,1,T]
        return kxX @ alpha  # [N,1,d]

    def interpolate_states(self, t):
        if self.state_alpha is None:
            raise RuntimeError('State interpolation factors are not available.')
        return self._interpolate(self.state_alpha, t)

    def __call__(self, s, t):
        t0 = time.perf_counter() if self._timing_enabled else None
        actions = self._interpolate(self.action_alpha, t)  # [N,1,m]
        actions = actions.permute(1, 0, 2)  # [1,N,m]
        if s.ndim == 2:
            out = actions
        elif s.ndim == 3:
            out = torch.cat([actions] * s.shape[0])
        elif s.ndim == 4:
            tmp = torch.stack([actions] * s.shape[1])
            out = torch.stack([tmp] * s.shape[0])
        else:
            raise ValueError(f'Unsupported state rank for actions: {s.ndim}')
        if self._timing_enabled:
            self._call_time_accum += (time.perf_counter() - t0)
        return out

class KernelInterpolatePolicy:
    def __init__(self, at, ts, sf=1.0, ell=0.5, eps=1e-5):
        [N,T,m] = at.shape
        sfs  = float(sf) * torch.ones([N,1,1],device=at.device, dtype=torch.float32)
        ells = float(ell) * torch.ones([N,1,1],device=at.device, dtype=torch.float32)
        self.kernel_int = KernelInterpolation(sfs, ells, ts.unsqueeze(-1), at, eps=float(eps))
        self._call_time_accum = 0.0
        self._timing_enabled = False

    def reset_timing(self):
        self._call_time_accum = 0.0
        self._timing_enabled = True

    def get_timing(self):
        return float(self._call_time_accum)
    
    def __call__(self,s,t):
        t0 = time.perf_counter() if self._timing_enabled else None
        actions = self.kernel_int(t.unsqueeze(-1).unsqueeze(-1)) # N,1,n_out
        actions = actions.permute(1,0,2) # 1,N,n_out
        if s.ndim==2:
            out = actions
        elif s.ndim==3:
            out = torch.cat([actions]*s.shape[0])
        elif s.ndim==4:
            tmp = torch.stack([actions]*s.shape[1])
            out = torch.stack([tmp]*s.shape[0])
        else:
            raise ValueError(f'Unsupported state rank for actions: {s.ndim}')
        if self._timing_enabled:
            self._call_time_accum += (time.perf_counter() - t0)
        return out
