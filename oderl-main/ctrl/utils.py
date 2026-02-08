import numpy as np
import copy, math, os, collections, time
import matplotlib.pyplot as plt
plt.switch_backend('agg')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch

from utils.utils import K, KernelInterpolation, numpy_to_torch, flatten_
from ctrl.dataset import Dataset
# from gpytorch.utils.cholesky import psd_safe_cholesky


def _wandb_log(wandb_run, metrics, step=None):
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
    if len(clean) == 0:
        return
    if step is None:
        wandb_run.log(clean)
    else:
        wandb_run.log(clean, step=step)


def _tensor_stats(prefix, tensor, quantiles=(0.05, 0.5, 0.95), max_elems=200000):
    if tensor is None:
        return {}
    flat = tensor.detach().to(torch.float32).reshape(-1)
    if flat.numel() == 0:
        return {}
    if max_elems is not None and flat.numel() > int(max_elems):
        step = int(np.ceil(flat.numel() / float(max_elems)))
        flat = flat[::step]
    stats = {
        f'{prefix}_mean': flat.mean(),
        f'{prefix}_std': flat.std(unbiased=False) if flat.numel() > 1 else flat.new_zeros(()),
        f'{prefix}_min': flat.min(),
        f'{prefix}_max': flat.max(),
    }
    if flat.numel() > 1 and quantiles is not None and len(quantiles) > 0:
        q_values = torch.quantile(flat, torch.tensor(list(quantiles), device=flat.device, dtype=flat.dtype))
        for q, val in zip(quantiles, q_values):
            stats[f'{prefix}_p{int(round(100 * q)):02d}'] = val
    return stats


def _step_rewards_from_dataset_rewards(rewards, rewards_are_accumulated):
    if rewards is None or rewards.numel() == 0:
        return rewards
    if not rewards_are_accumulated:
        return rewards
    step_rewards = rewards.clone()
    if rewards.shape[1] > 1:
        step_rewards[:, 1:] = rewards[:, 1:] - rewards[:, :-1]
    return step_rewards


def _dataset_summary_metrics(D, prefix='train/dataset', start_idx=0):
    start_idx = int(max(0, start_idx))
    if start_idx >= D.N:
        return {}

    s = D.s[start_idx:]
    a = D.a[start_idx:]
    r = D.r[start_idx:, :, 0]

    metrics = {
        f'{prefix}/num_sequences': int(s.shape[0]),
        f'{prefix}/horizon_steps': int(D.T),
        f'{prefix}/horizon_sec': float(D.T * D.dt),
    }
    metrics.update(_tensor_stats(f'{prefix}/reward', r))

    rewards_are_accumulated = getattr(D.env, 'rewards_are_accumulated', False)
    step_rewards = _step_rewards_from_dataset_rewards(r, rewards_are_accumulated)
    metrics.update(_tensor_stats(f'{prefix}/step_reward', step_rewards))

    episode_returns = r[:, -1] if rewards_are_accumulated else step_rewards.sum(dim=1)
    metrics.update(_tensor_stats(f'{prefix}/episode_return', episode_returns))

    metrics.update(_tensor_stats(f'{prefix}/abs_state', s.abs()))
    metrics.update(_tensor_stats(f'{prefix}/abs_action', a.abs()))
    metrics.update(_tensor_stats(f'{prefix}/state_l2', torch.linalg.norm(s, dim=-1)))
    metrics.update(_tensor_stats(f'{prefix}/action_l2', torch.linalg.norm(a, dim=-1)))

    ac_lb = getattr(D.env, 'ac_lb', None)
    ac_ub = getattr(D.env, 'ac_ub', None)
    if ac_lb is not None and ac_ub is not None:
        lb = torch.as_tensor(ac_lb, device=a.device, dtype=a.dtype).view(1, 1, -1)
        ub = torch.as_tensor(ac_ub, device=a.device, dtype=a.dtype).view(1, 1, -1)
        sat_tol = 1e-3
        saturation = ((a <= lb + sat_tol) | (a >= ub - sat_tol)).float().mean()
        metrics[f'{prefix}/action_saturation_ratio'] = saturation
    return metrics

##########################################################################################
######################################## PLOTTING ########################################
##########################################################################################
def plot_model(ctrl, D, rep_buf=10, H=None, L=10, fname=None, verbose=False, savefig=True):
    with torch.no_grad():
        if fname is None:
            fname = '{:s}-train.png'.format(ctrl.name)
        if verbose: 
            print('fname is {:s}'.format(fname))
        H = D.H if H is None else H
        rep_buf = min(rep_buf,D.N)
        idxs = -1*torch.arange(rep_buf,0,-1)
        g,st,at,rt,tobs = D.extract_data(H, ctrl.is_cont, idx=idxs)
        st_hat, rt_hat, at_hat, t = \
            ctrl.forward_simulate(tobs, st[:,0,:], g, L=L, compute_rew=False, record_actions=True)
        if verbose: 
            print(st_hat.shape)
        plot_sequences(ctrl, fname, tobs, st, at, rt, t, st_hat, at_hat, rt_hat, verbose=verbose, savefig=savefig)

def plot_test(ctrl, D, N=None, H=None, L=10, fname=None, verbose=False, savefig=True):
    with torch.no_grad():
        if fname is None:
            fname = '{:s}-test.png'.format(ctrl.name)
        if verbose: 
            print('fname is {:s}'.format(fname))
        H = D.H if H is None else H
        N = max(D.N,10) if N is None else N
        D = collect_test_sequences(ctrl, D, N=N, reset=False, explore=True)
        g,st,at,rt,tobs = D.extract_data(H, ctrl.is_cont)
        st_hat, rt_hat, at_hat, t = \
            ctrl.forward_simulate(H, st[:,0,:], g, L=L, compute_rew=False, record_actions=True)
        if verbose: 
            print(st_hat.shape)
        plot_sequences(ctrl, fname, tobs, st, at, rt, t, st_hat, at_hat, rt_hat, verbose=verbose, savefig=savefig)

def plot_sequences(ctrl, fname, tobs, st, at, rt, t, st_hat, at_hat, rt_hat, verbose=False, savefig=True):
    plt.close()
    [L,N,T,_] = st_hat.shape
    V = ctrl.V(st_hat).squeeze(-1).cpu().detach().numpy() # L,N,Td
    # rt_hat = ctrl.env.diff_reward(st_hat[:,:,:-1],at_hat)
    t      = t.cpu().detach().numpy() # T
    st_hat = st_hat.cpu().detach().numpy() # L,N,T,n
    rt_hat = rt_hat.cpu().detach().numpy() # L,N,T,m
    st_np = st.cpu().numpy()
    delta = st_hat - st_np
    finite_mask = np.isfinite(delta)
    if not finite_mask.all():
        bad = int((~finite_mask).sum())
        total = int(finite_mask.size)
        print(f'[plot_sequences] WARNING: non-finite state prediction entries: {bad}/{total}')
    # Bound before squaring to avoid overflow in diagnostic plots.
    delta = np.clip(np.nan_to_num(delta, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    err = (delta**2).mean(0).mean(0).mean(1)
    t_acts = torch.stack(list(at_hat.keys())).T
    act_hats = torch.stack(list(at_hat.values())).permute(1,2,0,3)
    t_acts   = t_acts.cpu().detach().numpy() # T
    tobs     = tobs.cpu().detach().numpy() # T
    act_hats = act_hats.cpu().detach().numpy() # T
    if verbose: 
        print(f'average error is {err}')
    n_plot = min(ctrl.env.n, 10)
    m_plot = min(ctrl.env.m, 5)
    w = n_plot + m_plot + 2
    plt.figure(1,((n_plot+m_plot)*5,N*3))
    for j in range(N):
        for i in range(n_plot):
            plt.subplot(N,w,j*w+i+1)
            plt.plot(t[j], st_hat[:,j,:,i].T, '-b',linewidth=.75)
            if i==0:
                plt.ylabel('Seq. {:d}'.format(j+1),fontsize=20)
            plt.plot(tobs[j], st[j,:,i].cpu().numpy(), '.r',linewidth=2,markersize=10)
            if j==0:
                plt.title(ctrl.env.state_actions_names[i],fontsize=25)    
            # rang = (st[j,:,i].max() - st[j,:,i].min()).item()
            # plt.ylim([st[j,:,i].min().item()-rang/5, st[j,:,i].max().item()+rang/5])
        for i in range(m_plot):
            plt.subplot(N,w,j*w+n_plot+i+1)
            plt.plot(tobs[j], at[j,:,i].cpu().numpy(), '.r',linewidth=1.0)
            plt.plot(t_acts[j], act_hats[:,j,:,i].T,'-b',linewidth=.75)
            if j==0:
                plt.title(ctrl.env.state_actions_names[ctrl.env.n+i],fontsize=25)
            plt.ylim([ctrl.env.ac_lb[i].item()-0.1,ctrl.env.ac_ub[i].item()+0.1])
        # plot reward
        plt.subplot(N,w,j*w+n_plot+m_plot+1) 
        # plt.plot(t[j,:-1], rt_hat[:,j].T,'-b')
        plt.plot(tobs[j], rt[j].cpu().numpy(),'or',markersize=4)
        if j==0:
            plt.title('rewards',fontsize=25)
        plt.ylim([ctrl.env.reward_range[0]-0.2,ctrl.env.reward_range[1]+0.2])
        # plot value
        plt.subplot(N,w,j*w+n_plot+m_plot+2) 
        # min,max = V[:,j].min().item(),V[:,j].max().item()
        # scaled_rew = (max-min)*rt[j]+min
        # plt.plot(t, scaled_rew.cpu().numpy(),'-or',markersize=3,linewidth=0.5)
        plt.plot(t[j], V[:,j].T,'-b')
        if j==0:
            plt.title('Values',fontsize=25)
        plt.grid(True)
    plt.tight_layout()
    if savefig:
        plt.savefig(fname)
    plt.close()        
    

##########################################################################################
######################################## TRAIN #########################################
##########################################################################################

def get_train_functions(dynamics):
    if 'ode' in dynamics:
        return train_ode
    elif 'pilco' in dynamics:
        return train_deep_pilco
    elif 'pets' in dynamics:
        return train_pets

def train_loop(ctrl, D, fname, Nround, **kwargs):
    H, L  = kwargs['H'], kwargs['L']
    train_kwargs = dict(kwargs)
    verbose = kwargs.get('verbose', True)
    print_times = kwargs.get('print_times', 10)
    N_pol_iter = kwargs.get('N_pol_iter', 250)
    Nexpseq = kwargs.get('Nexpseq', ctrl.env.Nexpseq)
    policy_tau = kwargs.get('policy_tau', 5.0)
    actor_lr = kwargs.get('actor_lr', 1e-3)
    critic_lr = kwargs.get('critic_lr', 1e-3)
    soft_tau = kwargs.get('soft_tau', None)
    model_save_interval_rounds = kwargs.get('model_save_interval_rounds', 1)
    dyn_lr = kwargs.get('dyn_lr', 1e-3)
    dyn_rep_buf = kwargs.get('dyn_rep_buf', 50)
    dyn_save_every = kwargs.get('dyn_save_every', 50)
    policy_save_every = kwargs.get('policy_save_every', 50)
    use_env_rewards = kwargs.get('use_env_rewards', getattr(ctrl.env, 'use_env_rewards', False))
    policy_batch_size = kwargs.get('policy_batch_size', 256)
    critic_updates = kwargs.get('critic_updates', 4)
    wandb_run = kwargs.get('wandb_run', None)
    eval_horizon_sec = float(kwargs.get('eval_horizon_sec', 2.5))
    dyn_tr_fnc = get_train_functions(ctrl.dynamics)
    print('fname is {:s}'.format(fname))
    # plot_model(ctrl, D, H=H, L=L, rep_buf=10, fname=fname+'-train.png')
    r0 = int( (D.N-ctrl.env.N0) / (1+ctrl.env.Nexpseq) )
    for round in range(r0,r0+Nround):
        print(f'Round {round}/{r0+Nround} starting')
        round_start_time = time.perf_counter()
        round_data_size_before = int(D.N)
        _wandb_log(
            wandb_run,
            {
                'train/round': round,
                'train/dataset_size': D.N,
                'train/use_env_rewards': int(use_env_rewards),
                'train/model_based_phase': int(not use_env_rewards),
            },
        )
        _wandb_log(
            wandb_run,
            {
                'train/round': round,
                **_dataset_summary_metrics(D, prefix='train/dataset'),
            },
        )
        # dynamics training
        dynamics_t0 = time.perf_counter()
        train_kwargs['dyn_lr'] = dyn_lr
        train_kwargs['dyn_rep_buf'] = dyn_rep_buf
        train_kwargs['dyn_save_every'] = dyn_save_every
        train_kwargs['wandb_run'] = wandb_run
        train_kwargs['round_idx'] = round
        dyn_tr_fnc(ctrl, D, fname, verbose, print_times, **train_kwargs)
        dynamics_dt = time.perf_counter() - dynamics_t0
        _wandb_log(
            wandb_run,
            {
                'train/round': round,
                'train/time_dynamics_sec': dynamics_dt,
            },
        )
        # policy learning
        policy_t0 = time.perf_counter()
        train_policy(ctrl, D, H=H, V_const=min(round/5.0,1), verbose=verbose, \
            Niter=N_pol_iter, L=L, save_fname=fname, print_times=print_times, tau=policy_tau, \
            actor_eta=actor_lr, value_eta=critic_lr, soft_tau=soft_tau, save_every=policy_save_every, \
            use_env_rewards=use_env_rewards, batch_size=policy_batch_size, critic_updates=critic_updates, \
            wandb_run=wandb_run, round_idx=round)
        policy_dt = time.perf_counter() - policy_t0
        _wandb_log(
            wandb_run,
            {
                'train/round': round,
                'train/time_policy_sec': policy_dt,
            },
        )
        # test the policy
        eval_t0 = time.perf_counter()
        Htest, Ntest = eval_horizon_sec, 10
        Ttest = max(1, int(np.round(Htest / ctrl.env.dt)))
        Tup = 0
        s0 = torch.stack([numpy_to_torch(ctrl.env.reset()) for _ in range(Ntest)]).to(ctrl.device) 
        _,_,test_rewards,_ = ctrl.env.integrate_system(T=Ttest, s0=s0, g=ctrl._g)
        # Evaluate over a fixed episode window.
        rewards_are_accumulated = getattr(ctrl.env, 'rewards_are_accumulated', False)
        if rewards_are_accumulated:
            true_test_rewards = test_rewards[..., -1].mean().item()
            min_test_reward = test_rewards[..., -1].min().item()
        else:
            true_test_rewards = test_rewards[...,Tup:].mean().item()
            min_test_reward = test_rewards[...,Tup:].min().item()
        st_hat, rt_hat, at_hat, t = ctrl.forward_simulate(Htest, s0, ctrl._g, compute_rew=True, record_actions=True)
        rt_hat = ctrl.env.diff_obs_reward_(st_hat[:,:,:-1])
        imagined_test_rewards = rt_hat[..., Tup:].mean().item()
        imagined_min_reward = rt_hat[..., Tup:].min().item()
        reward_gap = true_test_rewards - imagined_test_rewards
        print('Model tested. True/imagined reward sum is {:.2f}/{:.2f}'.\
              format(true_test_rewards,imagined_test_rewards))
        print(f'Minimum test reward is {min_test_reward}')
        eval_dt = time.perf_counter() - eval_t0
        solved_tensor = test_rewards[..., -1] if rewards_are_accumulated else test_rewards[...,Tup:]
        solved_ratio = (solved_tensor >= .8).float().mean().item()
        _wandb_log(
            wandb_run,
            {
                'eval/round': round,
                'eval/true_reward': true_test_rewards,
                'eval/model_reward': imagined_test_rewards,
                'eval/model_min_reward': imagined_min_reward,
                'eval/min_reward': min_test_reward,
                'eval/reward_gap': reward_gap,
                'eval/abs_reward_gap': abs(reward_gap),
                'eval/solved_ratio': solved_ratio,
                'eval/time_sec': eval_dt,
                **_tensor_stats('eval/test_reward', test_rewards),
            },
        )
        use_solved_threshold = getattr(ctrl.env, 'use_solved_threshold', True)
        solved = False
        if use_solved_threshold:
            solved = (solved_tensor < .8).sum() == 0
        if solved:
            print(f'All {Ntest} tests are solved in {round} rounds!')
            ctrl.save(D=D,fname=fname+'-solved')
            _wandb_log(
                wandb_run,
                {
                    'train/round': round,
                    'train/solved': 1,
                    'train/time_round_total_sec': time.perf_counter() - round_start_time,
                },
            )
            break
        print('Collecting experience...\n')
        mean_act = torch.stack(list(at_hat.values())).abs().mean()
        print('Imagined mean action is {:.2f}'.format(mean_act))
        _wandb_log(
            wandb_run,
            {
                'train/mean_abs_action': mean_act,
                'train/round': round,
            },
        )
        data_t0 = time.perf_counter()
        experience_mode = 0
        if mean_act<0.25: # if the policy does not explore, collect data with random policy
            D = collect_data(ctrl.env, H=D.dt*D.T, N=Nexpseq+1, D=D)
            experience_mode = 0
        else:
            D = collect_experience(ctrl, D=D, N=Nexpseq, H=D.dt*D.T, reset=False, explore=True) 
            D = collect_experience(ctrl, D=D, N=1, H=D.dt*D.T, reset=True)
            experience_mode = 1
        if dyn_rep_buf > 0 and D.N > int(dyn_rep_buf):
            D.keep_last(int(dyn_rep_buf))
        data_dt = time.perf_counter() - data_t0
        new_sequences = max(0, int(D.N) - round_data_size_before)
        train_metrics = {
            'train/round': round,
            'train/experience_mode': int(experience_mode),  # 0=random fallback, 1=policy-driven
            'train/new_sequences': new_sequences,
            'train/dataset_size': D.N,
            'train/time_data_collection_sec': data_dt,
            'train/time_round_total_sec': time.perf_counter() - round_start_time,
            'train/solved': 0,
        }
        train_metrics.update(_dataset_summary_metrics(D, prefix='train/dataset'))
        if new_sequences > 0:
            train_metrics.update(_dataset_summary_metrics(D, prefix='train/new_data', start_idx=round_data_size_before))
        _wandb_log(wandb_run, train_metrics)
        plot_model(ctrl, D, H=H, L=10, rep_buf=10, fname=fname+'-train.png', verbose=False)
        if model_save_interval_rounds > 0 and ((round - r0 + 1) % model_save_interval_rounds == 0):
            ctrl.save(D=D, fname=fname)
            _wandb_log(
                wandb_run,
                {
                    'train/checkpoint_saved': 1,
                    'train/round': round,
                },
            )

def _sample_transition_batch(D, batch_size):
    if D.T < 2:
        raise ValueError('Dataset trajectory length must be >=2 for TD learning.')
    batch_size = min(int(batch_size), D.N * (D.T - 1))
    batch_size = max(1, batch_size)
    seq_idx = torch.randint(0, D.N, (batch_size,), device=D.device)
    t_idx = torch.randint(0, D.T-1, (batch_size,), device=D.device)
    s = D.s[seq_idx, t_idx]
    s_next = D.s[seq_idx, t_idx+1]
    r = D.r[seq_idx, t_idx, 0]
    if getattr(D.env, 'rewards_are_accumulated', False):
        prev_t_idx = torch.clamp(t_idx - 1, min=0)
        r_prev = D.r[seq_idx, prev_t_idx, 0]
        r = torch.where(t_idx == 0, r, r - r_prev)
    return s, s_next, r

def train_policy(ctrl, D, H=2.0, Niter=250, verbose=True, tau=5.0, N=100, L=10, V_const=1.0, save_every=50, 
            eta=1e-3, actor_eta=None, value_eta=None, soft_tau=None, save_fname=None, rep_buf=5, opt='adam', print_times=10,
            use_env_rewards=False, batch_size=256, critic_updates=4, wandb_run=None, round_idx=None):
    if rep_buf<0:
        rep_buf = D.N
    if verbose: 
        print('policy training started')
    if save_fname is None:
        save_fname = ctrl.name
    N = D.N if N==-1 else N
    opt_cls = get_opt(opt)
    if actor_eta is None:
        actor_eta = eta
    if value_eta is None:
        value_eta = eta
    policy_optimizer = opt_cls(ctrl._g.parameters(),lr=actor_eta)
    opt_cls = get_opt(opt)
    value_optimizer = opt_cls(ctrl.V.parameters(),lr=value_eta)
    L = ctrl.get_L(L)
    rewards,opt_objs = [],[]
    Vtarget = copy.deepcopy(ctrl.V)
    if use_env_rewards:
        gamma = float(np.exp(-ctrl.env.dt/max(tau,1e-6)))
        for itr in range(Niter):
            if soft_tau is None and itr%100==0:
                Vtarget = copy.deepcopy(ctrl.V)
            elif soft_tau is not None:
                with torch.no_grad():
                    for p_targ, p in zip(Vtarget.parameters(), ctrl.V.parameters()):
                        p_targ.data.mul_(1.0-soft_tau).add_(soft_tau*p.data)

            mean_val_err = 0.0
            mean_reward = 0.0
            first_val_err = 0.0
            for inner_iter in range(max(1, int(critic_updates))):
                s_b, s_next_b, r_b = _sample_transition_batch(D, batch_size)
                with torch.no_grad():
                    Vtargets = r_b + gamma * Vtarget(s_next_b).squeeze(-1)
                value_optimizer.zero_grad()
                td_error = ctrl.V(s_b).squeeze(-1) - Vtargets
                value_loss = torch.mean(td_error**2)
                value_loss.backward()
                value_optimizer.step()
                mean_val_err += value_loss.item() / max(1, int(critic_updates))
                mean_reward += r_b.mean().item() / max(1, int(critic_updates))
                if inner_iter == 0:
                    first_val_err = value_loss.item()

            s_actor, _, _ = _sample_transition_batch(D, batch_size)
            L_actor = ctrl.get_L(1)
            noise_vec = ctrl.draw_noise(L_actor)
            fs = ctrl.draw_f(L_actor, noise_vec=noise_vec)
            policy_optimizer.zero_grad()
            f_params = list(ctrl._f.parameters())
            V_params = list(ctrl.V.parameters())
            for p in f_params:
                p.requires_grad_(False)
            for p in V_params:
                p.requires_grad_(False)
            if L_actor == 1:
                s_in = s_actor
            else:
                s_in = torch.stack([s_actor]*L_actor)
            t_actor = torch.zeros(s_actor.shape[0], dtype=torch.float32, device=s_actor.device)
            a_in = ctrl._g(s_in, t_actor)
            ds = ctrl._f.ds_dt(fs, s_in, a_in)
            s_next_pred = s_in + ctrl.env.dt * ds
            pred_values = ctrl.V(s_next_pred).squeeze(-1)
            mean_cost = -pred_values.mean()
            mean_cost.backward()
            grad_parts = [p.grad for p in ctrl._g.parameters() if p.grad is not None]
            grad_norm = torch.norm(flatten_(grad_parts)).item() if len(grad_parts) > 0 else 0.0
            policy_optimizer.step()
            for p in f_params:
                p.requires_grad_(True)
            for p in V_params:
                p.requires_grad_(True)

            rewards.append(mean_reward)
            opt_objs.append(mean_cost.item())
            print_log = 'Iter:{:4d}/{:<4d},  opt. target:{:.3f}  mean reward:{:.3f}  '\
                .format(itr, Niter, np.mean(opt_objs), np.mean(rewards)) + \
                'H={:.2f},  grad_norm={:.3f},  '.format(H,grad_norm)
            print_log += 'first/final value error:{:.3f}/{:.3f}  kl:{:.3f}'\
                .format(first_val_err, mean_val_err, ctrl.V.kl().item())
            current_actor_lr = float(policy_optimizer.param_groups[0]['lr'])
            current_critic_lr = float(value_optimizer.param_groups[0]['lr'])
            _wandb_log(
                wandb_run,
                {
                    'policy/round': 0 if round_idx is None else int(round_idx),
                    'policy/global_step': (0 if round_idx is None else int(round_idx)) * int(Niter) + int(itr),
                    'policy/iter': itr,
                    'policy/opt_target': np.mean(opt_objs),
                    'policy/opt_target_cur': mean_cost.item(),
                    'policy/mean_reward': np.mean(rewards),
                    'policy/mean_reward_cur': mean_reward,
                    'policy/grad_norm': grad_norm,
                    'policy/mean_abs_action_cur': a_in.abs().mean().item(),
                    'policy/pred_value_mean_cur': pred_values.mean().item(),
                    'policy/value_target_mean_cur': Vtargets.mean().item(),
                    'policy/value_error_first': first_val_err,
                    'policy/value_error_final': mean_val_err,
                    'policy/value_error_cur': mean_val_err,
                    'policy/kl': ctrl.V.kl().item(),
                    'policy/actor_lr': current_actor_lr,
                    'policy/critic_lr': current_critic_lr,
                },
            )
            if verbose and itr % max(1, Niter//print_times) == 0:
                print(print_log)
            if (itr+1)%save_every == 0:
                ctrl.save(D, fname=save_fname)
        return

    for itr in range(Niter):
        # update the critic copy
        if soft_tau is None and itr%100==0:
            Vtarget = copy.deepcopy(ctrl.V)
        elif soft_tau is not None:
            with torch.no_grad():
                for p_targ, p in zip(Vtarget.parameters(), ctrl.V.parameters()):
                    p_targ.data.mul_(1.0-soft_tau).add_(soft_tau*p.data)
        s0 = get_ivs(ctrl.env,D,N,rep_buf) # N,n
        noise_vec = ctrl.draw_noise(L)
        fs = ctrl.draw_f(L, noise_vec=noise_vec)
        policy_optimizer.zero_grad()
        st, rt, at, ts = ctrl.forward_simulate(
            H, s0, ctrl._g, f=fs, L=L, tau=tau, compute_rew=True, record_actions=False
        )
        rew_int  = rt[:,:,-1].mean(0) # N
        if rt.isnan().any():
            print('Reward is nan. Breaking.')
            break
        ts = ts[0]
        st = torch.cat([st]*5) if st.shape[0]==1 else st
        [L,N_,Hdense,n] = st.shape
        gammas = (-ts/tau).exp() # H
        V_st_gam = ctrl.V(st.contiguous())[:,:,1:,0] * gammas[1:] # L,N,H-1
        n_step_returns = rt[:,:,1:] + V_const*V_st_gam # ---> n_step_returns[:,:,k] is the sum in (5)
        opimized_returns = n_step_returns.mean(-1) # L,N
        mean_cost = -opimized_returns.mean()
        mean_cost.backward()
        grad_norm = torch.norm(flatten_([p.grad for p in ctrl._g.parameters()])).item()
        policy_optimizer.step()
        rewards.append(rew_int.mean().item()/H)
        opt_objs.append(mean_cost.mean().item())
        print_log = 'Iter:{:4d}/{:<4d},  opt. target:{:.3f}  mean reward:{:.3f}  '\
            .format(itr, Niter, np.mean(opt_objs), np.mean(rewards)) + \
            'H={:.2f},  grad_norm={:.3f},  '.format(H,grad_norm) 
        # minimize TD error
        with torch.no_grad():
            # regress all intermediate values
            last_states = st.detach().contiguous()[:,:,1:,:] # L,N,T-1,n
            last_values = Vtarget(last_states).squeeze(-1) # L,N,T-1
            Vtargets = rt[:,:,1:] + (-ts[1:]/tau).exp()*last_values # L,N,T-1
            Vtargets = Vtargets.mean(0).mean(-1) # N
        mean_val_err = 0
        for inner_iter in range(10):
            value_optimizer.zero_grad()
            td_error = ctrl.V(s0).squeeze(-1) - Vtargets # L,N
            td_error = torch.mean(td_error**2)
            td_error.backward()
            mean_val_err += td_error.item() / 10
            if inner_iter==0:
                first_val_err = td_error.item()
            value_optimizer.step()
        print_log += 'first/final value error:{:.3f}/{:.3f}  kl:{:.3f}'\
            .format(first_val_err,td_error.item(),ctrl.V.kl().item())
        current_actor_lr = float(policy_optimizer.param_groups[0]['lr'])
        current_critic_lr = float(value_optimizer.param_groups[0]['lr'])
        mean_abs_action_cur = torch.stack(list(at.values())).abs().mean().item() if len(at) > 0 else 0.0
        _wandb_log(
            wandb_run,
            {
                'policy/round': 0 if round_idx is None else int(round_idx),
                'policy/global_step': (0 if round_idx is None else int(round_idx)) * int(Niter) + int(itr),
                'policy/iter': itr,
                'policy/opt_target': np.mean(opt_objs),
                'policy/opt_target_cur': mean_cost.mean().item(),
                'policy/mean_reward': np.mean(rewards),
                'policy/mean_reward_cur': rew_int.mean().item()/H,
                'policy/grad_norm': grad_norm,
                'policy/mean_abs_action_cur': mean_abs_action_cur,
                'policy/imagined_return_mean_cur': rew_int.mean().item(),
                'policy/imagined_return_std_cur': rew_int.std(unbiased=False).item(),
                'policy/value_target_mean_cur': Vtargets.mean().item(),
                'policy/value_error_first': first_val_err,
                'policy/value_error_final': td_error.item(),
                'policy/value_error_cur': td_error.item(),
                'policy/kl': ctrl.V.kl().item(),
                'policy/actor_lr': current_actor_lr,
                'policy/critic_lr': current_critic_lr,
            },
        )
        if verbose and itr % max(1, Niter//print_times) == 0:
            print(print_log)
        if (itr+1)%save_every == 0:
            ctrl.save(D, fname=save_fname)
    
    
def dynamics_loss(ctrl, st, ts, at, g, L=1):
    f = ctrl.draw_f(L)
    outputs = ctrl.forward_simulate(ts, st[:,0,:], g, f=f, L=L, compute_rew=False, record_actions=False)
    st_hat,at_hat = outputs[0], outputs[2]
    [N,T,n] = st.shape 
    [L,N,T,_] = st_hat.shape
    sq_err = (torch.stack([st]*L)-st_hat)**2  # L,N,T,n
    sq_err = sq_err.view([-1,ctrl.env.n]) / ctrl.sn[:n]**2 / 2
    lhood = -sq_err - torch.mean(ctrl.logsn[:n]) - 0.5*np.log(2*np.pi)
    lhood = lhood.sum() / L
    mse = sq_err.mean().item()
    return mse, lhood, st_hat, at_hat  # N,T,n
def train_dynamics(ctrl, D, Niter=1000, verbose=True, H=10, N=-1, L=1, eta=1e-3, eta_final=2e-4, \
        save_every=50, save_fname=None, func_KL=False, lr_sch=False, kl_w=1, rep_buf=-1, temp_opt=True, \
        num_plots=0, opt='adam', print_times=10, rnode=False, stop_mse=1e-3, nrep=3, \
        wandb_run=None, round_idx=None, grad_clip=10.0):
    if save_fname is None:
        save_fname = ctrl.name
    opt_cls = get_opt(opt, temp=temp_opt)
    losses, mses, lhoods, kls, grad_norms = [], [], [], [], []
    opt_pars = ctrl.dynamics_parameters
    optimizer = opt_cls(opt_pars,lr=eta)
    if verbose: 
        print('dynamics training started')
    if verbose: 
        print(f'Dataset size = {list(D.shape)}')
    num_below_thresholds = 0
    for k in range(Niter):
        idx_ = np.arange(D.N)[-rep_buf:] if rep_buf>0 else  np.arange(D.N)
        g,st,at,rt,tobs = D.extract_data(H=H, idx=idx_, nrep=nrep, cont=ctrl.is_cont)
        optimizer.zero_grad()
        mse, sum_lhood, st_hat, at_hat = dynamics_loss(ctrl, st, tobs, at, g, L=L) # lhood = N,T,n
        loss   = -sum_lhood * D.T / (H/ctrl.env.dt) / nrep
        lhood_ = sum_lhood.item()
        kl_ = 0.0
        if kl_w > 0:
            kl_w_ = kl_w*min(1,(2*k/Niter))
            kl = kl_w_ * ctrl._f.kl().sum()
            loss += kl
            kl_ = kl.item()
        loss.backward()
        if math.isnan(loss.item()):
            print('Dynamics loss is nan, no gradient computation.')
            break
        grad_tensors = [p.grad for p in opt_pars if p is not None and p.grad is not None]
        if len(grad_tensors) == 0:
            print('Dynamics gradients are empty; skipping optimizer step.')
            continue
        grad_norm_pre_ = torch.norm(flatten_(grad_tensors)).item()
        grad_norm_ = grad_norm_pre_
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(opt_pars, max_norm=float(grad_clip), error_if_nonfinite=False)
            grad_norm_ = torch.norm(flatten_(grad_tensors)).item()
        if not np.isfinite(grad_norm_):
            print(f'Dynamics grad norm is non-finite ({grad_norm_}); skipping optimizer step.')
            optimizer.zero_grad()
            continue
        optimizer.step()
        losses.append(loss.item())
        mses.append(mse)
        lhoods.append(lhood_)
        kls.append(kl_)
        grad_norms.append(grad_norm_)
        if verbose and k % max(1, Niter//print_times) == 0:
            print('Iter:{:4d}/{:<4d} loss:{:<.3f} lhood:{:<.3f} KL:{:<.3f} mse:{:<.3f}  grad norm:{:.4f} T:{:d}'.\
                format(k, Niter, np.mean(losses), np.mean(lhoods), np.mean(kls), np.mean(mses), np.mean(grad_norms), \
                st.shape[1]))
        if hasattr(optimizer, 'param_groups'):
            current_dyn_lr = float(optimizer.param_groups[0]['lr'])
        elif hasattr(optimizer, 'opt') and hasattr(optimizer.opt, 'param_groups'):
            current_dyn_lr = float(optimizer.opt.param_groups[0]['lr'])
        else:
            current_dyn_lr = float(eta)
        _wandb_log(
            wandb_run,
            {
                'dynamics/round': 0 if round_idx is None else int(round_idx),
                'dynamics/global_step': (0 if round_idx is None else int(round_idx)) * int(Niter) + int(k),
                'dynamics/iter': k,
                'dynamics/loss': np.mean(losses),
                'dynamics/loss_cur': loss.item(),
                'dynamics/lhood': np.mean(lhoods),
                'dynamics/lhood_cur': lhood_,
                'dynamics/kl': np.mean(kls),
                'dynamics/kl_cur': kl_,
                'dynamics/mse': np.mean(mses),
                'dynamics/mse_cur': mse,
                'dynamics/grad_norm': np.mean(grad_norms),
                'dynamics/grad_norm_cur': grad_norm_,
                'dynamics/grad_norm_pre_cur': grad_norm_pre_,
                'dynamics/lr': current_dyn_lr,
            },
        )
        if (k+1)%save_every == 0:
            ctrl.save(D, fname=save_fname)
        if num_plots>0 and (k+1)%(Niter//num_plots) == 0:
            plot_model(ctrl, D, L=10, rep_buf=10, fname=save_fname+'-train.png', verbose=False)
            plot_test(ctrl,  D, L=10, N=10, fname=save_fname+'-test.png', verbose=False)
        if mse/(H/ctrl.env.dt) < stop_mse:
            num_below_thresholds += 1
            if num_below_thresholds > 10:
                print(f'Optimization converged at iter {k}. Breaking...')
                ctrl.save(D, fname=save_fname)
                break
    if num_plots>0:
        plot_model(ctrl, D, L=10, rep_buf=10, fname=save_fname+'-train.png')
        plot_test(ctrl,  D, L=10, N=10, fname=save_fname+'-test.png')
    return losses,optimizer

def compute_full_batch_loss(ctrl, D, L=10):
    g,st,at,rt,tobs = D.extract_data(H=D.T*D.dt, cont=ctrl.is_cont)
    mse, sum_lhood, st_hat, at_hat = dynamics_loss(ctrl, st, tobs, at, g, L=L)
    return mse

def train_ode(ctrl, D, save_fname, verbose, print_times, **kwargs):
    dyn_lr = kwargs.get('dyn_lr', 1e-3)
    dyn_rep_buf = kwargs.get('dyn_rep_buf', 50)
    dyn_save_every = kwargs.get('dyn_save_every', 50)
    dyn_grad_clip = kwargs.get('dyn_grad_clip', 10.0)
    if D.N==ctrl.env.N0: # if the training has just started
        print('Drift is being initialized with gradient matching.')
        ctrl0 = copy.deepcopy(ctrl)
        loss0 = compute_full_batch_loss(ctrl0, D)
        gradient_match(ctrl, D, Niter=500, L=kwargs['L'], print_times=print_times)
        loss1 = compute_full_batch_loss(ctrl, D)
        ctrl = copy.deepcopy(ctrl0) if loss1>loss0 else ctrl
        print('Drift initialized.')
    for H_ in [ctrl.env.dt*5]:
        train_dynamics(ctrl, D, Niter=1250, L=kwargs['L'], H=H_, eta=dyn_lr, save_fname=save_fname, \
            verbose=verbose, print_times=print_times, rep_buf=dyn_rep_buf, nrep=3, save_every=dyn_save_every, \
            wandb_run=kwargs.get('wandb_run', None), round_idx=kwargs.get('round_idx', None), \
            grad_clip=dyn_grad_clip)

def train_deep_pilco(ctrl, D, save_fname, verbose, print_times, **kwargs):
    Niter = 5000
    L = 100
    rep_buf = -1
    rep_buf = D.N if rep_buf<1 else rep_buf
    ds = (D.s[-rep_buf:,1:]-D.s[-rep_buf:,:-1]).reshape(-1,ctrl.env.n)
    dt = (D.ts[-rep_buf:,1:] - D.ts[-rep_buf:,:-1]).reshape(-1,1)
    ds_dt_opt = ds / dt # _,n
    s_opt = D.s[-rep_buf:,:-1].reshape(-1,ctrl.env.n) # _,n 
    a_opt = D.a[-rep_buf:,:-1].reshape(-1,ctrl.env.m) # _,m
    errors = []
    L = ctrl.get_L(L=L)
    s_opt,a_opt = torch.stack([s_opt]*L), torch.stack([a_opt]*L)
    optimizer = torch.optim.Adam(ctrl._f.parameters(),lr=1e-3)
    for i in range(Niter):
        optimizer.zero_grad()
        f = ctrl.draw_f(L=L)
        # optimize buffer
        ds_dt_opt_hat = ctrl._f.ds_dt(f,s_opt,a_opt) # N*(T-1),n
        opt_error = torch.sum((ds_dt_opt_hat-ds_dt_opt)**2)
        error = opt_error + ctrl._f.kl()
        errors.append(error.item())
        error.backward()
        optimizer.step()
        if verbose and i%(Niter/print_times)==0:
            print('Iter={:4d}/{:<4d}   opt_error={:.3f}'.format(i, Niter, np.mean(errors)))
        if (i+1)%(Niter//10) == 0:
            ctrl.save(D, fname=save_fname)

def train_pets(ctrl, D, save_fname, verbose, print_times, **kwargs):
    Niter = 5000
    C = 0.01
    rep_buf = -1
    rep_buf = D.N if rep_buf<1 else rep_buf
    ds = (D.s[-rep_buf:,1:]-D.s[-rep_buf:,:-1]).reshape(-1,ctrl.env.n)
    dt = (D.ts[-rep_buf:,1:] - D.ts[-rep_buf:,:-1]).reshape(-1,1)
    ds_dt_opt = ds / dt # _,n
    ds_dt_opt_L = torch.stack([ds_dt_opt]*ctrl.n_ens)
    s_opt = D.s[-rep_buf:,:-1].reshape(-1,ctrl.env.n) # _,n 
    a_opt = D.a[-rep_buf:,:-1].reshape(-1,ctrl.env.m) # _,m
    errors = []
    L = ctrl.get_L()
    s_opt,a_opt = torch.stack([s_opt]*L), torch.stack([a_opt]*L)
    optimizer = torch.optim.Adam(ctrl._f.parameters(),lr=1e-3)
    for i in range(Niter):
        optimizer.zero_grad()
        means,sig = ctrl._f._f.get_probs(torch.cat([s_opt,a_opt],-1))
        lhood = torch.distributions.Normal(means,sig).log_prob(ds_dt_opt_L).sum() / L
        error = -lhood + C*(ctrl._f._f.max_logsig-ctrl._f._f.min_logsig).sum()
        errors.append(error.item())
        error.backward()
        optimizer.step()
        if verbose and i%(Niter/print_times)==0:
            print('Iter={:4d}/{:<4d}   opt_error={:.3f}'.format(i, Niter, np.mean(errors)))
        if (i+1)%(Niter//10) == 0:
            ctrl.save(D, fname=save_fname)

##########################################################################################
################################# DATA COLLECTION ########################################
##########################################################################################
def _clip_actions(env,U):
    if env.ac_lb is not None:
        ac_lb = env.ac_lb.repeat([*(U.shape[:-1]),1])
        U[U<ac_lb] = ac_lb[U<ac_lb]
    if env.ac_ub is not None:
        ac_ub = env.ac_ub.repeat([*(U.shape[:-1]),1])
        U[U>ac_ub] = ac_ub[U>ac_ub]
    return U

def draw_from_gp(inputs, sf, ell, L=1, N=1, n_out=1, eps=1e-5):
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(1) 
    T = inputs.shape[0]
    cov  = K(inputs,inputs,ell,sf,eps=eps) # T,T
    L_ = torch.linalg.cholesky(cov) # psd_safe_cholesky(cov)
    # L,N,T,n_out or N,T,n_out or T,n_out
    return L_ @ torch.randn([L,N,T,n_out],device=inputs.device).squeeze(0).squeeze(0)

def obtain_smooth_test_acts(env, T, sf=1.0, ell=0.5, eps=1e-5):
    with torch.no_grad():
        t = torch.arange(T,device=env.device) * env.dt
        acs = draw_from_gp(t, sf, ell, n_out=env.m, eps=eps).to(torch.float32)
        acs = acs * (env.ac_ub-env.ac_lb)/2 # T,m
        return _clip_actions(env,acs) # T,m

def get_kernel_interpolation(env, T, N=1, ell=0.25, sf=0.5):
    with torch.no_grad():
        ell = numpy_to_torch([ell], env.device)
        sf  = numpy_to_torch([sf],  env.device)
        ts = env.dt * torch.arange(T,device=env.device) # T
        smooth_noise = torch.stack([obtain_smooth_test_acts(env,T,sf=sf,ell=ell) for _ in range(N)]) # N,T,1
        ells = torch.stack([ell]*N).unsqueeze(-1) # N,1,1
        sfs  = torch.stack([sf]*N).unsqueeze(-1) # N,1,1
        tss  = torch.stack([ts]*N).unsqueeze(-1) # N,T,1
        kernel_int = KernelInterpolation(sfs, ells, tss, smooth_noise)
        def g(s,t):
            return kernel_int(torch.stack([t.view([1,1])]*N)).squeeze(1)
        return g
    
def build_policy(env, T, g_pol=None, sf=0.1, ell=0.5):
    g_exp = get_kernel_interpolation(env, T, ell=ell, sf=sf)
    tanh_ = torch.nn.Tanh()
    def g(s,t):
        a_pol = g_pol(s,t) if g_pol is not None else 0.0
        a_exp = g_exp(s,t)
        return tanh_(a_pol+a_exp) * (env.ac_ub-env.ac_lb) / 2.0
    return g

def collect_data(env, H, N=1, sf=0.5, ell=0.5, D=None):
    ''' H in seconds '''
    with torch.no_grad():
        if N<1:
            print('Since N<1, data not collected!')
            return D
        T = int(H/env.dt)
        s0 = torch.stack([numpy_to_torch(env.reset(), env.device) for _ in range(N)]) # N,n
        for i in range(N):
            g = build_policy(env,T)
            st,at,rt,ts = env.integrate_system(T, g, s0[i:i+1])
            st_at_rt = torch.cat([st,at,rt.unsqueeze(-1)],-1) # N,T,n+m+1
            if D is None:
                D = Dataset(env, st_at_rt, ts)
            else:
                D.add_experience(st_at_rt, ts)
        return D

def collect_test_sequences(ctrl, D, N=1, reset=True, explore=False, sf=0.1):
    with torch.no_grad():
        env = ctrl.env
        Dnew = None
        T = D.T
        if reset:
            s0 = torch.stack([torch.tensor(env.reset(),dtype=torch.float32)\
                   for _ in range(N)]).to(ctrl.device)
        else:
            s0 = get_high_f_uncertainty_iv(ctrl, D, N=N)
        for i in range(N):
            g = build_policy(env, T, sf=sf) if explore else ctrl._g
            st_obs,at,rt,ts = env.integrate_system(T, g, s0[i:i+1])
            st_at_rt = torch.cat([st_obs,at,rt.unsqueeze(-1)],-1) # T,n+m+1
            if Dnew is None:
                Dnew = Dataset(env, st_at_rt, ts)
            else:
                Dnew.add_experience(st_at_rt, ts)
        return Dnew


def collect_experience(ctrl, D, N=1, H=None, reset=False, explore=False, sf=0.1):
    with torch.no_grad():
        Dnew = collect_test_sequences(ctrl, D, N=N, reset=reset, explore=explore, sf=sf)
        D.add_experience(Dnew.D, Dnew.ts)
        return D


##########################################################################################
######################################## UTILS ########################################
##########################################################################################
class TemperedOpt:
    def __init__(self, OPT_CLS, params, lr):
        self.opt = OPT_CLS(params,lr=lr)
        my_lambda = lambda ep: min(10.0,10**(ep/100)) / 10
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=my_lambda)
    def zero_grad(self):
        self.opt.zero_grad()
    def step(self):
        self.opt.step()
        self.scheduler.step()
class OptWrapper:
    def __init__(self, opt_name):
        self.OPT_CLS = get_opt(opt_name)
    def __call__(self,params,lr=1e-3):
        return TemperedOpt(self.OPT_CLS, params, lr)

def get_opt(opt, temp=False):
    if opt=='adam':
        CLS = torch.optim.Adam
    elif opt=='sgd':
        CLS = torch.optim.SGD
    elif opt=='sgld':
        from utils.sgld import SGLD
        CLS = SGLD
    elif opt=='rmsprop':
        CLS = torch.optim.RMSprop
    elif opt=='radam':
        from utils.radam import RAdam
        CLS = RAdam
    else:
        raise ValueError('optimizer parameter is wrong\n')
    if temp:
        return OptWrapper(opt)
    else:
        return CLS


def gradient_match(ctrl, D, Niter=5000, eta=5e-3, verbose=False, L=10, print_times=100, \
                   rep_buf=-1, opt='adam'):
    rep_buf = D.N if rep_buf<1 else rep_buf
    ds = (D.s[-rep_buf:,1:]-D.s[-rep_buf:,:-1]).reshape(-1,ctrl.env.n)
    dt = (D.ts[-rep_buf:,1:] - D.ts[-rep_buf:,:-1]).reshape(-1,1)
    ds_dt_opt = ds / dt # _,n
    s_opt = D.s[-rep_buf:,:-1].reshape(-1,ctrl.env.n) # _,n 
    a_opt = D.a[-rep_buf:,:-1].reshape(-1,ctrl.env.m) # _,m
    errors = []
    L = ctrl.get_L(L=L)
    s_opt,a_opt = torch.stack([s_opt]*L), torch.stack([a_opt]*L)
    OPT_CLS = get_opt(opt)
    optimizer = OPT_CLS(ctrl._f.parameters(),lr=eta)
    for i in range(Niter):
        optimizer.zero_grad()
        f = ctrl.draw_f(L=L)
        # optimize buffer
        ds_dt_opt_hat = ctrl._f.ds_dt(f,s_opt,a_opt) # N*(T-1),n
        opt_error = torch.sum((ds_dt_opt_hat-ds_dt_opt)**2)
        error = opt_error + ctrl._f.kl()
        errors.append(error.item())
        error.backward()
        optimizer.step()
        if verbose and i%(Niter/print_times)==0:
            print('Iter={:4d}/{:<4d}   opt_error={:.3f}'.format(i, Niter, np.mean(errors)))

def get_ivs(env, D, N, rep_buf=None):
    if rep_buf is None and D is not None:
        rep_buf = D.N
    seq_idx = np.arange(D.N)[-rep_buf:]
    s0  = D.s[seq_idx].view([-1,env.n])
    idx = torch.randint(s0.shape[0],[N])
    s0  = torch.stack([s0[idx_.item()] for idx_ in idx])
    return s0.to(env.device)

def get_high_f_uncertainty_iv(ctrl, D, N=1, rep_buf=5, nrep=5, L=10):
    st = get_ivs(ctrl.env, D, 5*N, rep_buf=rep_buf)
    at = ctrl._g(st,None) # nrep,N,m
    rt = ctrl.env.diff_reward(st,at) # nrep,N
    st,at,rt = st.view([-1,D.n]), at.view([-1,D.m]), rt.view([-1])
    L = ctrl.get_L(L=L)
    f = ctrl.draw_f(L=L)
    stL,atL = torch.stack([st]*L),torch.stack([at]*L)
    fL = ctrl._f.ds_dt(f,stL,atL)
    var_ = fL.var(0).mean(-1)
    rt = rt / (rt.max()-rt.min())
    var_ = var_ / (var_.max()-var_.min())
    scores = rt + var_
    winner_idx = scores.argsort().flip(0)
    return st[winner_idx[:N]]













