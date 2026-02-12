#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ONPOLICY_DIR = ROOT / "on-policy"
if str(ONPOLICY_DIR) not in sys.path:
    sys.path.insert(0, str(ONPOLICY_DIR))

try:
    from gymnasium.spaces import Box
except Exception:
    from gym.spaces import Box

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from runner_coop_ma_enode import MAMUJOCO_ENV_SPECS, MaMuJoCoEnv
from runner_mappo import FIXED_EPISODE_SECONDS, _build_policy_inference_fn, evaluate_with_runner_strategy


DEFAULT_ENVS = "swimmer,cheetah6x1,ant2x4,ant2x4d,ant4x2,walker"
CKPT_ACTOR_RE = re.compile(
    r"^best_mappo_seed(?P<seed>\d+)_run(?P<run_ts>\d{8}_\d{6})_env(?P<env_name>.+)_ep(?P<episode>\d+)_actor\.pt$"
)


@dataclass(frozen=True)
class CandidateCheckpoint:
    env_alias: str
    env_name: str
    seed: int
    run_ts: str
    episode: int
    actor_path: Path
    critic_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Visualize one best MAPPO policy per MaMuJoCo environment. "
            "Best is selected by re-evaluating discovered best_mappo checkpoints."
        )
    )
    p.add_argument(
        "--envs",
        type=str,
        default=DEFAULT_ENVS,
        help="Comma-separated MuJoCo env aliases.",
    )
    p.add_argument("--checkpoint-dir", type=str, default="checkpoint")
    p.add_argument("--output-dir", type=str, default="vid/mappo")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--horizon-sec", type=float, default=FIXED_EPISODE_SECONDS)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--agent-obsk", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--algorithm-name", type=str, default="mappo", choices=["mappo", "rmappo"])
    p.add_argument(
        "--eval-seed-base",
        type=int,
        default=111,
        help="Evaluation uses fixed seeds [base, base+9] for all candidates.",
    )
    p.add_argument("--eval-env-workers", type=int, default=4)
    p.add_argument(
        "--max-candidates-per-env",
        type=int,
        default=0,
        help="0 means evaluate all candidates; >0 limits to the most recent N per env.",
    )
    p.add_argument(
        "--mujoco-gl",
        type=str,
        default="auto",
        choices=["auto", "egl", "osmesa", "glfw"],
    )
    p.add_argument("--worker-timeout-sec", type=float, default=300.0)
    p.add_argument("--save-actions", action="store_true")
    return p.parse_args()


def parse_csv(raw: str) -> List[str]:
    out: List[str] = []
    for token in str(raw).replace("，", ",").split(","):
        token = token.strip()
        if token:
            out.append(token)
    return out


def env_alias_to_env_name(env_alias: str) -> str:
    scenario, agent_conf = MAMUJOCO_ENV_SPECS[env_alias]
    return f"mamujoco-{scenario.lower()}-{agent_conf.lower()}"


def normalize_env_aliases(raw_aliases: List[str]) -> List[str]:
    dedup_by_env_name = set()
    normalized: List[str] = []
    for alias in raw_aliases:
        if alias not in MAMUJOCO_ENV_SPECS:
            raise ValueError(f"Unknown env alias: {alias}")
        env_name = env_alias_to_env_name(alias)
        if env_name in dedup_by_env_name:
            continue
        dedup_by_env_name.add(env_name)
        normalized.append(alias)
    return normalized


def resolve_output_dir(path_str: str) -> Path:
    out = Path(path_str)
    try:
        out.mkdir(parents=True, exist_ok=True)
        return out
    except PermissionError:
        fallback = Path.cwd() / "vid" / "mappo"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"[warn] no permission for {out}; fallback to {fallback}")
        return fallback


def configure_mujoco_gl(mode: str) -> str:
    existing = os.environ.get("MUJOCO_GL", "").strip().lower()
    if mode == "auto":
        if existing:
            chosen = existing
        else:
            chosen = "egl" if not os.environ.get("DISPLAY") else "glfw"
    else:
        chosen = str(mode).strip().lower()

    os.environ["MUJOCO_GL"] = chosen
    if chosen == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    elif chosen == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    return chosen


def candidate_gl_backends(mode: str) -> List[str]:
    if mode != "auto":
        return [mode]
    if os.environ.get("DISPLAY"):
        return ["glfw", "egl", "osmesa"]
    return ["egl", "osmesa", "glfw"]


def make_policy_args(algorithm_name: str) -> SimpleNamespace:
    use_recurrent = str(algorithm_name) == "rmappo"
    return SimpleNamespace(
        algorithm_name=str(algorithm_name),
        hidden_size=64,
        layer_N=1,
        lr=5e-4,
        critic_lr=5e-4,
        opti_eps=1e-5,
        weight_decay=0.0,
        recurrent_N=1,
        gain=0.01,
        use_orthogonal=True,
        use_policy_active_masks=True,
        use_naive_recurrent_policy=False,
        use_recurrent_policy=use_recurrent,
        use_feature_normalization=True,
        use_ReLU=True,
        stacked_frames=1,
        use_stacked_frames=False,
        use_popart=False,
    )


def build_policy_from_checkpoint(
    candidate: CandidateCheckpoint,
    dt: float,
    agent_obsk: int,
    device: torch.device,
    algorithm_name: str,
) -> Tuple[R_MAPPOPolicy, MaMuJoCoEnv, SimpleNamespace]:
    scenario, agent_conf = MAMUJOCO_ENV_SPECS[candidate.env_alias]
    env = MaMuJoCoEnv(
        scenario=scenario,
        agent_conf=agent_conf,
        dt=float(dt),
        device=device,
        obs_noise=0.0,
        ts_grid="fixed",
        solver="rk4",
        consensus_weight=0.02,
        num_env_workers=1,
        agent_obsk=int(agent_obsk),
    )

    act_dims = [int(d) for d in env.act_dims]
    if len(set(act_dims)) != 1:
        raise ValueError(
            f"MAPPO shared policy requires homogeneous action dims, got {act_dims} for env={candidate.env_alias}"
        )
    act_dim = int(act_dims[0])
    obs_dim = int(env.obs_dim)
    low = np.asarray(env.agent_action_lows[0], dtype=np.float32).reshape(-1)
    high = np.asarray(env.agent_action_highs[0], dtype=np.float32).reshape(-1)

    obs_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    share_obs_space = Box(low=-np.inf, high=np.inf, shape=(int(env.n),), dtype=np.float32)
    act_space = Box(low=low, high=high, shape=(act_dim,), dtype=np.float32)

    policy_args = make_policy_args(algorithm_name=algorithm_name)
    policy = R_MAPPOPolicy(policy_args, obs_space, share_obs_space, act_space, device=device)

    try:
        actor_sd = torch.load(candidate.actor_path, map_location=device, weights_only=True)
        critic_sd = torch.load(candidate.critic_path, map_location=device, weights_only=True)
    except TypeError:
        actor_sd = torch.load(candidate.actor_path, map_location=device)
        critic_sd = torch.load(candidate.critic_path, map_location=device)
    policy.actor.load_state_dict(actor_sd)
    policy.critic.load_state_dict(critic_sd)
    policy.actor.eval()
    policy.critic.eval()
    return policy, env, policy_args


def discover_candidate_checkpoints(
    checkpoint_dir: Path,
    env_aliases: List[str],
) -> Dict[str, List[CandidateCheckpoint]]:
    env_name_to_alias = {env_alias_to_env_name(alias): alias for alias in env_aliases}
    grouped: Dict[Tuple[str, int, str], CandidateCheckpoint] = {}

    for actor_path in checkpoint_dir.glob("best_mappo_seed*_run*_envmamujoco-*_ep*_actor.pt"):
        m = CKPT_ACTOR_RE.match(actor_path.name)
        if m is None:
            continue
        env_name = str(m.group("env_name"))
        env_alias = env_name_to_alias.get(env_name, None)
        if env_alias is None:
            continue
        seed = int(m.group("seed"))
        run_ts = str(m.group("run_ts"))
        episode = int(m.group("episode"))
        critic_path = actor_path.with_name(actor_path.name.replace("_actor.pt", "_critic.pt"))
        if not critic_path.is_file():
            continue
        cand = CandidateCheckpoint(
            env_alias=env_alias,
            env_name=env_name,
            seed=seed,
            run_ts=run_ts,
            episode=episode,
            actor_path=actor_path,
            critic_path=critic_path,
        )
        key = (env_alias, seed, run_ts)
        prev = grouped.get(key, None)
        if prev is None or cand.episode > prev.episode:
            grouped[key] = cand

    by_env: Dict[str, List[CandidateCheckpoint]] = {alias: [] for alias in env_aliases}
    for cand in grouped.values():
        by_env[cand.env_alias].append(cand)
    for alias in by_env:
        by_env[alias].sort(key=lambda x: (x.run_ts, x.episode, x.seed), reverse=True)
    return by_env


def evaluate_candidate(
    candidate: CandidateCheckpoint,
    dt: float,
    agent_obsk: int,
    device: torch.device,
    algorithm_name: str,
    eval_seed_base: int,
    eval_env_workers: int,
) -> Dict[str, float]:
    policy, env, policy_args = build_policy_from_checkpoint(
        candidate=candidate,
        dt=dt,
        agent_obsk=agent_obsk,
        device=device,
        algorithm_name=algorithm_name,
    )
    eval_args = SimpleNamespace(
        algorithm_name=str(algorithm_name),
        eval_env_workers=int(eval_env_workers),
        seed=int(eval_seed_base),
        recurrent_N=int(policy_args.recurrent_N),
        hidden_size=int(policy_args.hidden_size),
        use_recurrent_policy=bool(policy_args.use_recurrent_policy),
        use_naive_recurrent_policy=bool(policy_args.use_naive_recurrent_policy),
    )
    metrics = evaluate_with_runner_strategy(env, policy, eval_args)
    return {k: float(v) if isinstance(v, (int, float, np.generic)) else v for k, v in metrics.items()}


def select_best_candidates(
    by_env: Dict[str, List[CandidateCheckpoint]],
    env_aliases: List[str],
    dt: float,
    agent_obsk: int,
    device: torch.device,
    algorithm_name: str,
    eval_seed_base: int,
    eval_env_workers: int,
    max_candidates_per_env: int,
) -> Dict[str, Dict[str, object]]:
    selected: Dict[str, Dict[str, object]] = {}
    for env_alias in env_aliases:
        candidates = list(by_env.get(env_alias, []))
        if len(candidates) == 0:
            print(f"[warn] env={env_alias}: no candidate checkpoints found.")
            continue
        if int(max_candidates_per_env) > 0:
            candidates = candidates[: int(max_candidates_per_env)]

        best: Optional[Dict[str, object]] = None
        print(f"[select] env={env_alias}: evaluating {len(candidates)} candidate(s)")
        for idx, cand in enumerate(candidates, start=1):
            try:
                metrics = evaluate_candidate(
                    candidate=cand,
                    dt=dt,
                    agent_obsk=agent_obsk,
                    device=device,
                    algorithm_name=algorithm_name,
                    eval_seed_base=eval_seed_base,
                    eval_env_workers=eval_env_workers,
                )
            except Exception as e:
                print(
                    f"[warn] env={env_alias} candidate {idx}/{len(candidates)} "
                    f"(seed={cand.seed}, run={cand.run_ts}, ep={cand.episode}) failed: {e}"
                )
                continue

            score = float(metrics.get("eval/true_reward", -np.inf))
            print(
                f"[eval] env={env_alias} cand={idx}/{len(candidates)} "
                f"seed={cand.seed} run={cand.run_ts} ep={cand.episode} "
                f"true_reward={score:.4f}"
            )

            cand_pack = {
                "candidate": cand,
                "metrics": metrics,
            }
            if best is None:
                best = cand_pack
            else:
                best_score = float(best["metrics"].get("eval/true_reward", -np.inf))
                if (score > best_score) or (
                    np.isclose(score, best_score)
                    and (cand.run_ts, cand.episode, cand.seed)
                    > (
                        best["candidate"].run_ts,
                        int(best["candidate"].episode),
                        int(best["candidate"].seed),
                    )
                ):
                    best = cand_pack

        if best is None:
            print(f"[warn] env={env_alias}: all candidates failed during evaluation.")
            continue

        selected[env_alias] = best
        best_cand = best["candidate"]
        best_score = float(best["metrics"]["eval/true_reward"])
        print(
            f"[best] env={env_alias} seed={best_cand.seed} run={best_cand.run_ts} "
            f"ep={best_cand.episode} true_reward={best_score:.4f}"
        )
    return selected


def make_render_env(scenario: str, agent_conf: str, agent_obsk: int):
    from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1 as mamujoco

    kwargs = {
        "scenario": scenario,
        "agent_conf": agent_conf,
        "agent_obsk": int(agent_obsk),
        "render_mode": "rgb_array",
    }
    if scenario in {"Ant", "Walker2d", "Hopper", "Humanoid", "HumanoidStandup"}:
        kwargs["terminate_when_unhealthy"] = False
    return mamujoco.parallel_env(**kwargs)


@torch.no_grad()
def rollout_with_policy_video(
    candidate: CandidateCheckpoint,
    seed: int,
    horizon_sec: float,
    dt: float,
    agent_obsk: int,
    device_str: str,
    algorithm_name: str,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[str]]:
    device = torch.device(device_str)
    policy, env_adapter, policy_args = build_policy_from_checkpoint(
        candidate=candidate,
        dt=dt,
        agent_obsk=agent_obsk,
        device=device,
        algorithm_name=algorithm_name,
    )
    policy_fn = _build_policy_inference_fn(policy, env_adapter, policy_args)
    if hasattr(policy_fn, "reset_state"):
        policy_fn.reset_state()

    scenario, agent_conf = MAMUJOCO_ENV_SPECS[candidate.env_alias]
    env = make_render_env(scenario=scenario, agent_conf=agent_conf, agent_obsk=agent_obsk)

    frames: List[np.ndarray] = []
    action_seq: List[np.ndarray] = []
    times: List[float] = []
    t_steps = max(1, int(round(float(horizon_sec) / float(dt))))

    try:
        obs, _ = env.reset(seed=int(seed))
        agent_ids = list(env_adapter.agent_ids)

        frame0 = env.render()
        if isinstance(frame0, np.ndarray):
            frames.append(frame0)

        for t_idx in range(t_steps):
            if len(env.agents) == 0:
                break

            joint_obs = env_adapter._obs_dict_to_joint(obs)
            s = torch.as_tensor(joint_obs, dtype=torch.float32, device=device).unsqueeze(0)
            t = torch.tensor(float(t_idx) * float(dt), dtype=torch.float32, device=device)
            action = policy_fn(s, t)
            if isinstance(action, torch.Tensor):
                flat_action = action.reshape(-1).detach().cpu().numpy().astype(np.float32)
            else:
                flat_action = np.asarray(action, dtype=np.float32).reshape(-1)

            act_dict = env_adapter._split_joint_action(flat_action)
            obs, rewards, terms, truncs, infos = env.step(act_dict)

            action_seq.append(flat_action.copy())
            times.append(float(t_idx) * float(dt))

            frame = env.render()
            if isinstance(frame, np.ndarray):
                frames.append(frame)

            done = (
                len(env.agents) == 0
                or (len(terms) > 0 and all(bool(v) for v in terms.values()))
                or (len(truncs) > 0 and all(bool(v) for v in truncs.values()))
            )
            if done:
                break
    finally:
        env.close()

    if len(action_seq) == 0:
        action_arr = np.zeros((0, 0), dtype=np.float32)
    else:
        action_arr = np.stack(action_seq, axis=0).astype(np.float32)
    time_arr = np.asarray(times, dtype=np.float32)
    return frames, action_arr, time_arr, list(env_adapter.agent_ids)


def write_video(frames: List[np.ndarray], out_path: Path, fps: int) -> Path:
    if len(frames) == 0:
        raise RuntimeError("No frames captured from env.render().")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(str(out_path), fps=int(fps), macro_block_size=None) as writer:
            for fr in frames:
                writer.append_data(fr)
        return out_path
    except Exception as e:
        gif_path = out_path.with_suffix(".gif")
        imageio.mimsave(str(gif_path), frames, duration=max(1e-3, 1.0 / float(fps)))
        print(f"[warn] mp4 writer failed ({e}); wrote gif instead: {gif_path}")
        return gif_path


def rollout_job_worker(
    queue,
    run_name: str,
    env_alias: str,
    actor_path: str,
    critic_path: str,
    seed: int,
    ep: int,
    out_dir: str,
    fps: int,
    save_actions: bool,
    gl_backend: str,
    horizon_sec: float,
    dt: float,
    agent_obsk: int,
    device_str: str,
    algorithm_name: str,
):
    try:
        torch.set_num_threads(1)
        chosen_gl = configure_mujoco_gl(gl_backend)

        candidate = CandidateCheckpoint(
            env_alias=str(env_alias),
            env_name=env_alias_to_env_name(str(env_alias)),
            seed=0,
            run_ts="",
            episode=0,
            actor_path=Path(actor_path),
            critic_path=Path(critic_path),
        )
        frames, actions, times, agent_ids = rollout_with_policy_video(
            candidate=candidate,
            seed=seed,
            horizon_sec=horizon_sec,
            dt=dt,
            agent_obsk=agent_obsk,
            device_str=device_str,
            algorithm_name=algorithm_name,
        )

        run_token = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name)
        out_path = Path(out_dir) / f"{run_token}_ep{ep}_H{horizon_sec:.1f}_dt{dt:.2f}_{chosen_gl}.mp4"
        actual_path = write_video(frames, out_path=out_path, fps=fps)

        saved_npz = ""
        if save_actions:
            npz_path = Path(out_dir) / f"{run_token}_ep{ep}_actions_{chosen_gl}.npz"
            np.savez_compressed(
                npz_path,
                actions=actions,
                times=times,
                agent_ids=np.asarray(agent_ids, dtype=object),
                dt=np.float32(dt),
                horizon_sec=np.float32(horizon_sec),
                mujoco_gl=np.asarray(chosen_gl),
                actor_path=np.asarray(actor_path),
                critic_path=np.asarray(critic_path),
            )
            saved_npz = str(npz_path)

        queue.put(
            {
                "ok": True,
                "video_path": str(actual_path),
                "npz_path": saved_npz,
                "gl_backend": chosen_gl,
            }
        )
    except Exception as e:
        queue.put({"ok": False, "error": str(e), "gl_backend": gl_backend})


def run_rollout_job_in_subprocess(
    run_name: str,
    env_alias: str,
    actor_path: str,
    critic_path: str,
    seed: int,
    ep: int,
    out_dir: Path,
    fps: int,
    save_actions: bool,
    gl_backend: str,
    timeout_sec: float,
    horizon_sec: float,
    dt: float,
    agent_obsk: int,
    device_str: str,
    algorithm_name: str,
) -> Dict[str, object]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=rollout_job_worker,
        args=(
            q,
            run_name,
            env_alias,
            actor_path,
            critic_path,
            int(seed),
            int(ep),
            str(out_dir),
            int(fps),
            bool(save_actions),
            str(gl_backend),
            float(horizon_sec),
            float(dt),
            int(agent_obsk),
            str(device_str),
            str(algorithm_name),
        ),
    )
    p.start()
    p.join(timeout=float(timeout_sec))

    if p.is_alive():
        p.terminate()
        p.join()
        return {
            "ok": False,
            "error": f"worker timeout after {timeout_sec:.1f}s",
            "exitcode": p.exitcode,
            "gl_backend": gl_backend,
        }

    result = None
    try:
        if not q.empty():
            result = q.get_nowait()
    except Exception:
        result = None

    if p.exitcode != 0 and (result is None or bool(result.get("ok", False)) is False):
        return {
            "ok": False,
            "error": f"worker exited abnormally (exitcode={p.exitcode})",
            "exitcode": p.exitcode,
            "gl_backend": gl_backend,
        }
    if result is None:
        return {
            "ok": False,
            "error": "worker produced no result",
            "exitcode": p.exitcode,
            "gl_backend": gl_backend,
        }
    return result


def main() -> None:
    args = parse_args()
    env_aliases = normalize_env_aliases(parse_csv(args.envs))
    if len(env_aliases) == 0:
        raise SystemExit("No valid MuJoCo env aliases provided.")

    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = resolve_output_dir(args.output_dir)
    gl_candidates = candidate_gl_backends(args.mujoco_gl)
    eval_device = torch.device(args.device)

    by_env = discover_candidate_checkpoints(checkpoint_dir=checkpoint_dir, env_aliases=env_aliases)
    for env_alias in env_aliases:
        print(f"[discover] env={env_alias} candidates={len(by_env.get(env_alias, []))}")

    selected = select_best_candidates(
        by_env=by_env,
        env_aliases=env_aliases,
        dt=float(args.dt),
        agent_obsk=int(args.agent_obsk),
        device=eval_device,
        algorithm_name=str(args.algorithm_name),
        eval_seed_base=int(args.eval_seed_base),
        eval_env_workers=int(args.eval_env_workers),
        max_candidates_per_env=int(args.max_candidates_per_env),
    )

    if len(selected) == 0:
        raise SystemExit("No best checkpoints selected; nothing to visualize.")

    print(
        f"Recording best MAPPO videos: envs={len(selected)}, episodes_per_env={int(args.episodes)}, "
        f"H={float(args.horizon_sec)}, dt={float(args.dt)}, fps={int(args.fps)}, "
        f"MUJOCO_GL_candidates={gl_candidates}, out={out_dir}"
    )

    total = 0
    for env_alias in env_aliases:
        sel = selected.get(env_alias, None)
        if sel is None:
            continue
        cand: CandidateCheckpoint = sel["candidate"]
        score = float(sel["metrics"]["eval/true_reward"])
        run_name = (
            f"{env_alias}-mappo-best-seed{cand.seed}-run{cand.run_ts}"
            f"-ep{cand.episode}-score{score:.3f}"
        )
        print(
            f"[map] env={env_alias} run={cand.run_ts} seed={cand.seed} ep={cand.episode} "
            f"score={score:.4f} actor={cand.actor_path}"
        )

        for ep in range(int(args.episodes)):
            viz_seed = int(cand.seed) + ep
            success = False
            for gl_backend in gl_candidates:
                res = run_rollout_job_in_subprocess(
                    run_name=run_name,
                    env_alias=env_alias,
                    actor_path=str(cand.actor_path),
                    critic_path=str(cand.critic_path),
                    seed=viz_seed,
                    ep=ep,
                    out_dir=out_dir,
                    fps=int(args.fps),
                    save_actions=bool(args.save_actions),
                    gl_backend=gl_backend,
                    timeout_sec=float(args.worker_timeout_sec),
                    horizon_sec=float(args.horizon_sec),
                    dt=float(args.dt),
                    agent_obsk=int(args.agent_obsk),
                    device_str=str(args.device),
                    algorithm_name=str(args.algorithm_name),
                )
                if bool(res.get("ok", False)):
                    if args.save_actions and str(res.get("npz_path", "")):
                        print(
                            f"[ok] env={env_alias} ep={ep} seed={viz_seed} gl={res.get('gl_backend')} "
                            f"saved: {res.get('video_path')} and {res.get('npz_path')}"
                        )
                    else:
                        print(
                            f"[ok] env={env_alias} ep={ep} seed={viz_seed} gl={res.get('gl_backend')} "
                            f"saved: {res.get('video_path')}"
                        )
                    total += 1
                    success = True
                    break
                print(
                    f"[warn] env={env_alias} ep={ep} seed={viz_seed} gl={gl_backend} failed: "
                    f"{res.get('error', 'unknown error')}"
                )
            if not success:
                print(f"[warn] env={env_alias} ep={ep} seed={viz_seed} failed on all backends: {gl_candidates}")

    if total == 0:
        raise SystemExit("No videos generated.")
    print(f"Done. Generated {total} video file(s) in {out_dir}")


if __name__ == "__main__":
    main()
