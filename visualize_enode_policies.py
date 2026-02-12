#!/usr/bin/env python3
import argparse
import io
import json
import multiprocessing as mp
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch

from runner_enode import (
    MAMUJOCO_ENV_SPECS,
    MaMuJoCoEnv,
    build_ctde_multi_controller_policy,
)

import ctrl.ctrl as base


DEFAULT_RUNS = (
    "walker-enode-gpu2-seed111-20260210_151028,"
    "swimmer-enode-gpu1-seed113-20260210_181308,"
    "cheetah6x1-enode-gpu6-seed112-20260210_151028,"
    "ant2x4-enode-gpu0-seed112-20260210_151028,"
    "ant2x4d-enode-gpu3-seed113-20260210_151028,"
    "ant2x4-enode-gpu0-seed112-20260210_181308,"
    "ant4x2-enode-gpu4-seed112-20260210_181308"
)

RUN_RE = re.compile(
    r"^(?P<env>[A-Za-z0-9]+)-enode-gpu(?P<gpu>\d+)-seed(?P<seed>\d+)-(?P<ts>\d{8}_\d{6})$"
)
BEST_LINE_RE = re.compile(r"Best-checkpoint tracking enabled: .*?run_ts=([0-9_]+), .*?seed=(\d+)")
FNAME_LINE_RE = re.compile(r"^fname is (output/[^\r\n]+)$", flags=re.MULTILINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Visualize policy rollouts for specific ENODE runs and save videos "
            "to vid/enode."
        )
    )
    p.add_argument(
        "--runs",
        type=str,
        default=DEFAULT_RUNS,
        help="Comma-separated run names.",
    )
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--agent-obsk", type=int, default=1)
    p.add_argument("--horizon-sec", type=float, default=2.5)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--output-dir", type=str, default="vid/enode")
    p.add_argument("--logs-dir", type=str, default="logs")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoint")
    p.add_argument(
        "--mujoco-gl",
        type=str,
        default="auto",
        choices=["auto", "egl", "osmesa", "glfw"],
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--worker-timeout-sec", type=float, default=300.0)
    p.add_argument("--save-actions", action="store_true")
    return p.parse_args()


def parse_csv(raw: str) -> List[str]:
    vals: List[str] = []
    for token in str(raw).replace("，", ",").split(","):
        token = token.strip()
        if token:
            vals.append(token)
    return vals


def parse_run_name(run_name: str) -> Dict[str, object]:
    m = RUN_RE.match(run_name.strip())
    if m is None:
        raise ValueError(
            f"Invalid run name format: {run_name}. "
            "Expected: <env>-enode-gpu<id>-seed<seed>-YYYYMMDD_HHMMSS"
        )
    env_alias = m.group("env")
    if env_alias not in MAMUJOCO_ENV_SPECS:
        raise ValueError(f"Unknown env alias in run name: {env_alias}")
    return {
        "run_name": run_name.strip(),
        "env_alias": env_alias,
        "gpu_id": int(m.group("gpu")),
        "seed": int(m.group("seed")),
        "wandb_ts": m.group("ts"),
    }


def env_alias_to_env_name(env_alias: str) -> str:
    scenario, agent_conf = MAMUJOCO_ENV_SPECS[env_alias]
    return f"mamujoco-{scenario.lower()}-{agent_conf.lower()}"


def resolve_output_dir(path_str: str) -> Path:
    out = Path(path_str)
    try:
        out.mkdir(parents=True, exist_ok=True)
        return out
    except PermissionError:
        fallback = Path.cwd() / "vid" / "enode"
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


def _load_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _resolve_log_path(run_info: Dict[str, object], logs_dir: Path) -> Path:
    env_alias = str(run_info["env_alias"])
    gpu_id = int(run_info["gpu_id"])
    seed = int(run_info["seed"])
    wandb_ts = str(run_info["wandb_ts"])
    return logs_dir / f"{env_alias}_enode_gpu{gpu_id}_seed{seed}_{wandb_ts}.log"


def _extract_run_ts(log_text: str, expected_seed: int) -> Optional[str]:
    for m in BEST_LINE_RE.finditer(log_text):
        run_ts = str(m.group(1))
        seed = int(m.group(2))
        if seed == int(expected_seed):
            return run_ts
    return None


def _extract_output_prefix(log_text: str) -> Optional[str]:
    m = FNAME_LINE_RE.search(log_text)
    if m is None:
        return None
    return m.group(1).strip()


def _find_best_checkpoint_by_seed_env(
    checkpoint_dir: Path,
    seed: int,
    env_name: str,
) -> Optional[Path]:
    best_path = None
    best_score = -np.inf
    best_ts = ""
    pattern = f"best_seed{int(seed)}_run*_algoenode_env{env_name}.meta.json"
    for meta_path in checkpoint_dir.glob(pattern):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            score = float(meta.get("best_eval_true_reward", -np.inf))
            run_ts = str(meta.get("training_run_timestamp", ""))
            ckpt_file = Path(str(meta.get("checkpoint_file", "")))
            if not ckpt_file.is_file():
                continue
            if (score > best_score) or (np.isclose(score, best_score) and run_ts > best_ts):
                best_score = score
                best_ts = run_ts
                best_path = ckpt_file
        except Exception:
            continue
    return best_path


def resolve_model_for_run(
    run_info: Dict[str, object],
    logs_dir: Path,
    checkpoint_dir: Path,
) -> Dict[str, object]:
    env_alias = str(run_info["env_alias"])
    seed = int(run_info["seed"])
    run_name = str(run_info["run_name"])
    env_name = env_alias_to_env_name(env_alias)

    log_path = _resolve_log_path(run_info, logs_dir)
    log_text = _load_text(log_path)

    if log_text:
        run_ts = _extract_run_ts(log_text, expected_seed=seed)
        if run_ts:
            ckpt = checkpoint_dir / f"best_seed{seed}_run{run_ts}_algoenode_env{env_name}.pkl"
            if ckpt.is_file():
                return {
                    **run_info,
                    "model_path": str(ckpt),
                    "model_source": "best_checkpoint_exact",
                    "training_run_ts": run_ts,
                    "log_path": str(log_path),
                }

        out_prefix = _extract_output_prefix(log_text)
        if out_prefix:
            out_pkl = Path(out_prefix + ".pkl")
            if out_pkl.is_file():
                return {
                    **run_info,
                    "model_path": str(out_pkl),
                    "model_source": "run_output_final",
                    "training_run_ts": "",
                    "log_path": str(log_path),
                }

    fallback_ckpt = _find_best_checkpoint_by_seed_env(
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        env_name=env_name,
    )
    if fallback_ckpt is not None:
        return {
            **run_info,
            "model_path": str(fallback_ckpt),
            "model_source": "best_checkpoint_seed_env_fallback",
            "training_run_ts": "",
            "log_path": str(log_path),
        }

    raise FileNotFoundError(
        f"Unable to resolve model for run={run_name}. "
        f"log={log_path} checkpoint_dir={checkpoint_dir}"
    )


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def load_checkpoint_raw(pkl_path: Path):
    with pkl_path.open("rb") as f:
        return CPUUnpickler(f).load()


def build_ctrl_from_checkpoint(
    env_alias: str,
    ckpt_path: Path,
    dt: float,
    agent_obsk: int,
    device: torch.device,
):
    scenario, agent_conf = MAMUJOCO_ENV_SPECS[env_alias]
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
    raw = load_checkpoint_raw(ckpt_path)
    dynamics = raw["dynamics"]
    kwargs = raw["kwargs"]
    state_dict = raw["state_dict"]

    ctrl = base.CTRL(env, dynamics, **kwargs).to(device)
    is_ctde = any(str(k).startswith("_g.agent_ctrls.") for k in state_dict.keys())
    if is_ctde:
        n_agents = int(getattr(env, "n_agents", 1))
        joint_policy, _, _, _ = build_ctde_multi_controller_policy(
            env=env,
            n_agents=n_agents,
            device=device,
            nl_g=int(kwargs.get("nl_g", 2)),
            nn_g=int(kwargs.get("nn_g", 200)),
            act_g=str(kwargs.get("act_g", "relu")),
        )
        ctrl._g = joint_policy
        ctrl = ctrl.to(device)

    incompatible = ctrl.load_state_dict(state_dict, strict=False)
    if len(incompatible.missing_keys) > 0 or len(incompatible.unexpected_keys) > 0:
        raise RuntimeError(
            "State dict mismatch while loading policy "
            f"(missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys})"
        )
    ctrl.eval()
    return ctrl, env


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


def obs_dict_to_joint(obs_dict: Dict[str, np.ndarray], agent_ids: List[str], obs_dim: int) -> np.ndarray:
    parts: List[np.ndarray] = []
    for aid in agent_ids:
        obs = np.asarray(obs_dict[aid], dtype=np.float32).reshape(-1)
        if obs.shape[0] < obs_dim:
            obs = np.pad(obs, (0, obs_dim - obs.shape[0]), mode="constant")
        elif obs.shape[0] > obs_dim:
            obs = obs[:obs_dim]
        parts.append(obs)
    if len(parts) == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def split_joint_action(joint_action: np.ndarray, env, agent_ids: List[str]) -> Dict[str, np.ndarray]:
    acts: Dict[str, np.ndarray] = {}
    offset = 0
    flat = np.asarray(joint_action, dtype=np.float32).reshape(-1)
    for aid in agent_ids:
        space = env.action_space(aid)
        dim = int(np.prod(space.shape))
        raw = flat[offset : offset + dim].reshape(space.shape)
        clipped = np.clip(raw, space.low, space.high).astype(np.float32)
        acts[aid] = clipped
        offset += dim
    if offset != flat.shape[0]:
        raise RuntimeError(
            f"Action dim mismatch: consumed={offset}, policy_out={flat.shape[0]}"
        )
    return acts


@torch.no_grad()
def rollout_with_policy_video(
    env_alias: str,
    model_path: Path,
    seed: int,
    horizon_sec: float,
    dt: float,
    agent_obsk: int,
    device_str: str,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[str]]:
    scenario, agent_conf = MAMUJOCO_ENV_SPECS[env_alias]
    device = torch.device(device_str)
    ctrl, ctrl_env = build_ctrl_from_checkpoint(
        env_alias=env_alias,
        ckpt_path=model_path,
        dt=dt,
        agent_obsk=agent_obsk,
        device=device,
    )
    obs_dim = int(getattr(ctrl_env, "obs_dim", 0))
    t_steps = max(1, int(round(float(horizon_sec) / float(dt))))

    env = make_render_env(scenario=scenario, agent_conf=agent_conf, agent_obsk=agent_obsk)
    frames: List[np.ndarray] = []
    action_seq: List[np.ndarray] = []
    times: List[float] = []

    try:
        obs, _ = env.reset(seed=int(seed))
        agent_ids = list(env.agents)
        if obs_dim <= 0:
            obs_dim = max(int(np.asarray(obs[aid]).reshape(-1).shape[0]) for aid in agent_ids)

        frame0 = env.render()
        if isinstance(frame0, np.ndarray):
            frames.append(frame0)

        for t_idx in range(t_steps):
            if len(env.agents) == 0:
                break

            joint_obs = obs_dict_to_joint(obs_dict=obs, agent_ids=agent_ids, obs_dim=obs_dim)
            s = torch.as_tensor(joint_obs, dtype=torch.float32, device=device).unsqueeze(0)
            t = torch.tensor(float(t_idx) * float(dt), dtype=torch.float32, device=device)
            action = ctrl._g(s, t)
            if isinstance(action, torch.Tensor):
                flat_action = action.reshape(-1).detach().cpu().numpy().astype(np.float32)
            else:
                flat_action = np.asarray(action, dtype=np.float32).reshape(-1)

            act_dict = split_joint_action(flat_action, env=env, agent_ids=agent_ids)
            obs, rewards, terms, truncs, infos = env.step(act_dict)

            action_seq.append(flat_action.copy())
            times.append(float(t_idx) * float(dt))

            frame = env.render()
            if isinstance(frame, np.ndarray):
                frames.append(frame)

            done = all(bool(v) for v in terms.values()) or all(bool(v) for v in truncs.values())
            if done:
                break
    finally:
        env.close()

    if len(action_seq) == 0:
        action_arr = np.zeros((0, 0), dtype=np.float32)
    else:
        action_arr = np.stack(action_seq, axis=0).astype(np.float32)
    time_arr = np.asarray(times, dtype=np.float32)
    return frames, action_arr, time_arr, agent_ids


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
    model_path: str,
    model_source: str,
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
):
    try:
        torch.set_num_threads(1)
        chosen_gl = configure_mujoco_gl(gl_backend)
        frames, actions, times, agent_ids = rollout_with_policy_video(
            env_alias=env_alias,
            model_path=Path(model_path),
            seed=seed,
            horizon_sec=horizon_sec,
            dt=dt,
            agent_obsk=agent_obsk,
            device_str=device_str,
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
                model_path=np.asarray(model_path),
                model_source=np.asarray(model_source),
            )
            saved_npz = str(npz_path)

        queue.put(
            {
                "ok": True,
                "video_path": str(actual_path),
                "npz_path": saved_npz,
                "gl_backend": chosen_gl,
                "model_source": model_source,
            }
        )
    except Exception as e:
        queue.put(
            {
                "ok": False,
                "error": str(e),
                "gl_backend": gl_backend,
                "model_source": model_source,
            }
        )


def run_rollout_job_in_subprocess(
    run_name: str,
    env_alias: str,
    model_path: str,
    model_source: str,
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
):
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=rollout_job_worker,
        args=(
            q,
            run_name,
            env_alias,
            model_path,
            model_source,
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
    run_names = parse_csv(args.runs)
    if len(run_names) == 0:
        raise SystemExit("No runs specified.")

    out_dir = resolve_output_dir(args.output_dir)
    logs_dir = Path(args.logs_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    gl_candidates = candidate_gl_backends(args.mujoco_gl)

    run_infos: List[Dict[str, object]] = []
    for run_name in run_names:
        parsed = parse_run_name(run_name)
        resolved = resolve_model_for_run(
            run_info=parsed,
            logs_dir=logs_dir,
            checkpoint_dir=checkpoint_dir,
        )
        run_infos.append(resolved)

    print(
        f"Recording policy videos: runs={len(run_infos)}, H={args.horizon_sec}, dt={args.dt}, "
        f"fps={args.fps}, MUJOCO_GL_candidates={gl_candidates}, out={out_dir}"
    )
    for item in run_infos:
        print(
            f"[map] run={item['run_name']} env={item['env_alias']} seed={item['seed']} "
            f"source={item['model_source']} model={item['model_path']}"
        )

    total = 0
    for item in run_infos:
        run_name = str(item["run_name"])
        env_alias = str(item["env_alias"])
        seed_base = int(item["seed"])
        model_path = str(item["model_path"])
        model_source = str(item["model_source"])

        for ep in range(int(args.episodes)):
            seed = seed_base + ep
            success = False
            for gl_backend in gl_candidates:
                res = run_rollout_job_in_subprocess(
                    run_name=run_name,
                    env_alias=env_alias,
                    model_path=model_path,
                    model_source=model_source,
                    seed=seed,
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
                )
                if bool(res.get("ok", False)):
                    if args.save_actions and str(res.get("npz_path", "")):
                        print(
                            f"[ok] run={run_name} ep={ep} gl={res.get('gl_backend')} "
                            f"source={res.get('model_source')} saved: {res.get('video_path')} "
                            f"and {res.get('npz_path')}"
                        )
                    else:
                        print(
                            f"[ok] run={run_name} ep={ep} gl={res.get('gl_backend')} "
                            f"source={res.get('model_source')} saved: {res.get('video_path')}"
                        )
                    total += 1
                    success = True
                    break
                print(
                    f"[warn] run={run_name} ep={ep} gl={gl_backend} failed: "
                    f"{res.get('error', 'unknown error')}"
                )
            if not success:
                print(f"[warn] run={run_name} ep={ep} failed on all backends: {gl_candidates}")

    if total == 0:
        raise SystemExit("No videos generated.")
    print(f"Done. Generated {total} video file(s) in {out_dir}")


if __name__ == "__main__":
    main()
