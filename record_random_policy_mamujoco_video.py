#!/usr/bin/env python3
import argparse
import os
import multiprocessing as mp
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np

from runner_coop_ma_enode import MAMUJOCO_ENV_SPECS


H_TEST_SEC = 2.5
DT_FIXED = 0.05
T_STEPS = int(round(H_TEST_SEC / DT_FIXED))
FPS = int(round(1.0 / DT_FIXED))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Record MuJoCo rollout videos under random policy for environments "
            "from run_enode_mujoco.sh."
        )
    )
    p.add_argument("--run-script", type=str, default="run_enode_mujoco.sh")
    p.add_argument(
        "--envs",
        type=str,
        default="",
        help="Comma-separated env aliases. Empty means parse from --run-script.",
    )
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--agent-obsk", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="/vid/random/")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument(
        "--mujoco-gl",
        type=str,
        default="auto",
        choices=["auto", "egl", "osmesa", "glfw"],
        help=(
            "MuJoCo OpenGL backend. auto: if DISPLAY missing, use egl; "
            "otherwise keep glfw."
        ),
    )
    p.add_argument(
        "--save-actions",
        action="store_true",
        help="Also save sampled actions/times to .npz next to each video.",
    )
    p.add_argument(
        "--worker-timeout-sec",
        type=float,
        default=180.0,
        help="Timeout for each env/episode recording worker process.",
    )
    return p.parse_args()


def parse_csv(raw: str) -> List[str]:
    vals = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            vals.append(token)
    return vals


def discover_envs_from_run_script(run_script_path: Path) -> List[str]:
    if not run_script_path.is_file():
        raise FileNotFoundError(f"run script not found: {run_script_path}")
    text = run_script_path.read_text(encoding="utf-8")
    child_scripts = re.findall(r'^\s*"([^"]+\.sh)"', text, flags=re.MULTILINE)
    if not child_scripts:
        raise RuntimeError(f"no shell scripts found in {run_script_path}")

    envs = OrderedDict()
    for script_name in child_scripts:
        script_path = (run_script_path.parent / script_name).resolve()
        if not script_path.is_file():
            print(f"[warn] script not found, skip: {script_path}")
            continue
        script_text = script_path.read_text(encoding="utf-8")
        m = re.search(r"--env\s+([^\s\\]+)", script_text)
        if m is None:
            print(f"[warn] --env missing in {script_path}, skip.")
            continue
        env_alias = m.group(1).strip()
        if env_alias not in envs:
            envs[env_alias] = True
    return list(envs.keys())


def resolve_envs(args: argparse.Namespace) -> List[str]:
    if args.envs.strip():
        return parse_csv(args.envs)
    return discover_envs_from_run_script(Path(args.run_script))


def resolve_output_dir(path_str: str) -> Path:
    out = Path(path_str)
    try:
        out.mkdir(parents=True, exist_ok=True)
        return out
    except PermissionError:
        default_out = Path("/vid/random/")
        if out.resolve() == default_out.resolve():
            fallback = Path.cwd() / "vid" / "random"
            fallback.mkdir(parents=True, exist_ok=True)
            print(f"[warn] no permission for {out}; fallback to {fallback}")
            return fallback
        raise SystemExit(
            f"No permission to create output dir: {out}. "
            "Use --output-dir with a writable path."
        )


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


def make_parallel_env(scenario: str, agent_conf: str, agent_obsk: int):
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


def sample_random_actions(env, agent_ids: List[str], rng: np.random.Generator) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    act_dict: Dict[str, np.ndarray] = {}
    flat = []
    for aid in agent_ids:
        space = env.action_space(aid)
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        act = rng.uniform(low, high).astype(np.float32)
        act_dict[aid] = act
        flat.append(act.reshape(-1))
    return act_dict, np.concatenate(flat, axis=0) if len(flat) > 0 else np.zeros((0,), dtype=np.float32)


def rollout_with_video(
    scenario: str,
    agent_conf: str,
    agent_obsk: int,
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[str]]:
    env = make_parallel_env(scenario=scenario, agent_conf=agent_conf, agent_obsk=agent_obsk)
    rng = np.random.default_rng(int(seed))
    frames: List[np.ndarray] = []
    action_seq: List[np.ndarray] = []
    times: List[float] = []

    try:
        obs, info = env.reset(seed=int(seed))
        agent_ids = list(env.agents)

        frame0 = env.render()
        if isinstance(frame0, np.ndarray):
            frames.append(frame0)

        for t_idx in range(T_STEPS):
            if len(env.agents) == 0:
                break
            act_dict, flat_action = sample_random_actions(env, agent_ids=agent_ids, rng=rng)
            obs, rewards, terms, truncs, infos = env.step(act_dict)

            action_seq.append(flat_action)
            times.append(t_idx * DT_FIXED)

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
    env_alias: str,
    scenario: str,
    agent_conf: str,
    agent_obsk: int,
    seed: int,
    ep: int,
    out_dir: str,
    fps: int,
    save_actions: bool,
    gl_backend: str,
):
    try:
        chosen_gl = configure_mujoco_gl(gl_backend)
        frames, actions, times, agent_ids = rollout_with_video(
            scenario=scenario,
            agent_conf=agent_conf,
            agent_obsk=agent_obsk,
            seed=seed,
        )
        out_path = Path(out_dir) / (
            f"{env_alias}_seed{seed}_ep{ep}_H{H_TEST_SEC:.1f}_dt{DT_FIXED:.2f}_{chosen_gl}.mp4"
        )
        actual_path = write_video(frames, out_path=out_path, fps=fps)

        saved_npz = ""
        if save_actions:
            npz_path = Path(out_dir) / f"{env_alias}_seed{seed}_ep{ep}_actions_{chosen_gl}.npz"
            np.savez_compressed(
                npz_path,
                actions=actions,
                times=times,
                agent_ids=np.asarray(agent_ids, dtype=object),
                dt=np.float32(DT_FIXED),
                horizon_sec=np.float32(H_TEST_SEC),
                mujoco_gl=np.asarray(chosen_gl),
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
        queue.put(
            {
                "ok": False,
                "error": str(e),
                "gl_backend": gl_backend,
            }
        )


def run_rollout_job_in_subprocess(
    env_alias: str,
    scenario: str,
    agent_conf: str,
    agent_obsk: int,
    seed: int,
    ep: int,
    out_dir: Path,
    fps: int,
    save_actions: bool,
    gl_backend: str,
    timeout_sec: float,
):
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=rollout_job_worker,
        args=(
            q,
            env_alias,
            scenario,
            agent_conf,
            agent_obsk,
            seed,
            ep,
            str(out_dir),
            int(fps),
            bool(save_actions),
            str(gl_backend),
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
    env_aliases = resolve_envs(args)
    if len(env_aliases) == 0:
        raise SystemExit("No environments found.")

    out_dir = resolve_output_dir(args.output_dir)
    gl_candidates = candidate_gl_backends(args.mujoco_gl)
    print(
        f"Recording random-policy videos: envs={env_aliases}, "
        f"Htest={H_TEST_SEC}, dt={DT_FIXED}, T={T_STEPS}, fps={args.fps}, "
        f"MUJOCO_GL_candidates={gl_candidates}, DISPLAY={os.environ.get('DISPLAY', '')!r}, out={out_dir}"
    )

    total = 0
    for env_alias in env_aliases:
        if env_alias not in MAMUJOCO_ENV_SPECS:
            print(f"[warn] unknown env alias, skip: {env_alias}")
            continue
        scenario, agent_conf = MAMUJOCO_ENV_SPECS[env_alias]

        for ep in range(int(args.episodes)):
            seed = int(args.seed) + ep
            success = False
            for gl_backend in gl_candidates:
                res = run_rollout_job_in_subprocess(
                    env_alias=env_alias,
                    scenario=scenario,
                    agent_conf=agent_conf,
                    agent_obsk=int(args.agent_obsk),
                    seed=seed,
                    ep=ep,
                    out_dir=out_dir,
                    fps=int(args.fps),
                    save_actions=bool(args.save_actions),
                    gl_backend=gl_backend,
                    timeout_sec=float(args.worker_timeout_sec),
                )
                if bool(res.get("ok", False)):
                    if args.save_actions and str(res.get("npz_path", "")):
                        print(
                            f"[ok] env={env_alias} ep={ep} gl={res.get('gl_backend')} "
                            f"saved: {res.get('video_path')} and {res.get('npz_path')}"
                        )
                    else:
                        print(
                            f"[ok] env={env_alias} ep={ep} gl={res.get('gl_backend')} "
                            f"saved: {res.get('video_path')}"
                        )
                    total += 1
                    success = True
                    break
                print(
                    f"[warn] env={env_alias} ep={ep} gl={gl_backend} failed: "
                    f"{res.get('error', 'unknown error')}"
                )
            if not success:
                print(
                    f"[warn] env={env_alias} ep={ep} failed on all backends: {gl_candidates}"
                )

    if total == 0:
        raise SystemExit("No videos generated.")
    print(f"Done. Generated {total} video file(s) in {out_dir}")


if __name__ == "__main__":
    main()
