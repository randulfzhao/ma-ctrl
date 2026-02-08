#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

to_bool_01() {
  local v="${1:-}"
  local v_lower
  v_lower="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v_lower" in
    1|true|yes|y|on) echo "1" ;;
    0|false|no|n|off|"") echo "0" ;;
    *)
      echo "[ant.sh] ERROR: boolean env expects one of {1,0,true,false,yes,no,on,off}, got: '$v'" >&2
      exit 1
      ;;
  esac
}

normalize_wandb_mode() {
  local v="${1:-online}"
  v="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    online|offline|disabled) echo "$v" ;;
    *)
      echo "[ant.sh] ERROR: WANDB_MODE must be one of {online,offline,disabled}, got: '$v'" >&2
      exit 1
      ;;
  esac
}

normalize_launch_mode() {
  local v="${1:-auto}"
  v="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    auto|direct|conda_run) echo "$v" ;;
    *)
      echo "[ant.sh] ERROR: LAUNCH_MODE must be one of {auto,direct,conda_run}, got: '$v'" >&2
      exit 1
      ;;
  esac
}

has_wandb_login() {
  [[ -n "${WANDB_API_KEY:-}" ]] && return 0
  [[ -f "$HOME/.netrc" ]] || return 1
  grep -Eq 'machine[[:space:]]+api\.wandb\.ai' "$HOME/.netrc"
}

wandb_runtime_ok() {
  python - <<'PY'
import os
import socket
import sys

try:
    os.makedirs("logs/.wandb_runtime_check", exist_ok=True)
    with open("logs/.wandb_runtime_check/probe.log", "w", encoding="utf-8") as f:
        f.write("ok\n")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.close()
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

# -----------------------------------------------------------------------------
# Run configuration (override any of these via env vars)
# -----------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-mbrl}"
LAUNCH_MODE="$(normalize_launch_mode "${LAUNCH_MODE:-auto}")"
DEVICE="${DEVICE:-cuda:0}"
EPISODES="${EPISODES:-1000}"

DT="${DT:-0.05}"
EPISODE_LENGTH="${EPISODE_LENGTH:-50}"
SOLVER="${SOLVER:-rk4}"
# SOLVER="${SOLVER:-dopri5}"
HORIZON="$(python - <<PY
dt = $DT
ep_len = $EPISODE_LENGTH
print(dt * ep_len)
PY
)"
H_DATA="${H_DATA:-$HORIZON}"
H_TRAIN="${H_TRAIN:-$HORIZON}"

L_SAMPLES="${L_SAMPLES:-20}"
N_ENS="${N_ENS:-5}"
CONSENSUS_WEIGHT="${CONSENSUS_WEIGHT:-0.02}"

REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-104}"
DISCOUNT_RHO="${DISCOUNT_RHO:-0.95}"
SOFT_UPDATE_TAU="${SOFT_UPDATE_TAU:-0.001}"
ACTOR_LR="${ACTOR_LR:-0.0001}"
CRITIC_LR="${CRITIC_LR:-0.001}"
DYN_LR="${DYN_LR:-0.001}"
DYN_GRAD_CLIP="${DYN_GRAD_CLIP:-10.0}"
REW_LR="${REW_LR:-0.001}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-256}"
CRITIC_UPDATES="${CRITIC_UPDATES:-4}"
EXPLORATION_STEPS="${EXPLORATION_STEPS:-1000}"
MODEL_SAVE_INTERVAL="${MODEL_SAVE_INTERVAL:-1000}"

USE_WANDB="$(to_bool_01 "${USE_WANDB:-1}")"
WANDB_PROJECT="${WANDB_PROJECT:-ma-ctrl}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-ant2x4-enode}"
WANDB_MODE="$(normalize_wandb_mode "${WANDB_MODE:-online}")"
WANDB_TAGS="${WANDB_TAGS:-enode,ctde,multi-agent,ant2x4}"
WANDB_RUN_NAME_PREFIX="${WANDB_RUN_NAME_PREFIX:-ant2x4-enode}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_BASE_URL="${WANDB_BASE_URL:-}"

SEED_START="${SEED_START:-111}"
SEED_END="${SEED_END:-120}"

# Runtime env
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p logs

ACTIVE_CONDA_ENV="${CONDA_DEFAULT_ENV:-}"
RESOLVED_LAUNCH_MODE="$LAUNCH_MODE"
if [[ "$RESOLVED_LAUNCH_MODE" == "auto" ]]; then
  if [[ -n "$ACTIVE_CONDA_ENV" && "$ACTIVE_CONDA_ENV" == "$CONDA_ENV" ]]; then
    RESOLVED_LAUNCH_MODE="direct"
  else
    RESOLVED_LAUNCH_MODE="conda_run"
  fi
fi

if [[ "$RESOLVED_LAUNCH_MODE" == "direct" && -n "$CONDA_ENV" && -n "$ACTIVE_CONDA_ENV" && "$ACTIVE_CONDA_ENV" != "$CONDA_ENV" ]]; then
  echo "[ant.sh] WARNING: launch_mode=direct but active conda env is '$ACTIVE_CONDA_ENV' (expected '$CONDA_ENV')." >&2
fi

if [[ "$USE_WANDB" == "1" ]]; then
  if [[ "$WANDB_MODE" == "online" ]] && ! has_wandb_login; then
    cat <<'EOF' >&2
[ant.sh] ERROR: WANDB online mode requested but no API auth found.
Set WANDB_API_KEY in env, or run `wandb login`, or set WANDB_MODE=offline.
EOF
    exit 1
  fi
  if [[ -n "$WANDB_API_KEY" ]]; then
    export WANDB_API_KEY
  fi
  if [[ -n "$WANDB_BASE_URL" ]]; then
    export WANDB_BASE_URL
  fi
  if ! wandb_runtime_ok; then
    echo "[ant.sh] WARNING: wandb runtime checks failed (socket/filesystem). Disabling wandb for this run." >&2
    USE_WANDB="0"
  fi
fi

# If CUDA is requested but unavailable on this host, fall back to CPU so runs can start.
if [[ "$DEVICE" == cuda* ]]; then
  if ! python - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "[ant.sh] WARNING: DEVICE=$DEVICE requested, but CUDA is not available. Falling back to cpu." >&2
    DEVICE="cpu"
  fi
fi

print_config() {
  cat <<EOF
[ant.sh] Config
  conda_env           = $CONDA_ENV
  active_conda_env    = ${ACTIVE_CONDA_ENV:-<none>}
  launch_mode         = $LAUNCH_MODE -> $RESOLVED_LAUNCH_MODE
  device              = $DEVICE
  episodes            = $EPISODES
  dt                  = $DT
  episode_length      = $EPISODE_LENGTH
  solver              = $SOLVER
  horizon_seconds     = $HORIZON
  h_data / h_train    = $H_DATA / $H_TRAIN
  replay_buffer_size  = $REPLAY_BUFFER_SIZE
  discount_rho        = $DISCOUNT_RHO
  soft_update_tau     = $SOFT_UPDATE_TAU
  actor_lr / critic_lr= $ACTOR_LR / $CRITIC_LR
  dyn_lr / rew_lr     = $DYN_LR / $REW_LR
  dyn_grad_clip       = $DYN_GRAD_CLIP
  policy_batch_size   = $POLICY_BATCH_SIZE
  critic_updates      = $CRITIC_UPDATES
  exploration_steps   = $EXPLORATION_STEPS
  model_save_interval = $MODEL_SAVE_INTERVAL
  use_wandb           = $USE_WANDB
  wandb_project       = $WANDB_PROJECT
  wandb_entity        = ${WANDB_ENTITY:-<empty>}
  wandb_group         = $WANDB_GROUP
  wandb_mode          = $WANDB_MODE
  wandb_tags          = $WANDB_TAGS
  wandb_name_prefix   = $WANDB_RUN_NAME_PREFIX
  wandb_api_auth      = $(if has_wandb_login; then echo "configured"; else echo "missing"; fi)
  wandb_base_url      = ${WANDB_BASE_URL:-<default>}
  seeds               = $SEED_START .. $SEED_END
EOF
}

run_one_seed() {
  local seed="$1"
  local ts log_file wandb_name
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="logs/ant2x4_enode_seed${seed}_${ts}.log"
  wandb_name="${WANDB_RUN_NAME_PREFIX}-seed${seed}-${ts}"

  local -a cmd_prefix=()
  if [[ "$RESOLVED_LAUNCH_MODE" == "direct" ]]; then
    cmd_prefix=(python)
  else
    cmd_prefix=(conda run -n "$CONDA_ENV" python)
  fi

  local -a cmd=(
    "${cmd_prefix[@]}" runner_coop_ma_enode.py
    --env ant2x4
    --episodes "$EPISODES"
    --dt "$DT"
    --solver "$SOLVER"
    --h_data "$H_DATA"
    --h_train "$H_TRAIN"
    --L "$L_SAMPLES"
    --n_ens "$N_ENS"
    --consensus_weight "$CONSENSUS_WEIGHT"
    --device "$DEVICE"
    --episode_length "$EPISODE_LENGTH"
    --replay_buffer_size "$REPLAY_BUFFER_SIZE"
    --discount_rho "$DISCOUNT_RHO"
    --soft_update_tau "$SOFT_UPDATE_TAU"
    --actor_lr "$ACTOR_LR"
    --critic_lr "$CRITIC_LR"
    --dyn_lr "$DYN_LR"
    --dyn_grad_clip "$DYN_GRAD_CLIP"
    --rew_lr "$REW_LR"
    --policy_batch_size "$POLICY_BATCH_SIZE"
    --critic_updates "$CRITIC_UPDATES"
    --exploration_steps "$EXPLORATION_STEPS"
    --model_save_interval "$MODEL_SAVE_INTERVAL"
    --seed "$seed"
  )
  if [[ "$USE_WANDB" == "1" ]]; then
    cmd+=(
      --use_wandb
      --wandb_project "$WANDB_PROJECT"
      --wandb_group "$WANDB_GROUP"
      --wandb_name "$wandb_name"
      --wandb_mode "$WANDB_MODE"
      --wandb_tags "$WANDB_TAGS"
    )
    if [[ -n "$WANDB_ENTITY" ]]; then
      cmd+=(--wandb_entity "$WANDB_ENTITY")
    fi
  fi

  echo "[ant.sh] Running seed=$seed"
  echo "[ant.sh] Command: ${cmd[*]}"
  echo "[ant.sh] Log: $log_file"
  "${cmd[@]}" 2>&1 | tee "$log_file"
}

print_config
for seed in $(seq "$SEED_START" "$SEED_END"); do
  run_one_seed "$seed"
done
