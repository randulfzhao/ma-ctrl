#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_ID=5
SEEDS=(113 114)
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
# ENODE parity notes (not consumed by runner_mappo.py):
# replay_buffer_size=104, soft_update_tau=0.001
# dyn_lr=0.001, rew_lr=0.001, exploration_steps=1000


for SEED in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python runner_mappo.py \
    --env ant4x2 \
    --algorithm_name mappo \
    --episodes 10000 \
    --episode_length 50 \
    --dt 0.05 \
    --solver rk4 \
    --consensus_weight 0.02 \
    --device "cuda:0" \
    --n_rollout_threads 16 \
    --env_step_workers 16 \
    --collect_parallel_workers 10 \
    --eval_every_episodes 200 \
    --eval_episodes 32 \
    --eval_env_workers 8 \
    --num_mini_batch 4 \
    --ppo_epoch 10 \
    --lr 0.0001 \
    --critic_lr 0.001 \
    --gamma 0.95 \
    --n_training_threads 1 \
    --save_interval_episodes 1000 \
    --seed "${SEED}" \
    --use_wandb \
    --wandb_project ma-ctrl \
    --wandb_group ant4x2-mappo \
    --wandb_name "ant4x2-mappo-gpu${GPU_ID}-seed${SEED}-${RUN_TS}" \
    --wandb_mode online \
    --wandb_tags mappo,ant4x2 \
    --wandb_entity "" \
    2>&1 | tee "logs/ant4x2_mappo_gpu${GPU_ID}_seed${SEED}_${RUN_TS}.log"
done
