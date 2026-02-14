#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_ID=2
SEEDS=(111 112)
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
# ENODE parity settings used below:
# rounds=100, exploration_steps=1000, new_episodes_per_round=4
ENODE_ROUNDS=100
ENODE_EXPLORATION_STEPS=1000
ENODE_NEW_EPISODES_PER_ROUND=4
EPISODE_LENGTH=50
DT=0.05
ROLLOUT_THREADS=10
EVAL_INTERVAL_EPISODES=${ENODE_NEW_EPISODES_PER_ROUND}
ENODE_INIT_EPISODES=$(( (ENODE_EXPLORATION_STEPS + EPISODE_LENGTH - 1) / EPISODE_LENGTH ))
ENODE_TOTAL_EPISODES=$(( ENODE_INIT_EPISODES + ENODE_ROUNDS * ENODE_NEW_EPISODES_PER_ROUND ))

for SEED in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python runner_mappo.py \
    --env walker \
    --algorithm_name mappo \
    --episodes "${ENODE_TOTAL_EPISODES}" \
    --episode_length "${EPISODE_LENGTH}" \
    --dt "${DT}" \
    --solver rk4 \
    --consensus_weight 0.02 \
    --device "cuda:0" \
    --n_rollout_threads "${ROLLOUT_THREADS}" \
    --env_step_workers "${ROLLOUT_THREADS}" \
    --collect_parallel_workers "${ROLLOUT_THREADS}" \
    --eval_every_episodes "${EVAL_INTERVAL_EPISODES}" \
    --eval_episodes 10 \
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
    --wandb_group walker-mappo \
    --wandb_name "walker-mappo-gpu${GPU_ID}-seed${SEED}-${RUN_TS}" \
    --wandb_mode online \
    --wandb_tags mappo,walker \
    --wandb_entity "" \
    2>&1 | tee "logs/walker_mappo_gpu${GPU_ID}_seed${SEED}_${RUN_TS}.log"
done
