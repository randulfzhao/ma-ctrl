#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_ID=4
SEEDS=(111 112)
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

for SEED in "${SEEDS[@]}"; do
  python runner_coop_ma_enode.py \
    --env ant4x2 \
    --rounds 100 \
    --dt 0.05 \
    --solver rk4 \
    --h_data 2.5 \
    --h_train 2.5 \
    --L 20 \
    --n_ens 5 \
    --consensus_weight 0.02 \
    --device "cuda:${GPU_ID}" \
    --episode_length 50 \
    --replay_buffer_size 20 \
    --dyn_max_episodes 20 \
    --discount_rho 0.95 \
    --soft_update_tau 0.001 \
    --actor_lr 0.0001 \
    --critic_lr 0.001 \
    --dyn_lr 0.001 \
    --dyn_grad_clip 10.0 \
    --dyn_batch_size 128 \
    --dyn_update_steps 200 \
    --dyn_window_steps 5 \
    --rew_lr 0.001 \
    --policy_batch_size 256 \
    --critic_updates 4 \
    --exploration_steps 1000 \
    --new_episodes_per_round 10 \
    --dyn_nrep 10 \
    --collect_parallel_workers 10 \
    --torch_num_threads 1 \
    --torch_num_interop_threads 1 \
    --model_save_interval 1000 \
    --seed "${SEED}" \
    --use_wandb \
    --wandb_project ma-ctrl \
    --wandb_group ant4x2-enode \
    --wandb_name "ant4x2-enode-gpu${GPU_ID}-seed${SEED}-${RUN_TS}" \
    --wandb_mode online \
    --wandb_tags enode,ant4x2 \
    --wandb_entity "" \
    2>&1 | tee "logs/ant4x2_enode_gpu${GPU_ID}_seed${SEED}_${RUN_TS}.log"
done
