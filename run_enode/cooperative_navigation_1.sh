#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_ID=1
SEEDS=(113 114)
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

for SEED in "${SEEDS[@]}"; do
  python runner_coop_ma_enode.py \
    --env cooperative_navigation \
    --rounds 100 \
    --dt 0.1 \
    --solver rk4 \
    --h_data 2.5 \
    --h_train 2.5 \
    --L 20 \
    --n_ens 5 \
    --consensus_weight 0.02 \
    --device "cuda:${GPU_ID}" \
    --episode_length 25 \
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
    --wandb_group cooperative-navigation-enode \
    --wandb_name "cooperative-navigation-enode-gpu${GPU_ID}-seed${SEED}-${RUN_TS}" \
    --wandb_mode online \
    --wandb_tags enode,mpe,cooperative_navigation \
    --wandb_entity "" \
    2>&1 | tee "logs/cooperative_navigation_enode_gpu${GPU_ID}_seed${SEED}_${RUN_TS}.log"
done
