#!/usr/bin/env bash

GPU_ID=1
RUN_TS="$(date +%Y%m%d_%H%M%S)"

python runner_coop_ma_enode.py \
  --env ant2x4 \
  --episodes 10000 \
  --dt 0.05 \
  --solver rk4 \
  --h_data 2.5 \
  --h_train 2.5 \
  --L 20 \
  --n_ens 5 \
  --consensus_weight 0.02 \
  --device "cuda:${GPU_ID}" \
  --episode_length 50 \
  --replay_buffer_size 104 \
  --dyn_max_episodes 200 \
  --discount_rho 0.95 \
  --soft_update_tau 0.001 \
  --actor_lr 0.0001 \
  --critic_lr 0.001 \
  --dyn_lr 0.001 \
  --dyn_grad_clip 10.0 \
  --dyn_batch_size 1024 \
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
  --seed 111 \
  --use_wandb \
  --wandb_project ma-ctrl \
  --wandb_group ant2x4-enode \
  --wandb_name "ant2x4-enode-gpu${GPU_ID}-seed111-${RUN_TS}" \
  --wandb_mode online \
  --wandb_tags enode,ctde,multi-agent,ant2x4 \
  --wandb_entity "" \
  2>&1 | tee "logs/ant2x4_enode_gpu${GPU_ID}_seed111_${RUN_TS}.log"
