#!/bin/bash
# Debug script — run interactively on a GPU node (not via sbatch).
# Usage: salloc --gres=gpu:a100l:1 -c 16 --mem=128G --time=01:00:00 --partition=short-unkillable
#        bash debug_drivor_mila.sh

export NAVSIM_DEVKIT_ROOT=/network/scratch/d/deschaer/DrivoR
export OPENSCENE_DATA_ROOT=/network/scratch/g/grandhia/navsim_data/drivor_dataset
export NUPLAN_MAPS_ROOT=/network/scratch/g/grandhia/navsim_data/drivor_dataset/maps
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT=$NAVSIM_DEVKIT_ROOT/exp
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled  # no wandb in debug

PYTHON=/network/scratch/d/deschaer/envs/drivoR/bin/python

mkdir -p $NAVSIM_EXP_ROOT

RUN_NAME=debug
OUTPUT_DIR=$NAVSIM_EXP_ROOT/$RUN_NAME
mkdir -p $OUTPUT_DIR

$PYTHON $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_grpo_finetuning.py \
    --config-name default_grpo_option3_training \
    agent.checkpoint_path=$NAVSIM_DEVKIT_ROOT/weights/checkpoints/drivor_Nav2_10epochs.pth \
    experiment_name=$RUN_NAME \
    output_dir=$OUTPUT_DIR \
    train_test_split=navtrain \
    use_cache_without_dataset=false \
    cache_path=null \
    trainer.params.max_epochs=1 \
    trainer.params.precision=32 \
    trainer.params.devices=1 \
    trainer.params.strategy=auto \
    dataloader.params.batch_size=4 \
    dataloader.params.num_workers=2 \
    dataloader.params.prefetch_factor=2 \
    agent.num_gpus=1 \
    agent.progress_bar=true \
    trainer.params.log_every_n_steps=1 \
    agent.grpo_loss.eps=0.2 \
    agent.grpo_loss.entropy_coeff=0.1 \
    agent.grpo_loss.use_constraints=true \
    agent.grpo_loss.advantage_method=scalarize_advantages \
    'agent.grpo_loss.constraint_names=[no_at_fault_collisions,drivable_area_compliance,ego_progress]' \
    'agent.grpo_loss.constraint_thresholds=[0.01,0.01,0.30]' \
    agent.grpo_loss.multiplier_lr=0.03 \
    resume_wandb=true
