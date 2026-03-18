#!/bin/bash
#SBATCH --job-name=drivor-train
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH -c 64
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

export NAVSIM_DEVKIT_ROOT=/network/scratch/d/deschaer/DrivoR
export OPENSCENE_DATA_ROOT=/network/scratch/g/grandhia/navsim_data/drivor_dataset
export NUPLAN_MAPS_ROOT=/network/scratch/g/grandhia/navsim_data/drivor_dataset/maps
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT=$NAVSIM_DEVKIT_ROOT/exp
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=$(cat ~/.wandb_api_key 2>/dev/null || echo "")

PYTHON=/network/scratch/d/deschaer/envs/drivoR/bin/python

mkdir -p $NAVSIM_EXP_ROOT

$PYTHON $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_grpo_finetuning.py \
    --config-name default_grpo_option3_training \
    agent.checkpoint_path=$NAVSIM_DEVKIT_ROOT/weights/checkpoints/drivor_Nav2_10epochs.pth \
    experiment_name=drivoR_grpo_option3 \
    train_test_split=navtrain \
    use_cache_without_dataset=false \
    cache_path=null \
    trainer.params.max_epochs=10 \
    trainer.params.precision=16-mixed \
    dataloader.params.batch_size=16 \
    dataloader.params.num_workers=8 \
    dataloader.params.prefetch_factor=2 \
    agent.num_gpus=4 \
    agent.progress_bar=false
