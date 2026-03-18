#!/bin/bash
#SBATCH --job-name=drivor-scratch
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

# Fixed output_dir so checkpoints accumulate in the same location across job restarts
OUTPUT_DIR=$NAVSIM_EXP_ROOT/ke/drivoR_scratch/run
mkdir -p $OUTPUT_DIR

# run_training.py has built-in auto-resume: searches for the latest ckpt under
# the parent of output_dir if train_ckpt_path is null (the default).

$PYTHON $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    experiment_name=drivoR_scratch \
    output_dir=$OUTPUT_DIR \
    train_test_split=navtrain \
    use_cache_without_dataset=false \
    cache_path=null \
    trainer.params.max_epochs=10 \
    trainer.params.precision=16-mixed \
    dataloader.params.batch_size=16 \
    dataloader.params.num_workers=8 \
    dataloader.params.prefetch_factor=2 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=2e-4 \
    agent.num_gpus=4 \
    agent.progress_bar=false
