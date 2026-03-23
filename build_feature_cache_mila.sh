#!/bin/bash
#SBATCH --job-name=drivor-feature-cache
#SBATCH --partition=long-cpu
#SBATCH -c 32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

export NAVSIM_DEVKIT_ROOT=/network/scratch/d/deschaer/DrivoR
export OPENSCENE_DATA_ROOT=/network/scratch/g/grandhia/navsim_data/drivor_dataset
export NUPLAN_MAPS_ROOT=/network/scratch/g/grandhia/navsim_data/drivor_dataset/maps
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT=$NAVSIM_DEVKIT_ROOT/exp
export HYDRA_FULL_ERROR=1

PYTHON=/network/scratch/d/deschaer/envs/drivoR/bin/python

mkdir -p $NAVSIM_EXP_ROOT

$PYTHON $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=drivoR \
    experiment_name=drivoR_feature_cache \
    train_test_split=navtrain \
    cache_path=$NAVSIM_EXP_ROOT/navsim_cache_nommcv \
    worker.threads_per_node=4
