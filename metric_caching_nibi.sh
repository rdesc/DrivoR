#!/bin/bash
#SBATCH --job-name=navsim-metric-cache
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

export NAVSIM_DEVKIT_ROOT=/home/rdesc/scratch/DrivoR

source $NAVSIM_DEVKIT_ROOT/.venv/bin/activate
source $NAVSIM_DEVKIT_ROOT/extract_navsim_nibi.sh $NAVSIM_DEVKIT_ROOT

export NUPLAN_MAPS_ROOT=$NAVSIM_DEVKIT_ROOT/download/maps
export HYDRA_FULL_ERROR=1

TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_train_metric_caching.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    gpu=false
