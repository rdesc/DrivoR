#!/bin/bash
#SBATCH --job-name=navsim-metric-cache
#SBATCH --partition=cpubase_bycore_b3
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

source /scratch/rdesc/DrivoR/.venv/bin/activate

source /project/6061241/rdesc/extract_navsim.sh /home/rdesc/scratch/DrivoR

export NUPLAN_MAPS_ROOT=/scratch/rdesc/DrivoR/download/maps
export HYDRA_FULL_ERROR=1

TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_train_metric_caching.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    gpu=false
