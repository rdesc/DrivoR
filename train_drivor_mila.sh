#!/bin/bash
#SBATCH --job-name=drivor-train
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH -c 64
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

export NAVSIM_DEVKIT_ROOT=/network/scratch/d/deschaer/DrivoR
export OPENSCENE_DATA_ROOT=/network/scratch/g/grandhia/navsim_data/drivor_dataset
export NUPLAN_MAPS_ROOT=/network/scratch/g/grandhia/navsim_data/drivor_dataset/maps
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT=$NAVSIM_DEVKIT_ROOT/exp
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=$(cat ~/.wandb_api_key 2>/dev/null || echo "")

PYTHON=/network/scratch/d/deschaer/envs/drivoR/bin/python

mkdir -p $NAVSIM_EXP_ROOT

# Use a fixed output_dir so checkpoints are always written to the same location across job restarts.
# Each new run with a timestamp would create a fresh directory, breaking resume.
OUTPUT_DIR=$NAVSIM_EXP_ROOT/ke/drivoR_grpo_option3/run
mkdir -p $OUTPUT_DIR

# Resume from last checkpoint if one exists (supports multi-job runs on 3-hour short-unkillable).
# Lightning saves to {output_dir}/lightning_logs/version_N/checkpoints/last.ckpt where N increments
# each restart, so we find the most recently modified last.ckpt anywhere under OUTPUT_DIR.
LAST_CKPT=$(find $OUTPUT_DIR -name "last.ckpt" -type f 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
RESUME_ARG=""
if [ -n "$LAST_CKPT" ]; then
    echo "Resuming from $LAST_CKPT"
    RESUME_ARG="train_ckpt_path=$LAST_CKPT"
fi

$PYTHON $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_grpo_finetuning.py \
    --config-name default_grpo_option3_training \
    agent.checkpoint_path=$NAVSIM_DEVKIT_ROOT/weights/checkpoints/drivor_Nav2_10epochs.pth \
    experiment_name=drivoR_grpo_PPOclip_1step_pi_old_eps_0.2_bs_32 \
    output_dir=$OUTPUT_DIR \
    train_test_split=navtrain \
    use_cache_without_dataset=false \
    cache_path=null \
    trainer.params.max_epochs=10 \
    trainer.params.precision=16-mixed \
    dataloader.params.batch_size=32 \
    dataloader.params.num_workers=8 \
    dataloader.params.prefetch_factor=2 \
    agent.num_gpus=4 \
    agent.progress_bar=false \
    agent.grpo_loss.entropy_coeff=0.1 \
    agent.grpo_loss.eps=0.2 \
    $RESUME_ARG
