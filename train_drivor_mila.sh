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
export WANDB_INIT_TIMEOUT=120
export WANDB_MODE=online

PYTHON=/network/scratch/d/deschaer/envs/drivoR/bin/python

mkdir -p $NAVSIM_EXP_ROOT

# ── Change RUN_NAME to start a fresh run (new dir + new WandB run). ──────────
RUN_NAME=drivoR_cgrpo_screw_nc_0.01_dac_0.01_ep_0.15_multiplier_lr_0.005_base_lr_1e-4_temp_1.5
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR=$NAVSIM_EXP_ROOT/$RUN_NAME
mkdir -p $OUTPUT_DIR

# Auto-resume is handled by run_grpo_finetuning.py (looks for last.ckpt in checkpoint_dir).

$PYTHON $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_grpo_finetuning.py \
    --config-name default_grpo_option3_training \
    agent.checkpoint_path=$NAVSIM_DEVKIT_ROOT/weights/checkpoints/drivor_Nav2_10epochs.pth \
    experiment_name=$RUN_NAME \
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
    trainer.params.log_every_n_steps=50 \
    agent.grpo_loss.eps=0.2 \
    agent.grpo_loss.entropy_coeff=0.1 \
    agent.grpo_loss.use_constraints=true \
    agent.grpo_loss.advantage_method=scalarize_rewards \
    'agent.grpo_loss.constraint_names=[no_at_fault_collisions,drivable_area_compliance,ego_progress]' \
    'agent.grpo_loss.constraint_thresholds=[0.01,0.01,0.15]' \
    agent.grpo_loss.multiplier_lr=0.005 \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    agent.grpo_loss.multiplier_temperature=1.5
    # ── previous runs (commented out) ──
    # RUN_NAME=drivoR_constrained_grpo_smoke_test (multiplier update only, no scalarized advantages)
    # RUN_NAME=drivoR_grpo_entropy_0.1_no_clipping
    # agent.grpo_loss.multiplier_temperature=1.5 \
