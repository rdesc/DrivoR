#!/bin/bash
#SBATCH --job-name=drivor-train
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load cuda/12.2

export NAVSIM_DEVKIT_ROOT=/home/rdesc/scratch/DrivoR

source $NAVSIM_DEVKIT_ROOT/.venv/bin/activate
source $NAVSIM_DEVKIT_ROOT/extract_navsim_nibi.sh $NAVSIM_DEVKIT_ROOT

export NUPLAN_MAPS_ROOT=$NAVSIM_DEVKIT_ROOT/download/maps
export HYDRA_FULL_ERROR=1

EXPERIMENT=training_drivoR_Nav1_traj_long_25epochs
AGENT=drivoR
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=25 \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=16 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=4 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    agent.config.long_trajectory_additional_poses=2 \
    seed=2
