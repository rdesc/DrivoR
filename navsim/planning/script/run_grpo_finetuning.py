"""
GRPO fine-tuning script for the DrivoR scorer.

Supports both GRPO design options:
  Option 2 (dedicated scalar head):
    python navsim/planning/script/run_grpo_finetuning.py \
        agent=drivoR_grpo \
        agent.checkpoint_path=/path/to/drivoR_checkpoint.ckpt \
        train_test_split=navtrain

  Option 3 (single selection head, trained from scratch):
    python navsim/planning/script/run_grpo_finetuning.py \
        --config-name default_grpo_option3_training \
        agent.checkpoint_path=/path/to/drivoR_checkpoint.ckpt \
        train_test_split=navtrain
"""
import logging
import os

from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_grpo_training"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    logger.info("Building GRPO Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    # initialize() loads the pretrained checkpoint and sets up frozen reference modules.
    # Must be called explicitly here — the training loop (pl.Trainer) does not call it.
    agent.initialize()

    lightning_module = AgentLightningModule(agent=agent)

    if cfg.use_cache_without_dataset:
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
        train_scene_filter.log_names = [
            n for n in (train_scene_filter.log_names or cfg.train_logs)
            if n in cfg.train_logs
        ] or cfg.train_logs

        val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
        val_scene_filter.log_names = [
            n for n in (val_scene_filter.log_names or cfg.val_logs)
            if n in cfg.val_logs
        ] or cfg.val_logs

        data_path = Path(cfg.navsim_log_path)
        sensor_blobs_path = Path(cfg.sensor_blobs_path)

        train_data = Dataset(
            scene_loader=SceneLoader(
                sensor_blobs_path=sensor_blobs_path,
                data_path=data_path,
                scene_filter=train_scene_filter,
                sensor_config=agent.get_sensor_config(),
            ),
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            cache_path=cfg.cache_path,
            force_cache_computation=cfg.force_cache_computation,
        )
        val_data = Dataset(
            scene_loader=SceneLoader(
                sensor_blobs_path=sensor_blobs_path,
                data_path=data_path,
                scene_filter=val_scene_filter,
                sensor_config=agent.get_sensor_config(),
            ),
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            cache_path=cfg.cache_path,
            force_cache_computation=cfg.force_cache_computation,
        )

    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False, drop_last=True)
    logger.info("Train samples: %d  |  Val samples: %d", len(train_data), len(val_data))

    wandb_logger = WandbLogger(
        project="drivoR",
        entity="rdesc1-milaquebec",
        name=cfg.experiment_name,
    )
    # Pin checkpoint dirpath to output_dir/checkpoints so it doesn't get
    # redirected to the wandb run directory (WandbLogger overrides trainer.log_dir).
    checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints")
    callbacks = agent.get_training_callbacks()
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            cb.dirpath = checkpoint_dir
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks, logger=wandb_logger)
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.get("train_ckpt_path", None),
    )


if __name__ == "__main__":
    main()
