import math
from typing import Dict

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from .drivor_agent import DrivoRAgent


class DrivoRScorerBaselineAgent(DrivoRAgent):
    """
    Supervised BCE baseline for GRPO comparison.

    Loads backbone + trajectory generator from a pretrained checkpoint and
    freezes them. Re-initialises scorer (scorer_attention, pos_embed, and
    the 6 BCE heads) from scratch, then trains only those modules with the
    standard DrivoR BCE loss.

    This isolates the contribution of GRPO vs plain supervised training when
    both start from the same frozen backbone.
    """

    def __init__(self, checkpoint_path: str = "", **kwargs):
        # Pass checkpoint_path="" to DrivoRAgent so it sets up training infra
        # (metric cache, loss, Ray worker). Store the real path for initialize().
        self._baseline_checkpoint_path = checkpoint_path
        super().__init__(checkpoint_path="", **kwargs)

    @staticmethod
    def _reset_weights(module: nn.Module) -> None:
        """Re-initialise all Linear/LayerNorm weights in a module."""
        for m in module.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def initialize(self) -> None:
        """Load checkpoint for backbone+generator; train scorer from scratch with BCE."""
        if self._baseline_checkpoint_path:
            map_loc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(self._baseline_checkpoint_path, map_location=map_loc)["state_dict"]
            self.load_state_dict(
                {k.replace("agent._drivor_model", "_drivor_model"): v for k, v in state_dict.items()},
                strict=False,
            )

        # re-initialise scorer modules so they truly start from scratch
        self._reset_weights(self._drivor_model.scorer_attention)
        self._reset_weights(self._drivor_model.pos_embed)
        self._reset_weights(self._drivor_model.scorer)

        # freeze everything first
        for param in self._drivor_model.parameters():
            param.requires_grad_(False)

        # unfreeze scorer_attention, pos_embed, and the full scorer (6 BCE heads)
        for param in self._drivor_model.scorer_attention.parameters():
            param.requires_grad_(True)
        for param in self._drivor_model.pos_embed.parameters():
            param.requires_grad_(True)
        for param in self._drivor_model.scorer.parameters():
            param.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[DrivoRScorerBaselineAgent] trainable params: {trainable:,} / {total:,}")

    # compute_loss() is inherited from DrivoRAgent — uses standard BCE loss

    def get_optimizers(self):
        lr = self._lr_args["base_lr"]
        trainable_params = (
            list(self._drivor_model.scorer_attention.parameters())
            + list(self._drivor_model.pos_embed.parameters())
            + list(self._drivor_model.scorer.parameters())
        )
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        if self.scheduler_args is not None:
            global_batchsize = self.batch_size * self.num_gpus
            T_max = int(math.ceil(self.scheduler_args.dataset_size / global_batchsize) * self.scheduler_args.num_epochs)
            T_max_ramp = int(T_max * 0.1)
            scheduler_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=T_max_ramp)
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max - T_max_ramp, eta_min=0.0)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[scheduler_ramp, scheduler_cosine], milestones=[T_max_ramp]
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return [optimizer]

    def get_training_callbacks(self):
        checkpoint_cb_best = ModelCheckpoint(
            save_top_k=1, monitor='val/score_epoch', filename='best-{epoch}-{step}', mode="max"
        )
        checkpoint_cb = ModelCheckpoint(save_last=True, every_n_train_steps=600)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [checkpoint_cb_best, checkpoint_cb, lr_monitor]
