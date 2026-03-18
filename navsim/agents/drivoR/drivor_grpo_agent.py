import copy
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .drivor_agent import DrivoRAgent
from .layers.losses.grpo_loss import GRPOLoss
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


class DrivoRGRPOAgent(DrivoRAgent):
    """
    Fine-tunes only the dedicated grpo_head (Option 2).

    Everything else — image backbone, trajectory decoder, trajectory heads,
    scorer_attention, pos_embed, and the 6 BCE heads — stays frozen.

    The reference policy is a frozen clone of grpo_head at initialisation,
    used for KL regularisation and PPO-style clipping.
    """

    def __init__(self, grpo_loss: nn.Module, checkpoint_path: str = "", **kwargs):
        # checkpoint_path="" is passed to DrivoRAgent so it sets up training infra
        # (metric cache, Ray worker, loss). The real checkpoint path is stored
        # separately and loaded manually in initialize(), which is called by
        # run_grpo_finetuning.py before training starts.
        self._grpo_checkpoint_path = checkpoint_path
        super().__init__(checkpoint_path="", **kwargs)
        self.grpo_loss_fn = grpo_loss

    def initialize(self) -> None:
        """Load checkpoint, then freeze all params except grpo_head."""
        if self._grpo_checkpoint_path:
            map_loc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(self._grpo_checkpoint_path, map_location=map_loc)["state_dict"]
            self.load_state_dict(
                {k.replace("agent._drivor_model", "_drivor_model"): v for k, v in state_dict.items()},
                strict=False,
            )

        # freeze everything
        for param in self._drivor_model.parameters():
            param.requires_grad_(False)

        # unfreeze only the grpo_head
        for param in self._drivor_model.scorer.grpo_head.parameters():
            param.requires_grad_(True)

        # create frozen reference as a clone of the initial grpo_head
        self.ref_grpo_head = copy.deepcopy(self._drivor_model.scorer.grpo_head)
        for param in self.ref_grpo_head.parameters():
            param.requires_grad_(False)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[DrivoRGRPOAgent] trainable params: {trainable:,} / {total:,}")

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
    ) -> Dict:
        grpo_logit = pred["grpo_logit"]       # [B, K]
        tr_out = pred["tr_out"]               # [B, K, d_model]

        # reference logits from frozen clone — no grad, same tr_out (scorer_attention frozen)
        with torch.no_grad():
            ref_grpo_logit = self.ref_grpo_head(tr_out).squeeze(-1)  # [B, K]

        # oracle PDMS rewards for all K proposals
        final_scores, best_scores, target_scores, _, _, _ = self.compute_score(
            targets, pred["proposals"], test=False
        )
        oracle_rewards = final_scores  # [B, K]

        grpo_loss, grpo_loss_dict = self.grpo_loss_fn(grpo_logit, ref_grpo_logit, oracle_rewards)

        # selection using grpo_logit for monitoring
        with torch.no_grad():
            top_proposals = torch.argmax(grpo_logit, dim=1)
            grpo_score = final_scores[np.arange(len(final_scores)), top_proposals.cpu().numpy()].mean()
            best_score = best_scores.mean()

        loss_dict = {
            "loss": grpo_loss,
            "score": grpo_score,
            "best_score": best_score,
            **grpo_loss_dict,
        }

        return loss_dict

    def get_optimizers(self):
        lr = self._lr_args["base_lr"]
        optimizer = torch.optim.AdamW(
            self._drivor_model.scorer.grpo_head.parameters(), lr=lr
        )
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
        checkpoint_cb = ModelCheckpoint(save_last=True)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [checkpoint_cb_best, checkpoint_cb, lr_monitor]
