import copy
import math
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .drivor_agent import DrivoRAgent
from .layers.losses.grpo_loss import GRPOLoss
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


class RefPolicyUpdateCallback(pl.Callback):
    """
    Updates π_old (the reference policy used for PPO clipping) every
    `update_interval` optimizer steps.

    With 1 gradient update per batch, π_old must lag behind the current
    policy by at least 1 step for the clip ratio to be non-trivial.
    Updating every ~200 steps keeps the policy drift within a range where
    eps=0.2 is still a meaningful constraint (clip_fraction ~10-30%).
    """

    def __init__(self, agent: "DrivoRGRPOOption3Agent", update_interval: int = 200):
        self.agent = agent
        self.update_interval = update_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # global_step counts optimizer steps across all epochs
        if (trainer.global_step + 1) % self.update_interval == 0:
            self.agent._update_reference()


class DrivoRGRPOOption3Agent(DrivoRAgent):
    """
    Option 3: single selection_head trained from scratch with GRPO.

    Loads image_backbone + trajectory_decoder (+ associated components) from a
    pretrained checkpoint and freezes them. scorer_attention, pos_embed, and
    selection_head are re-initialised randomly and trained end-to-end with GRPO.

    The 6 BCE heads are present but frozen and unused during this fine-tuning.

    π_old (reference for PPO clipping) is initialised to the random init and
    refreshed every `ref_update_interval` optimizer steps so that clip ratios
    stay in a meaningful range throughout training (see RefPolicyUpdateCallback).
    """

    def __init__(self, grpo_loss: nn.Module, checkpoint_path: str = "",
                 ref_update_interval: int = 200, **kwargs):
        # Pass checkpoint_path="" to DrivoRAgent so it sets up training infra
        # (metric cache, loss, Ray worker). Store the real path for initialize().
        self._grpo_checkpoint_path = checkpoint_path
        self.ref_update_interval = ref_update_interval
        super().__init__(checkpoint_path="", **kwargs)
        self.grpo_loss_fn = grpo_loss

    @staticmethod
    def _reset_weights(module: nn.Module) -> None:
        """Re-initialise all Linear/LayerNorm weights in a module."""
        for m in module.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def initialize(self) -> None:
        """Load checkpoint for perception+generator; train scorer from scratch."""
        # Load pretrained weights directly (super().initialize() uses self._checkpoint_path
        # which was set to "" to allow training infra setup in __init__)
        if self._grpo_checkpoint_path:
            map_loc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(self._grpo_checkpoint_path, map_location=map_loc)["state_dict"]
            self.load_state_dict(
                {k.replace("agent._drivor_model", "_drivor_model"): v for k, v in state_dict.items()},
                strict=False,
            )

        # re-initialise scorer modules so they truly start from scratch
        self._reset_weights(self._drivor_model.scorer_attention)
        self._reset_weights(self._drivor_model.pos_embed)
        self._reset_weights(self._drivor_model.scorer.selection_head)

        # freeze everything first
        for param in self._drivor_model.parameters():
            param.requires_grad_(False)

        # unfreeze scorer_attention, pos_embed, and selection_head
        for param in self._drivor_model.scorer_attention.parameters():
            param.requires_grad_(True)
        for param in self._drivor_model.pos_embed.parameters():
            param.requires_grad_(True)
        for param in self._drivor_model.scorer.selection_head.parameters():
            param.requires_grad_(True)

        # π_old = clone of the randomly-initialised scorer modules;
        # refreshed every ref_update_interval steps by RefPolicyUpdateCallback
        self.ref_scorer_attention = copy.deepcopy(self._drivor_model.scorer_attention)
        self.ref_pos_embed = copy.deepcopy(self._drivor_model.pos_embed)
        self.ref_selection_head = copy.deepcopy(self._drivor_model.scorer.selection_head)
        for ref_module in [self.ref_scorer_attention, self.ref_pos_embed, self.ref_selection_head]:
            for param in ref_module.parameters():
                param.requires_grad_(False)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[DrivoRGRPOOption3Agent] trainable params: {trainable:,} / {total:,}")

    def _update_reference(self) -> None:
        """Copy current policy weights into π_old (reference modules)."""
        for ref_p, p in zip(self.ref_scorer_attention.parameters(),
                            self._drivor_model.scorer_attention.parameters()):
            ref_p.data.copy_(p.data)
        for ref_p, p in zip(self.ref_pos_embed.parameters(),
                            self._drivor_model.pos_embed.parameters()):
            ref_p.data.copy_(p.data)
        for ref_p, p in zip(self.ref_selection_head.parameters(),
                            self._drivor_model.scorer.selection_head.parameters()):
            ref_p.data.copy_(p.data)

    def _ref_selection_logit(self, proposals: torch.Tensor, scene_features: torch.Tensor, ego_token: torch.Tensor) -> torch.Tensor:
        """Compute selection logits under π_old (frozen reference policy)."""
        B, N, _, _ = proposals.shape
        embedded_traj = self.ref_pos_embed(proposals.reshape(B, N, -1))  # (B, N, d_model)
        tr_out_ref = self.ref_scorer_attention(embedded_traj, scene_features)  # (B, N, d_model)
        tr_out_ref = tr_out_ref + ego_token
        return self.ref_selection_head(tr_out_ref).squeeze(-1)  # (B, N)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
    ) -> Dict:
        selection_logit = pred["selection_logit"]  # [B, K]

        # reference logits from π_old (updated every ref_update_interval steps)
        with torch.no_grad():
            ref_logit = self._ref_selection_logit(
                pred["proposals"].detach(),
                pred["scene_features"],
                pred["ego_token"],
            )  # [B, K]

        # oracle PDMS rewards for all K proposals
        final_scores, best_scores, target_scores, _, _, _ = self.compute_score(
            targets, pred["proposals"], test=False
        )
        oracle_rewards = final_scores  # [B, K]

        grpo_loss, grpo_loss_dict = self.grpo_loss_fn(selection_logit, ref_logit, oracle_rewards)

        # monitor selection quality
        with torch.no_grad():
            top_proposals = torch.argmax(selection_logit, dim=1)
            selection_score = final_scores[np.arange(len(final_scores)), top_proposals.cpu().numpy()].mean()
            best_score = best_scores.mean()

        loss_dict = {
            "loss": grpo_loss,
            "score": selection_score,
            "best_score": best_score,
            **grpo_loss_dict,
        }

        return loss_dict

    def get_optimizers(self):
        lr = self._lr_args["base_lr"]
        trainable_params = (
            list(self._drivor_model.scorer_attention.parameters())
            + list(self._drivor_model.pos_embed.parameters())
            + list(self._drivor_model.scorer.selection_head.parameters())
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
        checkpoint_cb = ModelCheckpoint(save_last=True)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        ref_update_cb = RefPolicyUpdateCallback(self, update_interval=self.ref_update_interval)
        return [checkpoint_cb_best, checkpoint_cb, lr_monitor, ref_update_cb]
