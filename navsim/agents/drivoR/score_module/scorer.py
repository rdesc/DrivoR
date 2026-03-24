import torch.nn as nn
# from ..bevformer.transformer_decoder import MyTransformeDecoder,MLP
# from ..bevformer.transformer_decoder import MLP
from ..layers.utils.mlp import MLP
# from .map_head import MapHead


class Scorer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.b2d=config.b2d

        self.proposal_num=config.proposal_num
        self.score_num = 6

        # self.pred_score = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)


        self.pred_score = nn.ModuleDict({
            'no_at_fault_collisions': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'drivable_area_compliance':
                nn.Sequential(
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, 1),
                ),
            'time_to_collision_within_bound': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'ego_progress': nn.Sequential(
                nn.Linear(config.tf_d_model,config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'driving_direction_compliance': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            # 'lane_keeping': nn.Sequential(
            #     nn.Linear(d_model, d_ffn),
            #     nn.ReLU(),
            #     nn.Linear(d_ffn, 1),
            # ),
            # 'traffic_light_compliance': nn.Sequential(
            #     nn.Linear(config.tf_d_model, config.tf_d_ffn),
            #     nn.ReLU(),
            #     nn.Linear(config.tf_d_ffn, 1),
            # ),
            'comfort': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1)
            )
        })





        self.double_score=config.double_score

        if self.double_score:
            self.pred_score2 = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)

        self.agent_pred= config.agent_pred

        if self.agent_pred:
            if self.b2d:
                self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn, 2*6* 9)
            else:
                self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn,2* 40 * 9)

        self.area_pred=config.area_pred

        if self.area_pred:
            if config.one_token_per_traj:
                if self.b2d:
                    self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, config.num_poses*2)
                else:
                    self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, config.num_poses*5*2)
            else:
                if self.b2d:
                    self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, 2)
                else:
                    self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, 5*2)

        self.use_grpo_head = getattr(config, 'use_grpo_head', False)
        if self.use_grpo_head:
            self.grpo_head = nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_model),
                nn.ReLU(),
                nn.Linear(config.tf_d_model, 1),
            )

        # Option 3: single selection head replacing the 6 BCE heads entirely
        self.use_selection_head = getattr(config, 'use_selection_head', False)
        if self.use_selection_head:
            self.selection_head = nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_model),
                nn.ReLU(),
                # nn.LayerNorm(config.tf_d_model),  # should we keep this?
                nn.Linear(config.tf_d_model, 1),
            )

        self.bev_map=config.bev_map
        self.bev_agent=config.bev_agent

        if config.bev_agent:
            raise NotImplementedError
            self._agent_head=MyTransformeDecoder(config,config.num_bounding_boxes,6,trajenc=False)

        if config.bev_map:
            raise NotImplementedError
            self.map_head=MapHead(config)

        self.poses_num=config.num_poses
        self.tf_d_model=config.tf_d_model
        self.one_token_per_traj = config.one_token_per_traj


    def forward(self, proposals,bev_feature):
        batch_size=len(proposals)
        p_size=proposals.shape[1]
        t_size=proposals.shape[2]

        proposal_feature = bev_feature
        pred_logit = {}

        # selected_indices: B,
        for k, head in self.pred_score.items():
            pred_logit[k] = head(proposal_feature).squeeze(-1)

        grpo_logit = self.grpo_head(proposal_feature).squeeze(-1) if self.use_grpo_head else None
        selection_logit = self.selection_head(proposal_feature).squeeze(-1) if self.use_selection_head else None
        pred_logit2=pred_agents_states=pred_area_logit=bev_semantic_map=agent_states=agent_labels=None

        if self.double_score:
            pred_logit2 = self.pred_score2(proposal_feature).reshape(batch_size, -1, self.score_num)

        if  self.training:
            if self.area_pred:
                pred_area_logit = self.pred_area(bev_feature)
                if self.one_token_per_traj:
                    pred_area_logit = pred_area_logit.reshape(batch_size,p_size,t_size,-1)

            if self.agent_pred:
                pred_agents_states = self.pred_col_agent(proposal_feature).reshape(batch_size,p_size,t_size,-1,2,9)

            if self.bev_map:
                bev_semantic_map = self.map_head(bev_feature)

            if self.bev_agent:
                agents =self._agent_head(None,bev_feature)
                agent_states = agents[:, :, :-1]
                agent_labels = agents[:, :, -1]

        return (
            pred_logit,         # dict[str, [B, K]]     — 6 BCE metric logits (always populated)
            pred_logit2,        # [B, K, 6] or None     — double-score variant (double_score=True)
            pred_agents_states, # [B,K,T,N,2,9] or None — agent collision predictions (agent_pred=True, train only)
            pred_area_logit,    # [B,K,T,...] or None   — drivable area predictions (area_pred=True, train only)
            bev_semantic_map,   # tensor or None        — BEV semantic map (bev_map=True, train only)
            agent_states,       # tensor or None        — agent states from bev decoder (bev_agent=True, train only)
            agent_labels,       # tensor or None        — agent labels from bev decoder (bev_agent=True, train only)
            grpo_logit,         # [B, K] or None        — Option 2 dedicated GRPO head logits (use_grpo_head=True)
            selection_logit,    # [B, K] or None        — Option 3 single selection head logits (use_selection_head=True)
        )
