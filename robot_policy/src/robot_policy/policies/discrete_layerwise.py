from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import PolicyBase
from .common import LayerwiseDiT, cosine_corruption, expected_token_distance, maskgit_update


class LayerwiseDiscretePolicy(PolicyBase):
    def __init__(self, cfg: Any):
        super().__init__(cfg)
        d = cfg.policy.hidden_dim
        self.mask_id = 256
        self.input_embedding = nn.Embedding(257, d)
        self.future_tokens = nn.Parameter(torch.randn(32, d) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(self.action_positions, d) * 0.02)
        self.backbone = LayerwiseDiT(d, cfg.policy.depth, cfg.policy.heads, cfg.policy.mlp_ratio, cfg.policy.dropout)
        self.output_head = nn.Linear(d, 256)

    def logits(self, tokens: torch.Tensor, obs) -> torch.Tensor:
        actions = self.input_embedding(tokens) + self.position_embedding[None]
        future = self.future_tokens[None].expand(len(tokens), -1, -1)
        hidden = torch.cat([future, actions], dim=1)
        hidden = self.backbone(hidden, obs.tokens, tokens.new_zeros(len(tokens)), obs.valid_mask)
        return self.output_head(hidden[:, -self.action_positions:])

    def loss(self, batch, rtc=None):
        target = batch["discrete_target"].long().flatten(1)
        valid = batch["control_valid_mask"].bool().flatten(1)
        corrupted, supervised = cosine_corruption(target, valid)
        if rtc is not None:
            fixed = rtc["fixed_mask"].bool().flatten(1)
            fixed_tokens = rtc["prefix_values"].long().flatten(1)
            corrupted = torch.where(fixed, fixed_tokens, corrupted)
            supervised &= ~fixed
        obs = self.observations(batch)
        logits = self.logits(corrupted, obs)
        ce, l1 = expected_token_distance(logits, target, supervised)
        total = ce + self.cfg.train.lambda_l1 * l1
        result = {"loss": total, "loss_ce": ce, "loss_l1": l1, "gradient_scale_ce": ce.detach(), "gradient_scale_l1": l1.detach()}
        result["action_mse"] = self.logits_action_mse(logits, batch, supervised).detach()
        return result

    @torch.no_grad()
    def sample(self, batch, rounds=None, prefix_values=None, fixed_mask=None, **kwargs):
        obs = self.observations(batch)
        b = len(obs.tokens); rounds = int(rounds or self.cfg.policy.discrete_rounds)
        tokens = torch.full((b, self.action_positions), self.mask_id, device=obs.tokens.device, dtype=torch.long)
        mutable = torch.ones_like(tokens, dtype=torch.bool)
        if fixed_mask is not None:
            fixed_mask = fixed_mask.flatten(1).bool(); prefix_values = prefix_values.flatten(1).long()
            tokens = torch.where(fixed_mask, prefix_values, tokens); mutable &= ~fixed_mask
        for step in range(rounds):
            tokens = maskgit_update(self.logits(tokens, obs), tokens, mutable, step, rounds)
            if fixed_mask is not None:
                tokens = torch.where(fixed_mask, prefix_values, tokens)
        return tokens.reshape(b, self.num_basis, self.action_dim)
