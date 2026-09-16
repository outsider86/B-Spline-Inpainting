from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .base import PolicyBase
from .common import expected_token_distance, maskgit_update, monotonic_block_corruption


class JointAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__(); self.heads = heads; self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim); self.out = nn.Linear(dim, dim); self.dropout = dropout

    def _split(self, x):
        return x.view(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[None, None], dropout_p=self.dropout if self.training else 0.0)
        return self.out(out.transpose(1, 2).contiguous().flatten(2))

    def project_kv(self, prefix_input: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        _,k,v=self.qkv(prefix_input).chunk(3,-1)
        return self._split(k),self._split(v)

    def suffix(self, suffix: torch.Tensor, prefix_kv: tuple[torch.Tensor,torch.Tensor], allowed_rows: torch.Tensor) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor]]:
        q,k_suffix,v_suffix=self.qkv(suffix).chunk(3,-1); q=self._split(q)
        suffix_kv=(self._split(k_suffix),self._split(v_suffix)); prefix_k,prefix_v=prefix_kv
        k=torch.cat([prefix_k,suffix_kv[0]],dim=2); v=torch.cat([prefix_v,suffix_kv[1]],dim=2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed_rows[None,None], dropout_p=0.0)
        return self.out(out.transpose(1,2).contiguous().flatten(2)),suffix_kv


class JointBlock(nn.Module):
    def __init__(self, dim, heads, ratio, dropout):
        super().__init__(); self.norm1=nn.LayerNorm(dim); self.attn=JointAttention(dim,heads,dropout); self.norm2=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,int(dim*ratio)),nn.GELU(),nn.Dropout(dropout),nn.Linear(int(dim*ratio),dim))
        # A compact, actively-used residual mixer closes the parameter-budget gap
        # introduced by the layerwise policies' time/AdaLN modules without
        # changing the joint policy's cache-safe attention topology.
        self.norm3=nn.LayerNorm(dim)
        self.mixer=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim,dim))

    def forward(self, x, allowed):
        x=x+self.attn(self.norm1(x),allowed); x=x+self.ff(self.norm2(x)); return x+self.mixer(self.norm3(x))

    def suffix(self, suffix, prefix_kv, allowed_rows):
        update,suffix_kv=self.attn.suffix(self.norm1(suffix),prefix_kv,allowed_rows); suffix=suffix+update
        suffix=suffix+self.ff(self.norm2(suffix)); suffix=suffix+self.mixer(self.norm3(suffix)); return suffix,suffix_kv


@dataclass
class JointKVCache:
    prefix_length: int
    layer_kv: list[tuple[torch.Tensor,torch.Tensor]]
    observation_signature: tuple[int, int]


class JointDiscretePolicy(PolicyBase):
    def __init__(self, cfg: Any):
        super().__init__(cfg); d=cfg.policy.hidden_dim; self.mask_id=256
        self.input_embedding=nn.Embedding(257,d); self.action_position=nn.Parameter(torch.randn(self.action_positions,d)*.02)
        self.blocks=nn.ModuleList([JointBlock(d,cfg.policy.heads,cfg.policy.mlp_ratio,cfg.policy.dropout) for _ in range(cfg.policy.depth)])
        self.norm=nn.LayerNorm(d); self.output_head=nn.Linear(d,256)

    def attention_mask(self, obs_len: int, action_len: int, device) -> torch.Tensor:
        length=obs_len+action_len; allowed=torch.zeros(length,length,dtype=torch.bool,device=device)
        allowed[:obs_len,:obs_len]=True
        bs=self.cfg.policy.block_size
        for q in range(action_len):
            block=q//bs; end=min((block+1)*bs,action_len)
            allowed[obs_len+q,:obs_len+end]=True
        return allowed

    def _sequence(self, action_tokens, obs):
        action=self.input_embedding(action_tokens)+self.action_position[None,:action_tokens.shape[1]]
        return torch.cat([obs.tokens,action],1)

    def logits(self, action_tokens, obs, capture_cache_prefix: int | None=None):
        x=self._sequence(action_tokens,obs); allowed=self.attention_mask(obs.tokens.shape[1],action_tokens.shape[1],x.device); captures=[]
        for block in self.blocks:
            if capture_cache_prefix is not None:
                k,v=block.attn.project_kv(block.norm1(x[:,:capture_cache_prefix])); captures.append((k.detach(),v.detach()))
            x=block(x,allowed)
        logits=self.output_head(self.norm(x[:,obs.tokens.shape[1]:]))
        if capture_cache_prefix is None: return logits
        sig=(obs.tokens.data_ptr(),obs.tokens.numel())
        return logits,JointKVCache(capture_cache_prefix,captures,sig)

    def observation_cache(self, obs) -> JointKVCache:
        """Build the action-independent observation K/V once per plan."""
        x=obs.tokens; allowed=torch.ones(x.shape[1],x.shape[1],dtype=torch.bool,device=x.device); captures=[]
        for block in self.blocks:
            k,v=block.attn.project_kv(block.norm1(x)); captures.append((k.detach(),v.detach())); x=block(x,allowed)
        sig=(obs.tokens.data_ptr(),obs.tokens.numel())
        return JointKVCache(obs.tokens.shape[1],captures,sig)

    def cached_suffix_logits(self, suffix_tokens, obs, cache: JointKVCache, return_suffix_kv: bool=False):
        sig=(obs.tokens.data_ptr(),obs.tokens.numel())
        if sig != cache.observation_signature: raise ValueError("observation/state changed; invalidate the joint KV cache")
        action_offset=cache.prefix_length-obs.tokens.shape[1]
        positions=self.action_position[action_offset:action_offset+suffix_tokens.shape[1]]
        x=self.input_embedding(suffix_tokens)+positions[None]
        total_actions=action_offset+suffix_tokens.shape[1]
        full=self.attention_mask(obs.tokens.shape[1],total_actions,x.device)
        rows=full[cache.prefix_length:,:cache.prefix_length+suffix_tokens.shape[1]]
        suffix_layers=[]
        for block,prefix_kv in zip(self.blocks,cache.layer_kv):
            x,suffix_kv=block.suffix(x,prefix_kv,rows); suffix_layers.append(suffix_kv)
        logits=self.output_head(self.norm(x))
        return (logits,suffix_layers) if return_suffix_kv else logits

    def extend_cache(self, cache: JointKVCache, suffix_kv: list[tuple[torch.Tensor,torch.Tensor]], length: int) -> JointKVCache:
        layers=[(torch.cat([old[0],new[0]],dim=2),torch.cat([old[1],new[1]],dim=2)) for old,new in zip(cache.layer_kv,suffix_kv)]
        return JointKVCache(cache.prefix_length+length,layers,cache.observation_signature)

    def loss(self,batch,rtc=None):
        target=batch["discrete_target"].long().flatten(1); valid=batch["control_valid_mask"].bool().flatten(1)
        corrupted,supervised=monotonic_block_corruption(target,valid,self.cfg.policy.block_size)
        if rtc is not None:
            fixed=rtc["fixed_mask"].bool().flatten(1); fixed_tokens=rtc["prefix_values"].long().flatten(1)
            corrupted=torch.where(fixed,fixed_tokens,corrupted); supervised &= ~fixed
        obs=self.observations(batch); logits=self.logits(corrupted,obs)
        ce,l1=expected_token_distance(logits,target,supervised); total=ce+self.cfg.train.lambda_l1*l1
        result={"loss":total,"loss_ce":ce,"loss_l1":l1,"gradient_scale_ce":ce.detach(),"gradient_scale_l1":l1.detach()}
        result["action_mse"]=self.logits_action_mse(logits,batch,supervised).detach()
        return result

    @torch.no_grad()
    def sample(self,batch,rounds=None,prefix_values=None,fixed_mask=None,use_cache=True,
               fuse_cache_transition=True,return_trace=False,**kwargs):
        obs=self.observations(batch); b=len(obs.tokens); rounds=int(rounds or self.cfg.policy.discrete_rounds); bs=self.cfg.policy.block_size
        tokens=torch.full((b,self.action_positions),self.mask_id,device=obs.tokens.device,dtype=torch.long)
        immutable=torch.zeros_like(tokens,dtype=torch.bool)
        if fixed_mask is not None:
            immutable=fixed_mask.flatten(1).bool(); prefix_values=prefix_values.flatten(1).long(); tokens=torch.where(immutable,prefix_values,tokens)
        trace=[]
        cache=self.observation_cache(obs) if use_cache else None; prefetched_logits=None
        for start in range(0,self.action_positions,bs):
            end=min(start+bs,self.action_positions); mutable=torch.zeros_like(tokens,dtype=torch.bool); mutable[:,start:end]=~immutable[:,start:end]
            for step in range(rounds):
                if cache is None: block_logits=self.logits(tokens,obs)[:,start:end]
                elif step==0 and prefetched_logits is not None:
                    block_logits=prefetched_logits; prefetched_logits=None
                else: block_logits=self.cached_suffix_logits(tokens[:,start:end],obs,cache)
                current=tokens[:,start:end]; local_mutable=mutable[:,start:end]
                tokens[:,start:end]=maskgit_update(block_logits,current,local_mutable,step,rounds)
                if fixed_mask is not None: tokens=torch.where(immutable,prefix_values,tokens)
                if return_trace: trace.append({"block_start":start,"step":step,"tokens":tokens.clone(),"logits":block_logits.clone()})
            if cache is not None and end < self.action_positions:
                if fuse_cache_transition:
                    # D2F-style inter-block transition: recompute the completed
                    # block for its stable K/V and process the next masked block
                    # in the same forward.  The prefetched next-block logits are
                    # exactly the first denoising round that would otherwise
                    # require a separate call.
                    next_end=min(end+bs,self.action_positions)
                    transition_logits,suffix_kv=self.cached_suffix_logits(
                        tokens[:,start:next_end],obs,cache,return_suffix_kv=True
                    )
                    committed=end-start
                    committed_kv=[(k[:,:,:committed],v[:,:,:committed]) for k,v in suffix_kv]
                    cache=self.extend_cache(cache,committed_kv,committed)
                    prefetched_logits=transition_logits[:,committed:]
                else:
                    _,suffix_kv=self.cached_suffix_logits(tokens[:,start:end],obs,cache,return_suffix_kv=True)
                    cache=self.extend_cache(cache,suffix_kv,end-start)
        result=tokens.reshape(b,self.num_basis,self.action_dim)
        return (result,trace) if return_trace else result
