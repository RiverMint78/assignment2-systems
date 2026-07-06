from __future__ import annotations

import math

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torch.autograd.function import FunctionCtx


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Float[Tensor, "batch n_queries d_model"],
        k: Float[Tensor, "batch n_keys d_model"],
        v: Float[Tensor, "batch n_keys d_model"],
        is_causal: bool = False,
    ) -> Float[Tensor, "batch n_queries d_model"]:
        d_model = q.shape[-1]
        scale = 1.0 / math.sqrt(d_model)
        scores: Float[Tensor, "batch n_queries n_keys"] = torch.einsum("bqd,bkd->bqk", q, k) * scale

        if is_causal:
            n_queries = q.shape[-2]
            n_keys = k.shape[-2]
            query_positions = torch.arange(n_queries, device=q.device)
            key_positions = torch.arange(n_keys, device=q.device)
            causal_mask: Bool[Tensor, "n_queries n_keys"] = key_positions[None, :] <= query_positions[:, None]
            scores.masked_fill_(~causal_mask[None, :, :], float("-inf"))

        lse: Float[Tensor, "batch n_queries"] = torch.logsumexp(scores, dim=-1) # implicit safe
        probs: Float[Tensor, "batch n_queries n_keys"] = torch.exp(scores - lse[..., None])
        o: Float[Tensor, "batch n_queries d_model"] = torch.einsum("bqk,bkd->bqd", probs, v)

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.is_causal = is_causal

        return o

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_o: Float[Tensor, "batch n_queries d_model"],
    ) -> tuple[
        Float[Tensor, "batch n_queries d_model"],
        Float[Tensor, "batch n_keys d_model"],
        Float[Tensor, "batch n_keys d_model"],
        None,
    ]:
        q, k, v, o, lse = ctx.saved_tensors
        is_causal: bool = ctx.is_causal

        d_model = q.shape[-1]
        scale = 1.0 / math.sqrt(d_model)
        scores: Float[Tensor, "batch n_queries n_keys"] = torch.einsum("bqd,bkd->bqk", q, k) * scale

        if is_causal:
            n_queries = q.shape[-2]
            n_keys = k.shape[-2]
            query_positions = torch.arange(n_queries, device=q.device)
            key_positions = torch.arange(n_keys, device=q.device)
            causal_mask: Bool[Tensor, "n_queries n_keys"] = key_positions[None, :] <= query_positions[:, None]
            scores.masked_fill_(~causal_mask[None, :, :], float("-inf"))

        probs: Float[Tensor, "batch n_queries n_keys"] = torch.exp(scores - lse[..., None])

        grad_p: Float[Tensor, "batch n_queries n_keys"] = torch.einsum("bqd,bkd->bqk", grad_o, v)  # dP = dO V^T
        grad_s: Float[Tensor, "batch n_queries n_keys"] = probs * (grad_p - torch.sum(grad_o * o, dim=-1)[..., None])  # dS = P·(dP - rowsum(dO·O))

        grad_q: Float[Tensor, "batch n_queries d_model"] = torch.einsum("bqk,bkd->bqd", grad_s, k) * scale
        grad_k: Float[Tensor, "batch n_keys d_model"] = torch.einsum("bqk,bqd->bkd", grad_s, q) * scale
        grad_v: Float[Tensor, "batch n_keys d_model"] = torch.einsum("bqk,bqd->bkd", probs, grad_o)  # dV = P^T dO

        return grad_q, grad_k, grad_v, None
