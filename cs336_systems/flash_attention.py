from __future__ import annotations

import torch
from jaxtyping import Float
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
        raise NotImplementedError

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
        raise NotImplementedError
