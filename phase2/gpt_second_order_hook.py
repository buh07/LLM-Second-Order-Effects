"""Hook utilities for capturing GPT-style second-order statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class LayerNormStats:
    mean: torch.Tensor  # (batch, seq, 1)
    std: torch.Tensor   # (batch, seq, 1)


class GptSecondOrderHook:
    """Capture MLP activations + LayerNorm statistics for GPT-style blocks."""

    def __init__(
        self,
        model,
        mlp_layer: int,
        device: torch.device,
        coefficient: float = 100.0,
    ) -> None:
        self.model = model
        self.mlp_layer = mlp_layer
        self.device = device
        self.coefficient = coefficient
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.num_layers = len(model.transformer.h)
        self.post_activation: Optional[torch.Tensor] = None
        self.ln1_stats: List[Optional[LayerNormStats]] = [None] * self.num_layers
        self.final_ln_stats: Optional[LayerNormStats] = None
        self.attn_maps: Optional[torch.Tensor] = None
        self.seq_len: Optional[int] = None
        self.batch_size: Optional[int] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        target_block = self.model.transformer.h[self.mlp_layer]
        self.handles.append(
            target_block.mlp.act.register_forward_hook(self._store_post_activation)
        )
        for idx, block in enumerate(self.model.transformer.h):
            self.handles.append(
                block.ln_1.register_forward_pre_hook(
                    self._make_layer_norm_logger(idx)
                )
            )
        self.handles.append(
            self.model.transformer.ln_f.register_forward_pre_hook(
                self._store_final_ln_stats
            )
        )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reinit(self, batch_size: int, seq_len: int) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.post_activation = None
        self.ln1_stats = [None] * self.num_layers
        self.final_ln_stats = None
        self.attn_maps = None

    def _store_post_activation(self, _module, _inputs, output) -> None:
        self.post_activation = output.detach().to("cpu")

    def _make_layer_norm_logger(self, layer_idx: int):
        def hook(_module, inputs):
            hidden = inputs[0]
            mean = hidden.mean(dim=-1, keepdim=True)
            var = hidden.var(dim=-1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + _module.eps)
            self.ln1_stats[layer_idx] = LayerNormStats(
                mean=mean.detach().to("cpu"),
                std=std.detach().to("cpu"),
            )

        return hook

    def _store_final_ln_stats(self, module, inputs):
        hidden = inputs[0]
        mean = hidden.mean(dim=-1, keepdim=True)
        var = hidden.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + module.eps)
        self.final_ln_stats = LayerNormStats(
            mean=mean.detach().to("cpu"),
            std=std.detach().to("cpu"),
        )

    def finalized(self) -> bool:
        return (
            self.post_activation is not None
            and all(stat is not None for stat in self.ln1_stats)
            and self.final_ln_stats is not None
            and self.attn_maps is not None
        )

    def set_attention_maps(self, attentions: List[torch.Tensor]) -> None:
        """Cache attention probabilities from model outputs."""

        stacked = torch.stack(attentions, dim=1)  # [batch, layers, heads, seq, seq]
        self.attn_maps = stacked.detach().to("cpu")

    def to_device(self) -> None:
        """Move cached tensors to the hook device for downstream math."""

        if self.post_activation is not None:
            self.post_activation = self.post_activation.to(self.device)
        self.ln1_stats = [
            LayerNormStats(
                mean=stat.mean.to(self.device),
                std=stat.std.to(self.device),
            )
            if stat is not None
            else None
            for stat in self.ln1_stats
        ]
        if self.final_ln_stats is not None:
            self.final_ln_stats = LayerNormStats(
                mean=self.final_ln_stats.mean.to(self.device),
                std=self.final_ln_stats.std.to(self.device),
            )
        if self.attn_maps is not None:
            self.attn_maps = self.attn_maps.to(self.device)
