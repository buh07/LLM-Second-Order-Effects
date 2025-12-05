"""Numerical helpers for second-order neuron calculations."""

from __future__ import annotations

import torch

from phase2.gpt_second_order_hook import LayerNormStats


def apply_mlp_post(post_activation: torch.Tensor, proj_weight: torch.Tensor) -> torch.Tensor:
    """Project post-activation neuron outputs through c_proj.

    Args:
        post_activation: Tensor with shape [B, T, neurons].
        proj_weight: Selected rows from c_proj.weight [neurons, hidden_dim].

    Returns:
        Tensor with shape [B, T, neurons, hidden_dim].
    """

    projected = torch.einsum("btn,nm->btnm", post_activation, proj_weight)
    return projected


def apply_layer_norm_linear(
    tensor: torch.Tensor,
    stats: LayerNormStats,
    layer_norm: torch.nn.LayerNorm,
) -> torch.Tensor:
    """Apply the linearized effect of layer norm to neuron contributions."""

    std = stats.std.unsqueeze(2).clamp_min(1e-6)  # [B, T, 1, 1]
    centered = tensor - tensor.mean(dim=-1, keepdim=True)
    normalized = centered / std
    weight = layer_norm.weight.view(1, 1, 1, -1)
    return normalized * weight


def attention_to_final_token(
    tensor: torch.Tensor,
    attn_probs: torch.Tensor,
    attn_module,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    """Propagate neuron contributions through a single attention layer.

    Args:
        tensor: [B, T, neurons, hidden_dim], normalized residual stream.
        attn_probs: [B, heads, seq, seq] attention maps.
        attn_module: GPT2Attention module for the layer.
        target_positions: [B] indices of the final token per example.

    Returns:
        Contributions at the final token after the attention output,
        shaped [B, neurons, hidden_dim].
    """

    hidden_size = tensor.shape[-1]
    num_heads = attn_module.num_heads
    head_dim = hidden_size // num_heads

    v_weight = attn_module.c_attn.weight[:, 2 * hidden_size :]
    value = torch.einsum("btnh,hk->btnk", tensor, v_weight)
    value = value.view(tensor.shape[0], tensor.shape[1], tensor.shape[2], num_heads, head_dim)
    value = value.permute(0, 3, 1, 2, 4)  # [B, heads, T, neurons, head_dim]

    batch_size = tensor.shape[0]
    device = attn_probs.device
    target_positions = target_positions.to(device)
    batch_idx = torch.arange(batch_size, device=device)
    weights = attn_probs[batch_idx, :, target_positions, :]  # [B, heads, T]

    context = torch.einsum("bht,bhtnd->bhnd", weights, value)
    context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, tensor.shape[2], hidden_size)

    projected = torch.einsum("bnh,hk->bnk", context, attn_module.c_proj.weight)
    return projected


def apply_final_layer_norm(
    tensor: torch.Tensor,
    stats: LayerNormStats,
    layer_norm: torch.nn.LayerNorm,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    """Map attention contributions into the final residual stream."""

    batch_size = tensor.shape[0]
    device = tensor.device
    batch_idx = torch.arange(batch_size, device=device)
    std = stats.std.to(device)[batch_idx, target_positions].unsqueeze(1).clamp_min(1e-6)
    centered = tensor - tensor.mean(dim=-1, keepdim=True)
    normalized = centered / std.unsqueeze(-1)
    weight = layer_norm.weight.view(1, 1, -1)
    return normalized * weight


def project_to_direction(
    tensor: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Project per-neuron contributions onto a direction vector."""

    direction = direction / direction.norm().clamp_min(1e-6)
    return torch.einsum("bnh,h->bn", tensor, direction)
