"""Numerical sanity checks for the GPT second-order hook."""

from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from phase2.compute_second_order_neuron_prs import (
    encode_batch,
    load_examples,
    select_neuron_indices,
)
from phase2.gpt_second_order_hook import GptSecondOrderHook
from phase2.second_order_math import (
    apply_final_layer_norm,
    apply_layer_norm_linear,
    apply_mlp_post,
    attention_to_final_token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate second-order reconstructions via ablation")
    parser.add_argument("--dataset", type=str, default="mawps")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--datasets_root", type=str, default="datasets")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--sample_size", type=int, default=16)
    parser.add_argument("--prompt_template", type=str, default="Solve the problem:\n{problem}\nAnswer:")
    parser.add_argument("--mlp_layer", type=int, default=9)
    parser.add_argument("--neuron_indices_path", type=str, default="")
    parser.add_argument("--max_neurons", type=int, default=64)
    return parser.parse_args()


def zero_mlp_context(model, layer_idx: int):
    block = model.transformer.h[layer_idx]

    def hook(_module, _inputs, output):
        return torch.zeros_like(output)

    handle = block.mlp.register_forward_hook(hook)
    return handle


def compute_second_order_vectors(
    model,
    hook: GptSecondOrderHook,
    attention_mask: torch.Tensor,
    neuron_indices: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> torch.Tensor:
    hook.to_device()
    neuron_indices = neuron_indices.to(hook.device)
    post_activation = torch.index_select(hook.post_activation, dim=-1, index=neuron_indices)
    proj_weight = model.transformer.h[hook.mlp_layer].mlp.c_proj.weight[neuron_indices].to(hook.device)
    mlp_outputs = apply_mlp_post(post_activation, proj_weight)
    lengths = attention_mask.sum(dim=-1) - 1
    lengths = lengths.clamp(min=0).to(torch.long)

    contributions = []
    for layer_idx in range(hook.mlp_layer + 1, hook.num_layers):
        stats = hook.ln1_stats[layer_idx]
        if stats is None:
            continue
        ln_module = model.transformer.h[layer_idx].ln_1
        normalized = apply_layer_norm_linear(mlp_outputs, stats, ln_module)
        attn_probs = hook.attn_maps[:, layer_idx]
        attn_module = model.transformer.h[layer_idx].attn
        attention_output = attention_to_final_token(normalized, attn_probs, attn_module, lengths)
        final_norm = apply_final_layer_norm(attention_output, hook.final_ln_stats, model.transformer.ln_f, lengths)
        contributions.append(final_norm)
    if not contributions:
        summed = torch.zeros(
            mlp_outputs.shape[0],
            len(neuron_indices),
            model.config.hidden_size,
            device=hook.device,
        )
    else:
        summed = torch.stack(contributions, dim=0).sum(dim=0)
    norms = final_hidden_states.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return summed / norms.unsqueeze(-1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    examples = load_examples(args)
    if args.sample_size and args.sample_size > 0:
        examples = examples[: min(args.sample_size, len(examples))]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    else:
        model.config._attn_implementation = "eager"
    model.config.output_attentions = True
    model.config.use_cache = False

    neuron_indices = torch.tensor(select_neuron_indices(args, model), dtype=torch.long)
    hook = GptSecondOrderHook(model, args.mlp_layer, device)

    total_examples = 0
    rel_errors: List[float] = []
    per_example_logs = []

    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        prompts = [args.prompt_template.format(problem=ex.prompt) for ex in batch]
        encoded = encode_batch(tokenizer, prompts, args.max_length, device)
        hook.reinit(encoded["input_ids"].shape[0], encoded["input_ids"].shape[-1])

        with torch.no_grad():
            outputs = model(
                **encoded,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
            )
        hook.set_attention_maps(list(outputs.attentions))
        vectors = compute_second_order_vectors(
            model,
            hook,
            encoded["attention_mask"],
            neuron_indices,
            outputs.hidden_states[-1][:, -1, :],
        )
        summed = vectors.sum(dim=1)

        handle = zero_mlp_context(model, args.mlp_layer)
        with torch.no_grad():
            ablated = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
            )
        handle.remove()

        baseline = outputs.hidden_states[-1][:, -1, :]
        ablated_hidden = ablated.hidden_states[-1][:, -1, :]
        delta = baseline - ablated_hidden

        error = torch.linalg.norm(summed - delta, dim=-1) / torch.linalg.norm(delta, dim=-1).clamp(min=1e-6)
        rel_errors.extend(error.detach().cpu().tolist())
        total_examples += len(batch)
        per_example_logs.append(
            {
                "start_index": batch_start,
                "batch_size": len(batch),
                "mean_relative_error": float(error.mean().item()),
                "max_relative_error": float(error.max().item()),
            }
        )

    overall = {
        "num_examples": total_examples,
        "mean_relative_error": float(np.mean(rel_errors)),
        "median_relative_error": float(np.median(rel_errors)),
        "max_relative_error": float(np.max(rel_errors)),
        "neuron_count": len(neuron_indices),
        "mlp_layer": args.mlp_layer,
    }
    print("Second-order validation summary:")
    print(json.dumps(overall, indent=2))
    print("Per-batch logs:")
    print(json.dumps(per_example_logs, indent=2))


if __name__ == "__main__":
    main()
