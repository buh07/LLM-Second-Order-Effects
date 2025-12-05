"""Ablation experiments on LLM second-order tensors and correctness direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_datasets.math_dataset_loader import MathDatasetLoader, MathExample
from phase2.compute_second_order_neuron_prs import encode_batch, load_examples, select_neuron_indices
from phase2.gpt_second_order_hook import GptSecondOrderHook
from phase2.second_order_math import (
    apply_final_layer_norm,
    apply_layer_norm_linear,
    apply_mlp_post,
    attention_to_final_token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM analogue of compute_ablations.py")
    parser.add_argument("--dataset", type=str, default="mawps")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--datasets_root", type=str, default="datasets")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--sample_size", type=int, default=256)
    parser.add_argument("--prompt_template", type=str, default="Solve the problem:\n{problem}\nAnswer:")
    parser.add_argument("--mlp_layer", type=int, default=9)
    parser.add_argument("--neuron_indices_path", type=str, default="")
    parser.add_argument("--max_neurons", type=int, default=100)
    parser.add_argument("--train_second_order_path", type=str, required=True, help="Training tensor for computing means")
    parser.add_argument("--pca_path", type=str, required=True)
    parser.add_argument("--norm_path", type=str, required=True)
    parser.add_argument("--correctness_direction", type=str, required=True)
    parser.add_argument("--correctness_labels", type=str, required=True)
    parser.add_argument("--norm_percentile", type=float, default=75.0)
    parser.add_argument("--output_json", type=str, default="phase2_outputs/ablations.json")
    parser.add_argument("--coefficient", type=float, default=100.0)
    return parser.parse_args()


def load_mean_vectors(train_path: Path, neuron_indices: np.ndarray | None = None) -> np.ndarray:
    tensors = np.load(train_path, mmap_mode="r")
    mean_vecs = tensors.mean(axis=0)
    if neuron_indices is not None:
        mean_vecs = mean_vecs[neuron_indices]
    return mean_vecs


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
    return summed * (hook.coefficient / norms.unsqueeze(-1))


def select_masks(norm_path: Path, percentile: float, neuron_indices: np.ndarray | None = None) -> np.ndarray:
    norms = np.load(norm_path)
    aggregated = norms.mean(axis=1)
    if neuron_indices is not None:
        aggregated = aggregated[neuron_indices]
    threshold = np.percentile(aggregated, percentile)
    return aggregated >= threshold


def project_logits(representations: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return representations @ direction


def main() -> None:
    args = parse_args()
    examples = load_examples(args)
    if args.sample_size and args.sample_size > 0:
        examples = examples[: min(args.sample_size, len(examples))]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    direction = np.load(args.correctness_direction)
    labels = np.load(args.correctness_labels)[: len(examples)]
    neuron_list = select_neuron_indices(args, model)
    normalized_indices = np.array(neuron_list)
    neuron_indices = torch.tensor(neuron_list, dtype=torch.long)
    mean_vectors = load_mean_vectors(Path(args.train_second_order_path), normalized_indices)
    pca_vectors = np.load(args.pca_path)[normalized_indices]
    significant_mask = select_masks(Path(args.norm_path), args.norm_percentile, normalized_indices)

    hook = GptSecondOrderHook(model, args.mlp_layer, torch.device(args.device), coefficient=args.coefficient)
    baseline_logits = []
    ablated_logits = {"mean": [], "remove_significant": [], "remove_insignificant": [], "rank1": []}

    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        prompts = [args.prompt_template.format(problem=ex.prompt) for ex in batch]
        encoded = encode_batch(tokenizer, prompts, args.max_length, torch.device(args.device))
        hook.reinit(encoded["input_ids"].shape[0], encoded["input_ids"].shape[-1])
        with torch.no_grad():
            outputs = model(
                **encoded,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
            )
        hook.set_attention_maps(list(outputs.attentions))
        contributions = compute_second_order_vectors(
            model,
            hook,
            encoded["attention_mask"],
            neuron_indices,
            outputs.hidden_states[-1][:, -1, :],
        )
        final_hidden = outputs.hidden_states[-1][:, -1, :].detach().cpu().numpy()
        coeffs = np.einsum("bnh,nh->bn", contributions.cpu().numpy(), pca_vectors)
        rank1_recon = coeffs[:, :, None] * pca_vectors[np.newaxis, :, :]
        mean_sum = mean_vectors.sum(axis=0)
        contrib_np = contributions.cpu().numpy()
        contrib_sum = contrib_np.sum(axis=1)

        baseline_repr = args.coefficient * final_hidden
        baseline_logits.extend(project_logits(baseline_repr, direction))

        # Mean ablation
        mean_repr = baseline_repr - contrib_sum + mean_sum
        ablated_logits["mean"].extend(project_logits(mean_repr, direction))

        # Remove significant
        mask = significant_mask[: contrib_np.shape[1]]
        sig_contrib = contrib_np[:, mask, :].sum(axis=1)
        nonsig_contrib = contrib_np[:, ~mask, :].sum(axis=1)
        edit_sig = baseline_repr - sig_contrib
        ablated_logits["remove_significant"].extend(project_logits(edit_sig, direction))
        edit_nonsig = baseline_repr - nonsig_contrib
        ablated_logits["remove_insignificant"].extend(project_logits(edit_nonsig, direction))

        # Rank-1 substitution
        rank_sum = rank1_recon.sum(axis=1)
        rank_repr = baseline_repr - contrib_sum + rank_sum
        ablated_logits["rank1"].extend(project_logits(rank_repr, direction))

    label_slice = labels[: len(baseline_logits)].astype(bool)
    baseline_pred = np.array(baseline_logits) > 0
    baseline_acc = float((baseline_pred == label_slice).mean())
    results = {"baseline_accuracy": baseline_acc}
    for key, logits in ablated_logits.items():
        preds = np.array(logits) > 0
        acc = float((preds == label_slice).mean())
        results[key] = {
            "accuracy": acc,
            "delta": acc - baseline_acc,
        }
    Path(args.output_json).write_text(json.dumps(results, indent=2))
    print("Ablation results:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
