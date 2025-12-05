"""Second-order neuron projection for GPT-style language models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_datasets.math_dataset_loader import MathDatasetLoader, MathExample
from phase2.gpt_second_order_hook import GptSecondOrderHook
from phase2.prompts import DEFAULT_PROMPT_TEMPLATE, build_inference_prompt
from phase2.second_order_math import (
    apply_final_layer_norm,
    apply_layer_norm_linear,
    apply_mlp_post,
    attention_to_final_token,
    project_to_direction,
)
from phase2.utils import clean_model_name, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM analogue of compute_second_order_neuron_prs.py")
    parser.add_argument("--dataset", type=str, default="mawps")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--datasets_root", type=str, default="datasets")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1, help="Large tensors require small batches.")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--mlp_layer", type=int, default=9)
    parser.add_argument("--neuron_indices_path", type=str, default="", help="Optional JSON from Phase 1 ranking.")
    parser.add_argument("--max_neurons", type=int, default=100, help="Limit number of neurons (after loading indices).")
    parser.add_argument("--correctness_direction", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="phase2_outputs")
    parser.add_argument("--coefficient", type=float, default=100.0)
    return parser.parse_args()


def load_examples(args: argparse.Namespace) -> List[MathExample]:
    loader = MathDatasetLoader(args.datasets_root)
    examples = loader.load(args.dataset, split=args.split)
    if args.sample_size and args.sample_size > 0:
        examples = examples[: min(args.sample_size, len(examples))]
    return examples


def select_neuron_indices(args: argparse.Namespace, model) -> List[int]:
    width = model.transformer.h[args.mlp_layer].mlp.c_proj.weight.shape[0]
    if args.neuron_indices_path:
        data = json.loads(Path(args.neuron_indices_path).read_text())
        layer_key = str(args.mlp_layer)
        entries = data.get(layer_key) or []
        indices = [entry["neuron_index"] for entry in entries]
    else:
        indices = list(range(width))
    if args.max_neurons and args.max_neurons > 0:
        indices = indices[: min(args.max_neurons, len(indices))]
    if not indices:
        raise ValueError("No neuron indices selected. Provide ranking JSON or adjust --max_neurons.")
    return indices


def encode_batch(tokenizer, prompts: List[str], max_length: int, device: torch.device):
    encodings = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encodings.items()}


def compute_batch_scores(
    model,
    hook: GptSecondOrderHook,
    attention_mask: torch.Tensor,
    neuron_indices: torch.Tensor,
    correctness_direction: torch.Tensor,
    coefficient: float,
    final_hidden_states: torch.Tensor,
):
    if not hook.finalized():
        raise RuntimeError("Hook buffers not finalized before score computation.")

    hook.to_device()
    neuron_indices = neuron_indices.to(hook.device)
    post_activation = torch.index_select(hook.post_activation, dim=-1, index=neuron_indices)

    proj_weight = model.transformer.h[hook.mlp_layer].mlp.c_proj.weight[neuron_indices].to(hook.device)
    mlp_outputs = apply_mlp_post(post_activation, proj_weight)

    lengths = attention_mask.sum(dim=-1) - 1
    lengths = lengths.clamp(min=0).to(torch.long)
    contributions = []
    batches = mlp_outputs.shape[0]
    for layer_idx in range(hook.mlp_layer + 1, hook.num_layers):
        stats = hook.ln1_stats[layer_idx]
        if stats is None:
            continue
        ln_module = model.transformer.h[layer_idx].ln_1
        normalized = apply_layer_norm_linear(mlp_outputs, stats, ln_module)
        attn_probs = hook.attn_maps[layer_idx]
        attn_module = model.transformer.h[layer_idx].attn
        attention_output = attention_to_final_token(normalized, attn_probs, attn_module, lengths)
        final_norm = apply_final_layer_norm(attention_output, hook.final_ln_stats, model.transformer.ln_f, lengths)
        contributions.append(final_norm)
    if not contributions:
        summed = torch.zeros(batches, len(neuron_indices), model.config.hidden_size, device=hook.device)
    else:
        summed = torch.stack(contributions, dim=0).sum(dim=0)
    final_norms = final_hidden_states.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    scaled = summed * coefficient / final_norms.unsqueeze(-1)
    scores = project_to_direction(scaled, correctness_direction)
    return scaled, scores


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    examples = load_examples(args)
    if not examples:
        raise RuntimeError("Dataset selection yielded zero examples.")

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
    direction = torch.from_numpy(np.load(args.correctness_direction)).to(device)

    hook = GptSecondOrderHook(model, args.mlp_layer, device, coefficient=args.coefficient)

    outputs = []
    projected_scores = []
    output_dir = ensure_dir(args.output_dir)
    prefix = f"{args.dataset}_{args.split}_{clean_model_name(args.model_name)}_layer{args.mlp_layer}"

    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        prompts = [build_inference_prompt(ex, args.prompt_template) for ex in batch]
        encoded = encode_batch(tokenizer, prompts, args.max_length, device)
        hook.reinit(batch_size=encoded["input_ids"].shape[0], seq_len=encoded["input_ids"].shape[-1])
        with torch.no_grad():
            model_outputs = model(
                **encoded,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
            )
        batch_vectors, batch_scores = compute_batch_scores(
            model,
            hook,
            encoded["attention_mask"],
            neuron_indices,
            direction,
            args.coefficient,
            model_outputs.hidden_states[-1][:, -1, :],
        )
        outputs.append(batch_vectors.detach().cpu().numpy())
        projected_scores.append(batch_scores.detach().cpu().numpy())

    merged = np.concatenate(outputs, axis=0)
    result_path = output_dir / f"{prefix}_second_order_layer{args.mlp_layer}.npy"
    np.save(result_path, merged)
    score_path = output_dir / f"{prefix}_correctness_scores_layer{args.mlp_layer}.npy"
    np.save(score_path, np.concatenate(projected_scores, axis=0))

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "model_name": args.model_name,
        "mlp_layer": args.mlp_layer,
        "num_examples": len(examples),
        "num_neurons": len(neuron_indices),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "correctness_direction": args.correctness_direction,
        "neuron_indices_path": args.neuron_indices_path,
        "output": str(result_path),
        "correctness_scores": str(score_path),
    }
    (output_dir / f"{prefix}_second_order_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
