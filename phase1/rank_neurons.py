import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_datasets.math_dataset_loader import MathDatasetLoader, MathExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank neurons using activation heuristics.")
    parser.add_argument("--dataset", type=str, default="mawps")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--datasets_root", type=str, default="datasets")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--layers", type=str, default="all", help="Comma-separated layer indices or 'all'")
    parser.add_argument("--sample_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--output_json", type=str, default="phase1_outputs/neuron_ranking.json")
    return parser.parse_args()


def get_target_layers(model, spec: str) -> List[int]:
    total_layers = len(model.transformer.h)
    if spec.lower() == "all":
        return list(range(total_layers))
    indices = [int(idx.strip()) for idx in spec.split(",") if idx.strip()]
    return [i for i in indices if 0 <= i < total_layers]


def build_input_text(example: MathExample) -> str:
    answer_text = example.answer if example.answer else example.reasoning
    return f"{example.prompt}\nAnswer: {answer_text}"


def record_stats(model, tokenizer, examples: List[MathExample], layers: List[int], device: torch.device):
    stats = {}
    handles = []

    for layer in layers:
        block = model.transformer.h[layer]
        proj_norm = block.mlp.c_proj.weight.detach().norm(dim=0).cpu()
        intermediate_size = proj_norm.shape[0]
        stats[layer] = {
            "sum_abs": torch.zeros(intermediate_size),
            "sum_sq": torch.zeros(intermediate_size),
            "sum_effect": torch.zeros(intermediate_size),
            "count": 0,
        }

        def make_hook(layer_idx: int, proj_norm_vec: torch.Tensor):
            def hook(_module, _inputs, output):
                activations = output[:, -1, :].detach().cpu()
                stats[layer_idx]["sum_abs"] += activations.abs().sum(dim=0)
                stats[layer_idx]["sum_sq"] += (activations ** 2).sum(dim=0)
                stats[layer_idx]["sum_effect"] += (activations.abs() * proj_norm_vec).sum(dim=0)
                stats[layer_idx]["count"] += activations.shape[0]
                return output

            return hook

        handle = block.mlp.c_fc.register_forward_hook(make_hook(layer, proj_norm))
        handles.append(handle)

    for example in examples:
        text = build_input_text(example)
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            model(**inputs)

    for handle in handles:
        handle.remove()

    return stats


def rank_neurons(statistics: Dict[int, Dict[str, torch.Tensor]], top_k: int) -> Dict[int, List[Dict]]:
    rankings = {}
    for layer, stat in statistics.items():
        count = max(stat["count"], 1)
        mean_abs = stat["sum_abs"] / count
        mean_sq = stat["sum_sq"] / count
        variance = torch.clamp(mean_sq - mean_abs ** 2, min=0.0)
        effect = stat["sum_effect"] / count

        score = mean_abs
        if torch.max(variance) > 0:
            score = score + variance / torch.max(variance)
        if torch.max(effect) > 0:
            score = score + effect / torch.max(effect)

        values, indices = torch.topk(score, k=min(top_k, score.shape[0]))
        layer_rankings = []
        for value, idx in zip(values.tolist(), indices.tolist()):
            layer_rankings.append(
                {
                    "neuron_index": idx,
                    "score": value,
                    "mean_abs_activation": mean_abs[idx].item(),
                    "variance": variance[idx].item(),
                    "mean_direct_effect": effect[idx].item(),
                }
            )
        rankings[layer] = layer_rankings
    return rankings


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loader = MathDatasetLoader(args.datasets_root)
    data = loader.load(args.dataset, split=args.split)
    if args.sample_size and args.sample_size < len(data):
        data = random.sample(data, args.sample_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    device = torch.device(args.device)
    target_layers = get_target_layers(model, args.layers)
    stats = record_stats(model, tokenizer, data, target_layers, device)
    rankings = rank_neurons(stats, args.top_k)

    with output_path.open("w") as f:
        json.dump(rankings, f, indent=2)


if __name__ == "__main__":
    main()
