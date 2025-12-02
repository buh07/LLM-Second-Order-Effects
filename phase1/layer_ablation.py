import argparse
import json
import random
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_datasets.math_dataset_loader import MathDatasetLoader, MathExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layerwise ablation sweep for math datasets.")
    parser.add_argument("--dataset", type=str, default="mawps")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--datasets_root", type=str, default="datasets")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=str, default="phase1_outputs/layer_ablation.json")
    return parser.parse_args()


def build_target_text(example: MathExample) -> str:
    if not example.answer:
        return example.prompt
    return f"{example.prompt}\nAnswer: {example.answer}"


def compute_average_loss(
    model,
    tokenizer,
    examples: List[MathExample],
    device: torch.device,
    hook_layer: int | None = None,
) -> float:
    hook_handle = None
    if hook_layer is not None:
        block = model.transformer.h[hook_layer]

        def zero_mlp_output(_module, _inputs, output):
            return torch.zeros_like(output)

        hook_handle = block.mlp.register_forward_hook(zero_mlp_output)

    total_loss = 0.0
    total_tokens = 0
    for example in examples:
        text = build_target_text(example)
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in encoded.items()}
        labels = inputs["input_ids"].clone()
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
        total_loss += outputs.loss.item() * labels.shape[-1]
        total_tokens += labels.shape[-1]

    if hook_handle is not None:
        hook_handle.remove()

    return total_loss / max(total_tokens, 1)


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
    baseline_loss = compute_average_loss(model, tokenizer, data, device)
    results = [{"layer": None, "loss": baseline_loss, "delta": 0.0}]

    for layer_idx in range(len(model.transformer.h)):
        ablated_loss = compute_average_loss(model, tokenizer, data, device, hook_layer=layer_idx)
        results.append(
            {
                "layer": layer_idx,
                "loss": ablated_loss,
                "delta": ablated_loss - baseline_loss,
            }
        )

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
