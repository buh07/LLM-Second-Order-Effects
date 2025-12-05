"""Compute neuron-wise PCA bases for LLM second-order tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tqdm

from phase2.utils import clean_model_name, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA for LLM second-order tensors")
    parser.add_argument("--dataset", type=str, default="mawps")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--mlp_layer", type=int, default=9)
    parser.add_argument("--input_dir", type=str, default="phase2_outputs")
    parser.add_argument("--output_dir", type=str, default="phase2_outputs")
    parser.add_argument("--top_k_pca", type=int, default=100)
    parser.add_argument("--max_examples", type=int, default=0, help="Optional limit for PCA computation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = f"{args.dataset}_{args.split}_{clean_model_name(args.model_name)}_layer{args.mlp_layer}"
    input_dir = Path(args.input_dir)
    array_path = input_dir / f"{prefix}_second_order_layer{args.mlp_layer}.npy"
    if not array_path.exists():
        raise FileNotFoundError(f"Second-order tensor not found at {array_path}")

    tensors = np.load(array_path, mmap_mode="r")
    if args.max_examples and args.max_examples > 0:
        tensors = tensors[: args.max_examples]
    num_examples, num_neurons, hidden_dim = tensors.shape
    top_k = min(args.top_k_pca, num_examples)
    if top_k < 2:
        raise ValueError("Need at least two examples to fit PCA.")

    neuron_mean = tensors.mean(axis=0)
    pcas = []
    norms = []
    for neuron_idx in tqdm.trange(num_neurons, desc="Neurons"):
        centered = tensors[:, neuron_idx] - neuron_mean[neuron_idx]
        sample_norms = np.linalg.norm(centered, axis=-1)
        important = np.argsort(sample_norms)[-top_k:]
        important_vectors = centered[important]
        norms.append(np.sort(sample_norms)[-top_k:])
        u, s, vh = np.linalg.svd(important_vectors, full_matrices=False)
        principal = vh[0]
        projections = important_vectors @ principal
        if (projections > 0).sum() < top_k // 2:
            principal = -principal
        pcas.append(principal)

    pcas = np.stack(pcas, axis=0)
    norms = np.stack(norms, axis=0)

    output_dir = ensure_dir(args.output_dir)
    pca_path = output_dir / f"{prefix}_{args.top_k_pca}_pca.npy"
    norm_path = output_dir / f"{prefix}_{args.top_k_pca}_norm.npy"
    np.save(pca_path, pcas)
    np.save(norm_path, norms)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "model_name": args.model_name,
        "mlp_layer": args.mlp_layer,
        "num_examples": num_examples,
        "num_neurons": num_neurons,
        "hidden_dim": hidden_dim,
        "top_k_pca": top_k,
        "input": str(array_path),
        "pca_path": str(pca_path),
        "norm_path": str(norm_path),
    }
    (output_dir / f"{prefix}_{args.top_k_pca}_pca_meta.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

