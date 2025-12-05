"""Sparse decomposition of LLM neuron PCAs using math lexicon embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tqdm

from phase2.utils import clean_model_name, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse decomposition for LLM neuron PCs")
    parser.add_argument("--dataset", type=str, default="mawps")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--mlp_layer", type=int, default=9)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--transform_alpha", type=float, default=1.0)
    parser.add_argument("--text_embeddings_path", type=str, required=True, help="Embeddings from compute_text_set_projection")
    parser.add_argument("--lexicon_path", type=str, required=True)
    parser.add_argument("--pca_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="phase2_outputs")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval_second_order_path", type=str, default="")
    parser.add_argument("--eval_labels_path", type=str, default="")
    parser.add_argument("--correctness_direction", type=str, default="")
    return parser.parse_args()


def load_lexicon(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def omp_decompose(dictionary: np.ndarray, vector: np.ndarray, components: int) -> np.ndarray:
    selected: list[int] = []
    residual = vector.copy()
    for _ in range(components):
        correlations = dictionary @ residual
        idx = int(np.argmax(np.abs(correlations)))
        if idx in selected:
            break
        selected.append(idx)
        subdict = dictionary[selected].T
        coeffs, *_ = np.linalg.lstsq(subdict, vector, rcond=None)
        residual = vector - subdict @ coeffs
        if np.linalg.norm(residual) < 1e-6:
            break
    result = np.zeros(dictionary.shape[0])
    if selected:
        subdict = dictionary[selected].T
        coeffs, *_ = np.linalg.lstsq(subdict, vector, rcond=None)
        for idx, coeff in zip(selected, coeffs):
            result[idx] = coeff
    return result


def build_decomposition(
    pca_vectors: np.ndarray,
    text_embeddings: np.ndarray,
    components: int,
) -> np.ndarray:
    dictionary = text_embeddings
    decomposition = np.zeros((pca_vectors.shape[0], dictionary.shape[0]))
    for idx in tqdm.trange(pca_vectors.shape[0], desc="OMP"):
        decomposition[idx] = omp_decompose(dictionary, pca_vectors[idx], components)
    return decomposition


def evaluate_reconstruction(
    decomposition: np.ndarray,
    text_embeddings: np.ndarray,
    pcas: np.ndarray,
    eval_tensor_path: Path,
    eval_labels_path: Path,
    correctness_direction_path: Path,
) -> dict:
    if not eval_tensor_path.exists():
        raise FileNotFoundError(f"Evaluation tensor not found at {eval_tensor_path}")
    if not eval_labels_path.exists():
        raise FileNotFoundError(f"Evaluation labels not found at {eval_labels_path}")
    if not correctness_direction_path.exists():
        raise FileNotFoundError(f"Correctness direction not found at {correctness_direction_path}")

    eval_tensors = np.load(eval_tensor_path, mmap_mode="r")
    labels = np.load(eval_labels_path, mmap_mode="r")
    direction = np.load(correctness_direction_path)

    reconstructed_pcas = decomposition @ text_embeddings  # [neurons, hidden]
    coeffs = np.einsum("bnh,nh->bn", eval_tensors, pcas)
    approx = coeffs[:, :, None] * reconstructed_pcas[None, :, :]

    diff = eval_tensors - approx
    rel_error = float(np.linalg.norm(diff) / np.linalg.norm(eval_tensors))

    baseline_scores = np.einsum("bnh,h->bn", eval_tensors, direction)
    approx_scores = np.einsum("bnh,h->bn", approx, direction)
    baseline_logits = baseline_scores.sum(axis=1)
    approx_logits = approx_scores.sum(axis=1)
    label_slice = labels[: baseline_logits.shape[0]]
    baseline_pred = baseline_logits > 0
    approx_pred = approx_logits > 0
    baseline_acc = float((baseline_pred == label_slice).mean())
    approx_acc = float((approx_pred == label_slice).mean())

    return {
        "relative_error": rel_error,
        "baseline_accuracy": baseline_acc,
        "approx_accuracy": approx_acc,
        "accuracy_drop": baseline_acc - approx_acc,
    }


def main() -> None:
    args = parse_args()
    prefix = f"{args.dataset}_{args.split}_{clean_model_name(args.model_name)}_layer{args.mlp_layer}"
    output_dir = ensure_dir(args.output_dir)

    text_embeddings = np.load(args.text_embeddings_path, mmap_mode="r")
    lexicon = load_lexicon(Path(args.lexicon_path))
    if text_embeddings.shape[0] != len(lexicon):
        raise ValueError("Lexicon length does not match embedding count")

    pca_vectors = np.load(args.pca_path, mmap_mode="r")
    decomposition = build_decomposition(
        pca_vectors,
        text_embeddings,
        components=args.components,
    )

    recon = decomposition @ text_embeddings
    json_map = {}
    for neuron_idx in tqdm.trange(decomposition.shape[0], desc="Formatting JSON"):
        coeffs = decomposition[neuron_idx]
        top_indices = np.argsort(np.abs(coeffs))[-args.components :]
        entries = []
        for idx in reversed(top_indices):
            entries.append(
                {
                    "lexicon_index": int(idx),
                    "coefficient": float(coeffs[idx]),
                    "text": lexicon[idx],
                }
            )
        json_map[neuron_idx] = entries

    name = (
        f"{prefix}_{Path(args.text_embeddings_path).stem}_omp_{args.transform_alpha}_{args.components}"
    )
    json_path = output_dir / f"{name}.json"
    npz_path = output_dir / f"{name}.npz"
    recon_path = output_dir / f"{name}_reconstructed_pcas.npy"

    json_path.write_text(json.dumps(json_map, indent=2))
    np.savez_compressed(npz_path, decomposition=decomposition)
    np.save(recon_path, recon)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "model_name": args.model_name,
        "mlp_layer": args.mlp_layer,
        "components": args.components,
        "transform_alpha": args.transform_alpha,
        "text_embeddings": args.text_embeddings_path,
        "lexicon": args.lexicon_path,
        "pca_path": args.pca_path,
        "output_prefix": name,
    }

    if args.evaluate:
        metrics = evaluate_reconstruction(
            decomposition,
            text_embeddings,
            pca_vectors,
            Path(args.eval_second_order_path),
            Path(args.eval_labels_path),
            Path(args.correctness_direction),
        )
        summary["evaluation"] = metrics
        print("Sparse decomposition evaluation:", json.dumps(metrics, indent=2))

    (output_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
