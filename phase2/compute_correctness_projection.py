"""Compute correctness direction vectors and optional logit probes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_datasets.math_dataset_loader import MathDatasetLoader, MathExample
from phase2.prompts import build_inference_prompt, extract_numeric_answer
from phase2.utils import clean_model_name, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM analogue of compute_classifier_projection.py"
    )
    parser.add_argument("--dataset", type=str, default="mawps")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--datasets_root", type=str, default="datasets")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256, help="Tokenization length (must match representations step)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="phase2_outputs")
    parser.add_argument("--representations_path", type=str, default="")
    parser.add_argument("--attention_mask_path", type=str, default="")
    parser.add_argument("--stats_path", type=str, default="")
    parser.add_argument(
        "--label_source",
        type=str,
        choices=("phase1", "phase2"),
        default="phase1",
        help="Use Phase 1 evaluation labels by default; pass 'phase2' to recompute via generation.",
    )
    parser.add_argument("--phase1_output_dir", type=str, default="phase1_outputs")
    parser.add_argument(
        "--probe_tokens",
        type=str,
        default="",
        help="Comma-separated list of tokens for correctness-direction dot products",
    )
    return parser.parse_args()


def default_prefix_paths(args: argparse.Namespace, prefix: str) -> tuple[Path, Path, Path]:
    base_dir = Path(args.output_dir)
    residual_path = Path(args.representations_path) if args.representations_path else base_dir / f"{prefix}_residual_stream.npy"
    mask_path = Path(args.attention_mask_path) if args.attention_mask_path else base_dir / f"{prefix}_attention_mask.npy"
    stats_path = Path(args.stats_path) if args.stats_path else base_dir / f"{prefix}_stats.json"
    return residual_path, mask_path, stats_path


def load_examples(args: argparse.Namespace) -> List[MathExample]:
    loader = MathDatasetLoader(args.datasets_root)
    examples = loader.load(args.dataset, split=args.split)
    if args.sample_size and args.sample_size > 0:
        examples = examples[: min(args.sample_size, len(examples))]
    return examples


def load_phase1_label_map(dataset: str, split: str, phase1_dir: str) -> dict:
    records_path = Path(phase1_dir) / f"{dataset}_{split}_eval.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(
            f"Phase 1 correctness log not found at {records_path}. "
            "Run phase1/evaluate_model.py or switch --label_source to phase2."
        )
    record_map: dict[str, List[dict]] = defaultdict(list)
    with records_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            record_map[data["prompt"]].append(data)
    return record_map


def align_phase1_labels(prompts: Sequence[str], record_map: dict[str, List[dict]]) -> List[dict]:
    missing: List[str] = []
    ordered: List[dict] = []
    for prompt in prompts:
        entries = record_map.get(prompt)
        if not entries:
            missing.append(prompt)
            continue
        ordered.append(entries.pop(0))
    if missing:
        raise ValueError(
            "Phase 1 correctness prompts do not match the representation order. "
            "Ensure both stages used the same dataset sample and prompt template. "
            f"Missing prompts: {missing[:3]}{'...' if len(missing) > 3 else ''}"
        )
    return ordered


def batched(prompts: Sequence[str], batch_size: int):
    for start in range(0, len(prompts), batch_size):
        yield start, prompts[start : start + batch_size]


def run_generation(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device,
    max_new_tokens: int,
    batch_size: int,
    max_length: int,
) -> List[str]:
    target_length = max_length or tokenizer.model_max_length
    generations: List[str] = [""] * len(prompts)
    for start, chunk in batched(prompts, batch_size):
        encodings = tokenizer(
            list(chunk),
            padding=True,
            truncation=True,
            max_length=target_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[-1]
        responses = output_ids[:, input_len:]
        decoded = tokenizer.batch_decode(responses, skip_special_tokens=True)
        for idx, text in enumerate(decoded):
            generations[start + idx] = text.strip()
    return generations


def build_records_from_generation(
    examples: Sequence[MathExample],
    prompts: Sequence[str],
    responses: Sequence[str],
) -> List[dict]:
    records: List[dict] = []
    for idx, (example, prompt_text, text) in enumerate(zip(examples, prompts, responses)):
        predicted = extract_numeric_answer(text)
        target = extract_numeric_answer(example.answer)
        correct = bool(predicted) and predicted == target and target != ""
        records.append(
            {
                "index": idx,
                "prompt": prompt_text,
                "response_text": text,
                "predicted_answer": predicted,
                "target_answer": example.answer,
                "correct": correct,
                "metadata": example.metadata,
            }
        )
    return records


def select_last_token_vectors(residuals: np.memmap, attention_mask: np.memmap) -> np.ndarray:
    mask = np.asarray(attention_mask, dtype=np.int64)
    lengths = mask.sum(axis=1)
    if np.any(lengths == 0):
        raise RuntimeError("Found empty sequence when computing correctness direction")
    positions = lengths - 1
    indices = np.arange(residuals.shape[0])
    return np.asarray(residuals[indices, positions, :])


def compute_direction(vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positives = vectors[labels == 1]
    negatives = vectors[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise RuntimeError("Need at least one correct and incorrect example to compute correctness direction")
    return positives.mean(axis=0) - negatives.mean(axis=0)


def build_probe_scores(direction: np.ndarray, tokenizer, model, probe_tokens: Sequence[str]) -> dict:
    if not probe_tokens:
        return {}
    embedding = model.get_output_embeddings()
    if embedding is None or not hasattr(embedding, "weight"):
        return {}
    weight = embedding.weight.detach().cpu().numpy()
    scores = {}
    for token_str in probe_tokens:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if not ids:
            continue
        token_vec = weight[ids].mean(axis=0)
        scores[token_str] = float(np.dot(direction, token_vec))
    return scores


def compute_correctness_projection(args: argparse.Namespace) -> None:
    examples = load_examples(args)
    if not examples:
        raise RuntimeError(f"No examples found for dataset={args.dataset} split={args.split}")

    prefix = f"{args.dataset}_{args.split}_{clean_model_name(args.model_name)}"
    residual_path, mask_path, stats_path = default_prefix_paths(args, prefix)
    stats_data = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    if "max_length" in stats_data:
        args.max_length = int(stats_data["max_length"])
    residuals = np.lib.format.open_memmap(residual_path, mode="r")
    attention_mask = np.lib.format.open_memmap(mask_path, mode="r")

    if residuals.shape[0] != len(examples):
        raise RuntimeError(
            f"Representation count ({residuals.shape[0]}) does not match dataset size ({len(examples)}). "
            "Ensure the same sampling order was used."
        )

    output_dir = ensure_dir(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    prompts = [build_inference_prompt(ex) for ex in examples]
    if args.label_source == "phase1":
        record_map = load_phase1_label_map(args.dataset, args.split, args.phase1_output_dir)
        ordered_records = align_phase1_labels(prompts, record_map)
    else:
        responses = run_generation(
            model,
            tokenizer,
            prompts,
            torch.device(args.device),
            args.max_new_tokens,
            args.batch_size,
            args.max_length,
        )
        ordered_records = build_records_from_generation(examples, prompts, responses)

    labels = np.array([1 if rec["correct"] else 0 for rec in ordered_records], dtype=np.int32)
    records_path = output_dir / f"{prefix}_correctness_records.jsonl"
    with records_path.open("w") as rec_file:
        for idx, rec in enumerate(ordered_records):
            rec_out = dict(rec)
            rec_out["index"] = idx
            rec_out["label_source"] = args.label_source
            rec_file.write(json.dumps(rec_out) + "\n")

    vectors = select_last_token_vectors(residuals, attention_mask)
    direction = compute_direction(vectors, labels)

    np.save(output_dir / f"{prefix}_correctness_direction.npy", direction)
    np.save(output_dir / f"{prefix}_correctness_labels.npy", labels)

    stats_data.update(
        {
            "total_examples": len(examples),
            "num_correct": int(labels.sum()),
            "num_incorrect": int((labels == 0).sum()),
            "accuracy": float(labels.mean()),
            "correctness_label_source": args.label_source,
        }
    )
    stats_path.write_text(json.dumps(stats_data, indent=2))

    probes = [tok.strip() for tok in args.probe_tokens.split(",") if tok.strip()]
    probe_scores = build_probe_scores(direction, tokenizer, model, probes)
    if probe_scores:
        probe_path = output_dir / f"{prefix}_probe_scores.json"
        probe_path.write_text(json.dumps(probe_scores, indent=2))


def main() -> None:
    compute_correctness_projection(parse_args())


if __name__ == "__main__":
    main()
