"""Compute normalized residual-stream representations for math datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_datasets.math_dataset_loader import MathDatasetLoader, MathExample
from phase2.prompts import (
    RecordMetadata,
    build_inference_prompt,
    DEFAULT_PROMPT_TEMPLATE,
)
from phase2.utils import chunked, clean_model_name, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM analogue of CLIP compute_representations.py"
    )
    parser.add_argument("--dataset", type=str, default="mawps")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--datasets_root", type=str, default="datasets")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=0, help="Optional limit on dataset size")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--output_dir", type=str, default="phase2_outputs")
    parser.add_argument("--normalize", action="store_true", help="Normalize token representations to unit norm")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    return parser.parse_args()


def prepare_dataset(args: argparse.Namespace) -> List[MathExample]:
    loader = MathDatasetLoader(args.datasets_root)
    examples = loader.load(args.dataset, split=args.split)
    if args.sample_size and args.sample_size > 0:
        examples = examples[: min(args.sample_size, len(examples))]
    return examples


def allocate_memmaps(
    output_dir: Path,
    prefix: str,
    num_examples: int,
    max_length: int,
    hidden_size: int,
):
    residual_path = output_dir / f"{prefix}_residual_stream.npy"
    tokens_path = output_dir / f"{prefix}_tokens.npy"
    mask_path = output_dir / f"{prefix}_attention_mask.npy"

    residual_map = np.lib.format.open_memmap(
        residual_path,
        mode="w+",
        shape=(num_examples, max_length, hidden_size),
        dtype=np.float32,
    )
    tokens_map = np.lib.format.open_memmap(
        tokens_path,
        mode="w+",
        shape=(num_examples, max_length),
        dtype=np.int32,
    )
    mask_map = np.lib.format.open_memmap(
        mask_path,
        mode="w+",
        shape=(num_examples, max_length),
        dtype=np.uint8,
    )
    return residual_map, tokens_map, mask_map


def encode_prompts(
    tokenizer,
    prompts: Sequence[str],
    max_length: int,
):
    return tokenizer(
        list(prompts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def compute_representations(args: argparse.Namespace) -> None:
    examples = prepare_dataset(args)
    if not examples:
        raise RuntimeError(f"No examples found for dataset={args.dataset} split={args.split}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    hidden_size = getattr(model.config, "hidden_size", getattr(model.config, "n_embd"))
    output_dir = ensure_dir(Path(args.output_dir))
    prefix = f"{args.dataset}_{args.split}_{clean_model_name(args.model_name)}"

    residual_map, tokens_map, mask_map = allocate_memmaps(
        output_dir, prefix, len(examples), args.max_length, hidden_size
    )
    metadata_path = output_dir / f"{prefix}_metadata.jsonl"

    with metadata_path.open("w") as meta_file:
        for start, batch_examples in chunked(examples, args.batch_size):
            prompts = [build_inference_prompt(ex, args.prompt_template) for ex in batch_examples]
            encodings = encode_prompts(tokenizer, prompts, args.max_length)
            inputs = {k: v.to(args.device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
            last_hidden = outputs.hidden_states[-1]
            if args.normalize:
                norms = torch.linalg.norm(last_hidden, dim=-1, keepdim=True).clamp(min=1e-6)
                last_hidden = last_hidden / norms

            batch_size = last_hidden.shape[0]
            end = start + batch_size
            residual_map[start:end] = last_hidden.detach().cpu().float().numpy()
            tokens_map[start:end] = encodings["input_ids"].cpu().numpy()
            mask_map[start:end] = encodings["attention_mask"].cpu().numpy()

            for offset, (example, prompt_text) in enumerate(zip(batch_examples, prompts)):
                metadata = RecordMetadata(
                    dataset=example.dataset,
                    split=example.split,
                    index=start + offset,
                    prompt=prompt_text,
                    answer=example.answer,
                )
                meta_file.write(json.dumps(metadata.__dict__) + "\n")

    stats = {
        "num_examples": len(examples),
        "max_length": args.max_length,
        "hidden_size": hidden_size,
        "model_name": args.model_name,
        "normalize": args.normalize,
        "prompt_template": args.prompt_template,
        "dataset": args.dataset,
        "split": args.split,
    }
    stats_path = output_dir / f"{prefix}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))


def main() -> None:
    compute_representations(parse_args())


if __name__ == "__main__":
    main()

