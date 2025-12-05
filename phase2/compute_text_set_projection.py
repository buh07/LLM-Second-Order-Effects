"""Embed math/logic lexicon entries using the target LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase2.utils import clean_model_name, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed math/logic lexicon entries")
    parser.add_argument("--lexicon_path", type=str, default="phase2/math_lexicon.txt")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="phase2_outputs")
    parser.add_argument("--coefficient", type=float, default=100.0)
    return parser.parse_args()


def chunk_lines(lines: List[str], batch_size: int):
    for idx in range(0, len(lines), batch_size):
        yield lines[idx : idx + batch_size]


def main() -> None:
    args = parse_args()
    lexicon_path = Path(args.lexicon_path)
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon file not found at {lexicon_path}")
    entries = [line.strip() for line in lexicon_path.read_text().splitlines() if line.strip()]
    if not entries:
        raise RuntimeError("Lexicon is empty.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    embeddings = []
    device = torch.device(args.device)
    for batch in chunk_lines(entries, args.batch_size):
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        attention_mask = encoded["attention_mask"].to(device)
        inputs = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        averaged = summed / counts
        normalized = torch.nn.functional.normalize(averaged, dim=-1)
        embeddings.append(normalized.cpu().numpy())

    features = np.concatenate(embeddings, axis=0) * args.coefficient
    output_dir = ensure_dir(args.output_dir)
    name = lexicon_path.stem
    suffix = f"{name}_{clean_model_name(args.model_name)}.npy"
    out_path = output_dir / suffix
    np.save(out_path, features)

    summary = {
        "lexicon": str(lexicon_path),
        "num_entries": len(entries),
        "model_name": args.model_name,
        "embedding_path": str(out_path),
        "coefficient": args.coefficient,
    }
    (output_dir / f"{suffix}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

