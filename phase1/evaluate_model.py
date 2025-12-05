import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_datasets.math_dataset_loader import MathDatasetLoader, MathExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an LLM on math datasets.")
    parser.add_argument("--dataset", type=str, default="mawps", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--datasets_root", type=str, default="datasets")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_size", type=int, default=512, help="Limit on dataset size (0 = full dataset)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset before sampling (useful for randomized evaluation subsets).",
    )
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=False)
    parser.add_argument("--output_dir", type=str, default="phase1_outputs")
    return parser.parse_args()


def build_prompt(example: MathExample) -> str:
    return (
        "Solve the following problem and answer with just a number:\n"
        f"{example.prompt.strip()}\n"
        "Answer:"
    )


def extract_numeric_answer(text: str) -> str:
    if not text:
        return ""
    numbers = [token for token in text.replace(",", " ").split() if any(ch.isdigit() for ch in token)]
    if not numbers:
        return text.strip().lower()
    return numbers[-1].strip().lower()


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer,
    examples: List[MathExample],
    device: torch.device,
    max_new_tokens: int,
) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    records = []
    representations = []
    labels = []

    for idx, example in enumerate(examples):
        prompt = build_prompt(example)
        encoded = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        response_ids = generated[:, inputs["input_ids"].shape[-1] :]
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
        predicted_answer = extract_numeric_answer(response_text)
        target_answer = extract_numeric_answer(example.answer)
        correct = bool(predicted_answer) and predicted_answer == target_answer and target_answer != ""
        labels.append(int(correct))

        full_ids = generated
        full_attention = torch.ones_like(full_ids, device=device)
        with torch.no_grad():
            outputs = model(full_ids, attention_mask=full_attention, output_hidden_states=True, use_cache=False)
        final_hidden = outputs.hidden_states[-1][:, -1, :].detach().cpu().numpy()[0]
        representations.append(final_hidden)

        records.append(
            {
                "index": idx,
                "prompt": prompt,
                "target_answer": example.answer,
                "predicted_answer": predicted_answer,
                "response_text": response_text,
                "correct": correct,
                "metadata": example.metadata,
            }
        )
    return records, np.stack(representations, axis=0), np.array(labels, dtype=np.int32)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = MathDatasetLoader(args.datasets_root)
    data = loader.load(args.dataset, split=args.split)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(data)
    if args.sample_size and args.sample_size > 0:
        data = data[: min(args.sample_size, len(data))]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    records, representations, labels = evaluate_model(
        model,
        tokenizer,
        data,
        torch.device(args.device),
        args.max_new_tokens,
    )

    records_path = output_dir / f"{args.dataset}_{args.split}_eval.jsonl"
    with records_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    np.save(output_dir / f"{args.dataset}_{args.split}_representations.npy", representations)
    np.save(output_dir / f"{args.dataset}_{args.split}_correctness.npy", labels)

    positives = labels == 1
    negatives = labels == 0
    if positives.any() and negatives.any():
        correctness_direction = representations[positives].mean(axis=0) - representations[negatives].mean(axis=0)
        np.save(output_dir / f"{args.dataset}_{args.split}_correctness_direction.npy", correctness_direction)


if __name__ == "__main__":
    main()
