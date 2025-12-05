"""Prompt construction helpers shared across Phase 2 scripts."""

from __future__ import annotations

from dataclasses import dataclass

from llm_datasets.math_dataset_loader import MathExample


DEFAULT_PROMPT_TEMPLATE = "Solve the following problem:\n{problem}\nAnswer:"


def build_inference_prompt(example: MathExample, template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    """Return the text shown to the language model for inference."""

    return template.format(problem=example.prompt.strip())


def build_supervised_prompt(example: MathExample) -> str:
    """Prompt that includes the reference answer for teacher-forcing tasks."""

    answer = example.answer or example.reasoning
    return f"{build_inference_prompt(example)} {answer}".strip()


def extract_numeric_answer(text: str) -> str:
    """Heuristic to recover the final numeric answer from generated text."""

    if not text:
        return ""
    tokens = [tok.strip().lower() for tok in text.replace(",", " ").split()]
    numbers = [tok for tok in tokens if any(ch.isdigit() for ch in tok)]
    return numbers[-1] if numbers else (tokens[-1] if tokens else "")


@dataclass(frozen=True)
class RecordMetadata:
    """Minimal metadata that accompanies saved representations."""

    dataset: str
    split: str
    index: int
    prompt: str
    answer: str

