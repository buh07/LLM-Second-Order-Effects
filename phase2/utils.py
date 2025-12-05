"""Reusable helpers for Phase 2 scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple, TypeVar


T = TypeVar("T")


def ensure_dir(path: Path | str) -> Path:
    """Create `path` if it does not exist and return it as a Path."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunked(items: Sequence[T], batch_size: int) -> Iterator[Tuple[int, Sequence[T]]]:
    """Yield (start_index, slice) pairs for batched iteration."""

    n = len(items)
    for start in range(0, n, batch_size):
        yield start, items[start : start + batch_size]


def clean_model_name(model_name: str) -> str:
    """Map a Hugging Face model identifier to a filesystem-friendly name."""

    safe = model_name.replace("/", "_").replace(":", "_")
    safe = safe.replace(" ", "_")
    return safe

