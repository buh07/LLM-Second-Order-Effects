"""Utilities for loading math/logic datasets used in Phase 1.

This module normalizes multiple sources (MAWPS, CAMEL math) into a single
question/answer schema so downstream code can treat them uniformly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd


@dataclass(frozen=True)
class MathExample:
    """Single math problem description with normalized fields."""

    dataset: str
    split: str
    prompt: str
    answer: str
    reasoning: str
    metadata: Dict[str, str]


class MathDatasetLoader:
    """Loads math datasets from local HuggingFace mirrors.

    Parameters
    ----------
    datasets_root:
        Path that contains the checked-out datasets/ directory.
    """

    SUPPORTED_DATASETS = ("mawps", "camel_math")

    def __init__(self, datasets_root: Path | str = Path("datasets")) -> None:
        self.datasets_root = Path(datasets_root)

    def load(
        self,
        dataset: str,
        split: str = "train",
        limit: Optional[int] = None,
    ) -> List[MathExample]:
        """Return a materialized list of MathExample objects."""

        dataset = dataset.lower()
        if dataset in {"mawps", "garrethlee_mawps"}:
            iterator = self._iter_mawps(split=split)
        elif dataset in {"camel_math", "camel-ai_math", "camel"}:
            iterator = self._iter_camel_math(split=split)
        else:
            raise ValueError(
                f"Unsupported dataset {dataset}. Supported: {self.SUPPORTED_DATASETS}"
            )

        if limit is None:
            return list(iterator)
        return [ex for _, ex in zip(range(limit), iterator)]

    def _iter_mawps(self, split: str) -> Iterator[MathExample]:
        """Yield MAWPS problems from the parquet shards."""

        dataset_dir = self.datasets_root / "garrethlee_MAWPS" / "data"
        shard_name = f"{split}-00000-of-00001.parquet"
        shard_path = dataset_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(
                f"Expected MAWPS shard at {shard_path}. "
                "Ensure the dataset was cloned via git-lfs."
            )
        frame = pd.read_parquet(shard_path)
        for row in frame.itertuples(index=False):
            reasoning, final_answer = self._split_mawps_answer(row.answer)
            yield MathExample(
                dataset="mawps",
                split=split,
                prompt=str(row.question).strip(),
                answer=final_answer,
                reasoning=reasoning,
                metadata={"raw_answer": str(row.answer).strip()},
            )

    def _iter_camel_math(self, split: str) -> Iterator[MathExample]:
        """Yield CAMEL math problems from the extracted JSON files."""

        dataset_dir = self.datasets_root / "camel_ai_math"
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Expected CAMEL math directory at {dataset_dir}. "
                "Clone https://huggingface.co/datasets/camel-ai/math first."
            )
        json_files = sorted(dataset_dir.glob("*.json"))
        if not json_files:
            raise RuntimeError(
                "No JSON files detected in CAMEL math directory. "
                "Did you unzip math.zip?"
            )
        for path in json_files:
            record = json.loads(path.read_text())
            prompt = str(record.get("message_1", "")).strip()
            solution = str(record.get("message_2", "")).strip()
            final_answer = self._extract_final_number(solution)
            metadata = {
                "topic": str(record.get("topic") or record.get("topic;", "")).strip(),
                "sub_topic": str(record.get("sub_topic", "")).strip(),
                "role": str(record.get("role_1", "")).strip(),
                "source_file": path.name,
            }
            if final_answer is not None:
                metadata["raw_answer"] = final_answer
            yield MathExample(
                dataset="camel_math",
                split=split,
                prompt=prompt,
                answer=final_answer or "",
                reasoning=solution,
                metadata=metadata,
            )

    @staticmethod
    def _split_mawps_answer(raw_answer: str) -> tuple[str, str]:
        """Split MAWPS answer text into reasoning steps and final answer."""

        if raw_answer is None:
            return "", ""
        parts = raw_answer.split("####")
        reasoning = parts[0].strip()
        final = parts[1].strip() if len(parts) > 1 else reasoning
        return reasoning, final

    @staticmethod
    def _extract_final_number(text: str) -> Optional[str]:
        """Heuristic extraction of the last numeric answer in CAMEL solutions."""

        if not text:
            return None
        matches = re.findall(r"-?\d+(?:\.\d+)?", text)
        if not matches:
            return None
        return matches[-1]
