# Usage Guide

This project reimplements the CLIP second-order pipeline for math-focused LLMs.
The steps below cover everything implemented so far (Phases 1–2) and how to
switch between Phase 1 vs. Phase 2 correctness labels.

## 1. Environment Setup

```bash
source 2OE_env/bin/activate
pip install -r requirements.txt   # once we collect dependencies
```

The virtualenv already contains core Python packages; install `transformers`,
`torch`, etc. if they are missing.

## 2. Data Preparation

The math datasets live in `datasets/` using the structure expected by
`llm_datasets.math_dataset_loader`. Ensure the MAWPS parquet shard and CAMEL
JSON files exist before running any scripts.

## 3. Phase 1 – Evaluation + Correctness Labels

`phase1/evaluate_model.py` runs an LLM on a dataset, caches final-token
representations, and produces correctness records based on answer parsing.

```bash
python phase1/evaluate_model.py \
  --dataset mawps \
  --split train \
  --datasets_root datasets \
  --model_name gpt2 \
  --sample_size 512 \
  --output_dir phase1_outputs
```

Artifacts:

- `*_eval.jsonl`: per-example prompt/response/correctness metadata.
- `*_representations.npy`: final-token hidden states.
- `*_correctness.npy`: binary labels.
- `*_correctness_direction.npy`: (optional) quick direction from Phase 1.

## 4. Phase 1 – Layer Ablations & Neuron Ranking

Use `phase1/layer_ablation.py` to identify critical layers and
`phase1/rank_neurons.py` to score neurons with the tri-criteria heuristic. These
files inform which layers/neurons to analyze in Phase 2 but produce only JSON
reports, so they are optional for reproducing the artifacts referenced later.

## 5. Phase 2 – Representation Capture

`phase2/compute_representations.py` mirrors CLIP's `compute_representations.py`
and caches token-level residual streams plus metadata.

```bash
python phase2/compute_representations.py \
  --dataset mawps \
  --split train \
  --datasets_root datasets \
  --model_name gpt2 \
  --batch_size 8 \
  --max_length 256 \
  --sample_size 512 \
  --output_dir phase2_outputs
```

Outputs (`phase2_outputs/{dataset}_{split}_{model}/...` naming):

- `*_residual_stream.npy`: `[N, max_len, hidden]` normalized residuals.
- `*_tokens.npy`, `*_attention_mask.npy`: tokenizer outputs.
- `*_metadata.jsonl`: prompt + answer metadata for each index.
- `*_stats.json`: config summary.

## 6. Phase 2 – Correctness Projection

`phase2/compute_correctness_projection.py` consumes the residual tensors and
produces the correctness direction that downstream second-order scripts will
use. It also logs correctness records in a uniform format.

### Switching Label Sources

- **Default (`--label_source phase1`)**: reuse existing Phase 1
  `*_eval.jsonl` entries, so we avoid regenerating answers. This requires that
  Phase 1 was run with the same dataset split and prompt template.
- **Alternative (`--label_source phase2`)**: set this flag to regenerate
  answers during Phase 2. The script calls the LLM again, parses the responses,
  and saves the new labels.

Example command (Phase 1 labels):

```bash
python phase2/compute_correctness_projection.py \
  --dataset mawps \
  --split train \
  --model_name gpt2 \
  --output_dir phase2_outputs
```

To switch later:

```bash
python phase2/compute_correctness_projection.py \
  --dataset mawps \
  --split train \
  --model_name gpt2 \
  --output_dir phase2_outputs \
  --label_source phase2 \
  --max_new_tokens 64
```

Artifacts:

- `*_correctness_direction.npy`: mean(correct) – mean(incorrect) vector.
- `*_correctness_labels.npy`: binary labels aligned with representations.
- `*_correctness_records.jsonl`: per-example log including `label_source`.
- Updated `*_stats.json` with accuracy and label provenance.

## 7. Phase 2 – Second-Order Neuron Capture

`phase2/compute_second_order_neuron_prs.py` ports CLIP's second-order script to
LLMs. It reuses the hooks in `phase2/gpt_second_order_hook.py`, propagates each
selected neuron through downstream attention heads, and projects the resulting
contribution onto the correctness direction.

Key requirements:

- A correctness direction `.npy` file (from the previous step).
- Optionally, a Phase 1 neuron-ranking JSON to limit the number of neurons.
- Small batch sizes (defaults to 1) to keep `[tokens × neurons × hidden]`
  tensors tractable.

Example invocation:

```bash
PYTHONPATH=. python phase2/compute_second_order_neuron_prs.py \
  --dataset mawps \
  --split train \
  --datasets_root datasets \
  --model_name gpt2 \
  --mlp_layer 8 \
  --correctness_direction phase2_outputs/mawps_train_gpt2_correctness_direction.npy \
  --neuron_indices_path phase1_outputs/neuron_ranking.json \
  --max_neurons 100 \
  --output_dir phase2_outputs
```

Outputs:

- `*_second_order_layer{L}.npy`: `[num_examples, num_neurons]` matrix of
  correctness-projected second-order contributions.
- `*_second_order_meta.json`: configuration summary for reproducibility.

## 7. Next Steps (Phase 2+)

The remaining Phase 2 items (second-order hooks, PCA, sparse decomposition, and
ablations) will reuse the files above once implemented. Follow the CLIP scripts
(`CLIP-2OE/*.py`) for naming conventions so downstream stages remain compatible.
