#!/usr/bin/env bash

# Run the entire GPT-2 pipeline (Phase 1 + Phase 2) for MAWPS.
# This script is intended to be executed inside the 2OE_env virtualenv.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="false"

DATASET="mawps"
MODEL_NAME="gpt2"
MLP_LAYER=8
SAMPLE_SIZE=256
MAX_LENGTH=256
MAX_NEW_TOKENS=64
BATCH_SIZE_REP=8
NEURON_LIMIT=64
OMP_COMPONENTS=16
PROMPT_TEMPLATE=$'Solve the following problem and answer with just a number:\n{problem}\nAnswer:'

log() {
  echo "[$(date --iso-8601=seconds)] $*"
}

run_phase1() {
  local split="$1"
  log "Phase 1 evaluate_model (${split})"
  python phase1/evaluate_model.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --datasets_root datasets \
    --model_name "${MODEL_NAME}" \
    --device cuda \
    --sample_size "${SAMPLE_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --output_dir phase1_outputs
}

run_representations() {
  local split="$1"
  log "Phase 2 compute_representations (${split})"
  python phase2/compute_representations.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --datasets_root datasets \
    --model_name "${MODEL_NAME}" \
    --device cuda \
    --batch_size "${BATCH_SIZE_REP}" \
    --max_length "${MAX_LENGTH}" \
    --sample_size "${SAMPLE_SIZE}" \
    --prompt_template "${PROMPT_TEMPLATE}" \
    --output_dir phase2_outputs
}

run_correctness_projection() {
  local split="$1"
  log "Phase 2 compute_correctness_projection (${split})"
  python phase2/compute_correctness_projection.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --datasets_root datasets \
    --model_name "${MODEL_NAME}" \
    --device cuda \
    --batch_size 8 \
    --max_length "${MAX_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --sample_size "${SAMPLE_SIZE}" \
    --output_dir phase2_outputs \
    --phase1_output_dir phase1_outputs \
    --label_source phase1
}

run_second_order() {
  local split="$1"
  local direction="phase2_outputs/${DATASET}_${split}_${MODEL_NAME}_correctness_direction.npy"
  log "Phase 2 compute_second_order_neuron_prs (${split})"
  python phase2/compute_second_order_neuron_prs.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --datasets_root datasets \
    --model_name "${MODEL_NAME}" \
    --device cuda \
    --batch_size 1 \
    --max_length "${MAX_LENGTH}" \
    --sample_size "${SAMPLE_SIZE}" \
    --prompt_template "${PROMPT_TEMPLATE}" \
    --mlp_layer "${MLP_LAYER}" \
    --max_neurons "${NEURON_LIMIT}" \
    --correctness_direction "${direction}" \
    --output_dir phase2_outputs
}

log "=== Phase 3 GPT-2 pipeline start ==="

run_phase1 "train"
run_phase1 "val"

run_representations "train"
run_representations "val"

run_correctness_projection "train"
run_correctness_projection "val"

run_second_order "train"
run_second_order "val"

log "Phase 2 compute_pcas (train)"
python phase2/compute_pcas.py \
  --dataset "${DATASET}" \
  --split "train" \
  --model_name "${MODEL_NAME}" \
  --mlp_layer "${MLP_LAYER}" \
  --input_dir phase2_outputs \
  --output_dir phase2_outputs \
  --top_k_pca "${NEURON_LIMIT}"

log "Phase 2 compute_text_set_projection"
python phase2/compute_text_set_projection.py \
  --lexicon_path phase2/math_lexicon.txt \
  --model_name "${MODEL_NAME}" \
  --device cuda \
  --output_dir phase2_outputs

log "Phase 2 compute_sparse_decomposition"
python phase2/compute_sparse_decomposition.py \
  --dataset "${DATASET}" \
  --split "train" \
  --model_name "${MODEL_NAME}" \
  --mlp_layer "${MLP_LAYER}" \
  --components "${OMP_COMPONENTS}" \
  --text_embeddings_path "phase2_outputs/math_lexicon_${MODEL_NAME}.npy" \
  --lexicon_path phase2/math_lexicon.txt \
  --pca_path "phase2_outputs/${DATASET}_train_${MODEL_NAME}_layer${MLP_LAYER}_${NEURON_LIMIT}_pca.npy" \
  --output_dir phase2_outputs \
  --evaluate \
  --eval_second_order_path "phase2_outputs/${DATASET}_val_${MODEL_NAME}_layer${MLP_LAYER}_second_order_layer${MLP_LAYER}.npy" \
  --eval_labels_path "phase2_outputs/${DATASET}_val_${MODEL_NAME}_correctness_labels.npy" \
  --correctness_direction "phase2_outputs/${DATASET}_val_${MODEL_NAME}_correctness_direction.npy"

log "Phase 2 compute_ablations (val)"
python phase2/compute_ablations.py \
  --dataset "${DATASET}" \
  --split "val" \
  --datasets_root datasets \
  --model_name "${MODEL_NAME}" \
  --device cuda \
  --batch_size 1 \
  --max_length "${MAX_LENGTH}" \
  --sample_size "${SAMPLE_SIZE}" \
  --prompt_template "${PROMPT_TEMPLATE}" \
  --mlp_layer "${MLP_LAYER}" \
  --max_neurons "${NEURON_LIMIT}" \
  --train_second_order_path "phase2_outputs/${DATASET}_train_${MODEL_NAME}_layer${MLP_LAYER}_second_order_layer${MLP_LAYER}.npy" \
  --pca_path "phase2_outputs/${DATASET}_train_${MODEL_NAME}_layer${MLP_LAYER}_${NEURON_LIMIT}_pca.npy" \
  --norm_path "phase2_outputs/${DATASET}_train_${MODEL_NAME}_layer${MLP_LAYER}_${NEURON_LIMIT}_norm.npy" \
  --correctness_direction "phase2_outputs/${DATASET}_val_${MODEL_NAME}_correctness_direction.npy" \
  --correctness_labels "phase2_outputs/${DATASET}_val_${MODEL_NAME}_correctness_labels.npy" \
  --output_json "phase2_outputs/${DATASET}_val_${MODEL_NAME}_ablations.json"

log "=== Phase 3 GPT-2 pipeline complete ==="
