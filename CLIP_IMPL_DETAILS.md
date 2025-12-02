# CLIP Implementation Details (Phase 0 Findings)

This document records the Phase 0 audit of the original CLIP second-order effects pipeline located in `CLIP-2OE/`. It captures every stage we need to mirror before adapting the framework to language models.

## Repository Layout Worth Mirroring

- `CLIP-2OE/compute_*.py` scripts ─ Single-responsibility entry points for each processing stage, each built as a CLI with `argparse`.
- `CLIP-2OE/utils/` ─ Common infrastructure:
  - `factory.py` constructs CLIP models, tokenizers, and preprocessing transforms.
  - `hook.py` implements the custom hook manager used by all PRS loggers.
  - `subsampled_imagenet.py` defines lightweight datasets (train/val/ImageNet-R) used across stages.
  - Supporting modules for vocabularies, tokenization, data transforms, and visualization.
- `prs_*.py` ─ Specialized hook definitions for capturing first/second-order statistics, ablation switches, and targeted neuron recordings.
- `text_descriptions/` ─ Source vocabularies for sparse decomposition (`30k.txt`, etc.).

## Stage-by-Stage Breakdown

### 1. `compute_representations.py`
- **Goal**: Run CLIP on a dataset and cache normalized image representations plus class labels. These representations become the baseline activations and normalization factors for later stages.
- **Inputs**:
  - Model configuration (`--model`, `--pretrained`).
  - Dataset selection (`--dataset` in `{imagenet, CIFAR10, CIFAR100}`) and root path.
  - Batch size/device/worker counts.
- **Key operations**:
  1. Build the CLIP model and preprocessing transforms via `create_model_and_transforms`.
  2. Load the requested dataset (default is `SubsampledImageNet` for efficiency).
  3. Iterate over the loader, encoding each batch with `model.encode_image(..., normalize=True)`.
  4. Multiply representations by a constant coefficient (default 100.0) so downstream files have numerically stable values.
- **Outputs**:
  - `{dataset}_{split}_representations_{model}_{pretrained}.npy`
  - `{dataset}_{split}_classes.npy`
- **Dependencies/assumptions**:
  - Uses the CLIP visual tower only (`model.visual` references), so it expects image tensors shaped after the preprocess pipeline.
  - Relies on deterministic dataset sampling (subsampled dataset strides) to keep file sizes small.

### 2. `compute_classifier_projection.py`
- **Goal**: Create the zero-shot classifier matrix used to evaluate representations and downstream ablations.
- **Inputs**:
  - Same `--model`, `--pretrained`, `--device`; dataset name is currently fixed to ImageNet classnames.
- **Key operations**:
  1. Load CLIP and its tokenizer.
  2. Enumerate ImageNet classes and the template set (`OPENAI_IMAGENET_TEMPLATES`).
  3. Encode every templated prompt, normalize, and average per class (autocast enabled).
  4. Stack class embeddings into the classifier weight matrix.
- **Outputs**:
  - `{dataset}_classifier_{model}_{pretrained}.npy` storing a `[d, num_classes]` float array.
- **Dependencies/assumptions**:
  - Text encoder and tokenizer must match the visual tower’s joint space.
  - Uses `torch.cuda.amp.autocast`; expects CUDA availability.

### 3. `compute_second_order_neuron_prs.py`
- **Goal**: Capture every neuron's second-order effect by tracing its activation through subsequent attention heads and projecting into the output embedding (the “projected residual stream”).
- **Inputs**:
  - CLIP model spec and target MLP layer index (`--mlp_layer`).
  - Dataset path/split plus loader parameters.
- **Key operations**:
  1. Initialize `SecondOrderPrsNeurons` hook via `hook_prs_logger(model, mlp_layer, device)`.
  2. For each batch:
     - Reset hook buffers (`prs.reinit()`).
     - Run `model.encode_image(..., attn_method="head", normalize=False)` to retain unnormalized residuals.
     - Call `prs.finalize()` to transfer cached numpy arrays into torch tensors on-device.
     - Loop over every later layer and attention head, calling `apply_attention_head_and_project_layer_neurons`.
     - Sum the resulting projections across heads to obtain `[batch, neurons, output_dim]`.
  3. Concatenate batch results and write a merged `.npy`.
- **Outputs**:
  - `{dataset}_{split}_mlps_{model}_{pretrained}_{mlp_layer}_merged.npy` with shape `[num_samples, neurons_per_layer, d_out]`.
- **Hook mechanics**:
  - `SecondOrderPrsNeurons` registers hooks on:
    - `visual.transformer.resblocks.*.mlp.gelu.post` (captures neuron activations).
    - `visual.transformer.resblocks.*.ln_1.{mean,sqrt_var}` (for pre-attention normalization).
    - `visual.transformer.resblocks.*.attn.attention.post_softmax` (attention maps).
    - `visual.ln_post.{mean,sqrt_var}` (final normalization).
  - During projection, it reconstructs each neuron's contribution by:
    - Applying the layer’s `c_proj` weights on a per-neuron basis.
    - Normalizing with recorded LN statistics.
    - Passing through attention (with stored softmax maps, V weights/bias, and out_proj weights).
    - Projecting to CLIP’s final embedding and normalizing by the original representation norm.
- **Dependencies/assumptions**:
  - Hook names depend on the CLIP visual module naming scheme.
  - Assumes uniform `mlp_width` across layers and consistent head dimensions.

### 4. `compute_pcas.py`
- **Goal**: For each neuron, fit a rank-1 approximation to its second-order effect tensor.
- **Inputs**:
  - The merged neuron tensor from the previous stage.
  - `--top_k_pca` specifying how many high-norm samples per neuron to use.
- **Key operations**:
  1. Load `{dataset}_train_mlps_*_merged.npy` in memory-mapped mode.
  2. Compute neuron-wise means (used later during ablations).
  3. For every neuron:
     - Subtract the mean.
     - Select the top-`k` examples by L2 norm.
     - Run SVD on this subset and take the first right singular vector (flipping sign so the majority of samples have positive projection).
     - Record the sorted norms for ablation thresholds.
 4. Save PCA vectors and norm thresholds.
- **Outputs**:
  - `{dataset}_train_mlps_*_{top_k}_pca.npy`  → shape `[neurons, d_out]`.
  - `{dataset}_train_mlps_*_{top_k}_norm.npy` → shape `[neurons, top_k]`.
- **Dependencies/assumptions**:
  - Operates purely on numpy arrays; assumes `neurons_mean` is two-dimensional `[neurons, d_out]`.
  - Expects the input file ordering to match the order of neurons captured by the hook (layer-specific).

### 5. `compute_sparse_decomposition.py`
- **Goal**: Describe each neuron's PCA vector with a sparse combination of text directions and optionally validate reconstruction quality.
- **Inputs**:
  - Classifier weights, text description embeddings, PCA results, neuron activations.
  - CLI args: `--components` (#non-zero entries), `--transform_alpha` (L1 penalty), `--text_descriptions`.
- **Key operations**:
  1. Load classifier matrix, text description embeddings, neuron tensors, and PCA vectors (memory-mapped to keep RAM manageable).
  2. Normalize text embeddings (mean-center, scale by coefficient, L2-normalize columns).
  3. Instantiate `Decompose` (wrapper over `sklearn.decomposition.SparseCoder`).
  4. Transform each neuron's PCA vector into sparse text coefficients; store both dense and zeroed-out versions.
  5. Serialize:
     - JSON mapping from neuron index to top `(token_index, coefficient, raw text)` tuples.
     - CSC sparse matrix `.npz`.
  6. If `--evaluate`:
     - Reconstruct neuron vectors (`sparse_coeffs @ text_embeddings`).
     - Run the validation dataset through CLIP with hooks to re-compute second-order effects.
     - Replace neuron contributions with reconstructions or ablated versions and recompute classifier logits, yielding accuracy metrics.
- **Outputs**:
  - Sparse decomposition JSON (`*_decomposition_omp_{alpha}_{components}.json`).
  - Sparse matrix `.npz`.
  - (Optionally) evaluation statistics printed to stdout.
- **Dependencies/assumptions**:
  - Text vocab file (e.g., `text_descriptions/30k.txt`) must align with the `.npy` embedding file.
  - Evaluation path reuses the `SecondOrderPrsNeurons` hook; expects image inputs even though decomposition is textual.

### 6. `compute_ablations.py`
- **Goal**: Quantify the functional importance of second-order effects by selectively ablating or approximating neuron contributions.
- **Inputs**:
  - Classifier weights, neuron tensors, PCA vectors, norm thresholds, dataset path.
  - CLI args controlling dataset split (`imagenet` vs. `imagenet_r`), layer index, etc.
- **Key operations**:
  1. Load the classifier and neuron PCA/norm files.
  2. Rebuild the model + hook as usual.
  3. For each validation batch:
     - Capture original (non-normalized) representations and log their classifier predictions.
     - Recompute neuron contributions using the hook.
     - Form multiple edited representations:
       * **Mean ablation**: subtract current neuron contributions and add back neuron-wise means.
       * **Without significant neurons**: add back only the small-norm neurons (below threshold).
       * **Without insignificant neurons**: add back the large-norm neurons.
       * **Projected to PCA**: replace contributions with their PCA reconstructions.
     - Multiply each edited representation by the classifier to obtain logits.
  4. After iterating, compute accuracy for each scenario with `utils.misc.accuracy`.
- **Outputs**:
  - Printed accuracies for baseline, without neurons, without significant/insignificant subsets, and PCA reconstruction.
- **Dependencies/assumptions**:
  - Reuses the same dataset subsampling as evaluation in other scripts.
  - Assumes the PCA + norm files correspond exactly to the neurons in the chosen `mlp_layer`.

## Hooking and Infrastructure Notes

### HookManager (`utils/hook.py`)
- Provides named hook registration independent of PyTorch’s raw handle objects.
- Supports glob-like patterns via `*` and numeric indices (matching `visual.transformer.resblocks.*`).
- Allows “forking” to reuse subsets of hooks for iterative structures.
- Tracks whether a registered hook was actually called; `finalize()` raises if any went unused—useful for catching naming mismatches.

### `SecondOrderPrsNeurons` (Core Phase 0 Artifact)
- Captures per-neuron activations, local LN stats, attention maps, and final LN stats as numpy arrays to reduce GPU memory during the forward pass.
- Replays the entire causal chain:
  1. `apply_mlp_post` multiplies per-neuron activations by the block’s `c_proj`.
  2. `normalize_before_attention_per_neuron` emulates the layernorm preceding attention, using cached means/stds and LN weights/bias.
  3. `apply_attention_matrix_per_neuron` feeds the normalized per-neuron contributions through the selected attention head (V projection → attention mixing → output projection + bias normalization).
  4. `project_attentions_to_output_per_neuron` applies the final layernorm and CLIP projector to obtain logits in the joint embedding.
  5. Results are scaled by the original representation’s norm to keep magnitudes comparable to normalized representations.
- All operations assume the CLIP ViT residual stream structure: alternating attention and MLP blocks with shared width, and a final `ln_post` + projection.

### Other Hook Utilities
- `prs_first_order_hook.py`: collects standard residual contributions (attentions + MLPs) for first-order comparisons.
- `prs_indirect_effect.py` and `prs_neuron_record.py`: provide targeted instrumentation for ablations (mean replacement) and activation recording. Helpful references for designing LLM-side toggles.

## Datasets and Preprocessing

- `SubsampledImageNet` / `SubsampledValImageNet` return every 250th (train) or 10th (val) sample to reduce runtime yet maintain class coverage. Their lengths scale down accordingly.
- `SubsampledValImageNetR` wraps ImageNet-R with the same stride logic and remaps class folders to the canonical ImageNet WNID list.
- All dataset classes rely on the `preprocess` transform returned with the CLIP model; this ensures consistent resizing, normalization, and center-cropping as expected by the OpenAI checkpoints.

## Data Flow Summary

```
images ──┐
         ├─ compute_representations.py ─▶ representations.npy + classes.npy
labels ──┘

ImageNet classnames/templates ─▶ compute_classifier_projection.py ─▶ classifier.npy

representations + hook instrumentation ─▶ compute_second_order_neuron_prs.py
    └─ outputs per-neuron tensors (mlps_..._merged.npy)

mlps_..._merged.npy ─▶ compute_pcas.py ─▶ PCA bases + norm thresholds

text_descriptions/???.txt ─▶ compute_text_set_projection.py ─▶ text embeddings (.npy)

{classifier, text embeddings, neuron tensors, PCA} ─▶ compute_sparse_decomposition.py
    └─ sparse codes + optional evaluation metrics

{classifier, neuron tensors, PCA, norm thresholds} ─▶ compute_ablations.py
    └─ printed correctness numbers for multiple ablations
```

Each stage reads from the previous one via deterministic filenames, so the entire pipeline can be resumed or partially re-run by reusing those artifacts.

## Assumptions to Preserve for the LLM Port

1. **Single-pass capture**: Hooks gather everything needed during the forward pass; heavy algebra happens post-hoc. This avoids gradient computations and keeps the pipeline inference-only.
2. **Normalization bookkeeping**: Every projection carefully accounts for layernorm means/stds and bias rescaling across attention heads. The LLM port must re-derive analogous formulas for textual transformers.
3. **File formats**: Downstream scripts expect `.npy` arrays with specific naming conventions and shapes. Aligning to those avoids rewriting PCA/decomposition/ablation code.
4. **Sampling strategies**: Subsampled datasets provide tractable runtimes; for LLMs we should emulate this via dataset curation (≈5k train / 500 val) to keep artifact sizes manageable.
5. **Stateless CLIs**: Each script is an independent CLI stage. Reproducing this structure for LLM experiments will simplify debugging and make it easier to swap CLIP vs. LLM code paths.

With these details documented, Phase 0 is complete. We can now proceed to Phase 1 with confidence about which components need direct analogs in the LLM setting.
