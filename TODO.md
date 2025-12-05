# TODO

## Guiding Objectives
- [ ] Keep the first iteration as close as possible to the existing CLIP pipeline (representations → classifier → second-order logs → PCA → sparse text decomposition) before making LLM-specific architectural changes, so we can validate the port step-by-step against a known workflow (`CLIP-2OE/README.md:19`).
- [ ] Apply the second-order methodology to a text-only LLM solving math/logic tasks, focusing on the "correctness direction" projection described in the project summary to get an initial measurement with minimal code churn (`SUMMARY.md:66`).
- [ ] After collecting the first wave of measurements (sparsity, layer concentration, rank-1 approximation accuracy), decide what to tune or redesign based on the observed gaps relative to the CLIP targets (`SUMMARY.md:148`).

## Phase 0 – Ground Truth on the CLIP Implementation
- [x] Produce a short internal doc explaining each existing stage (`compute_representations.py`, `compute_classifier_projection.py`, `compute_second_order_neuron_prs.py`, `compute_pcas.py`, `compute_sparse_decomposition.py`, `compute_ablations.py`) and the artifacts they emit, so we know exactly what to mirror when touching the LLM path (`CLIP-2OE/README.md:19`).
- [x] Map every dependency (datasets, hooks, utils, model wrappers) by reading through `CLIP-2OE/utils`, especially `hook.py` and `factory.py`, to understand how hooks are registered and models are created before we attempt to attach into a language model (`CLIP-2OE/utils/hook.py:1`).
- [x] Trace the call flow of `compute_second_order_neuron_prs.py` step-by-step (model construction, dataset loader, PRS hook, aggregation) and annotate which parts are CLIP-specific (image transform, visual tower references) vs. generic residual-stream math we can re-use (`CLIP-2OE/compute_second_order_neuron_prs.py:49`).
- [x] Inspect `prs_second_order_neurons_hook.py` in detail and document every assumption baked into the hook (availability of `visual.transformer.resblocks`, layer-norm shapes, attention tensor ordering) to know what has to be rewritten for GPT-style blocks (`CLIP-2OE/prs_second_order_neurons_hook.py:13`).
- [x] Confirm what the sparse decomposition code expects (precomputed PCA bases, text lexicon embeddings, JSON exports) so that we know which intermediate files the LLM pipeline must emit for compatibility (`CLIP-2OE/compute_sparse_decomposition.py:80`).

## Phase 1 – Environment, Data, and Evaluation Foundations
- [x] Decide whether to reuse the existing Conda env vs. starting a new Poetry/venv; verify PyTorch/CUDA, huggingface transformers, numpy/scipy, sklearn are all available for both CLIP and LLM workflows (`CLIP-2OE/README.md:11`).
- [x] Build a dataset loader that can emit (prompt, target answer, correctness label) tuples for GSM8K + synthetic arithmetic/logic sets described in the plan, sharding into train (≈5k) and validation (≈500) splits (`SUMMARY.md:131`).
- [x] Implement an evaluation harness that runs the target LLM (start with a manageable baseline such as GPT-2 small or Pythia-410M) on the curated math set, producing per-problem transcripts, logit histories, and binary correctness flags needed later for the correctness direction computation (`SUMMARY.md:66`).
- [x] Store dataset outputs and model responses in a format analogous to the CLIP representations (e.g., `.npy` for activations plus `.jsonl` metadata) so downstream scripts can stay close to the existing structure (`CLIP-2OE/compute_representations.py:71`).
- [x] Define the "correctness direction" vector by averaging the final residual stream / unembedding activations over correct vs. incorrect examples, and cache it in the same way that CLIP caches classifier weights for reuse in later stages (`SUMMARY.md:66`).
- [x] Automate lightweight layerwise ablations (temporarily zero or mean-replace feedforward outputs per layer) on the validation split to select the top ~6 critical layers as suggested in the summary before committing to expensive second-order calculations (`SUMMARY.md:85`).
- [x] Rank neurons within the chosen layers using the tri-criteria heuristic (mean |activation|, variance, direct effect magnitude) to preselect the top ~100 neurons per layer and persist these indices for the hook to focus on later (`SUMMARY.md:90`).

## Phase 2 – Minimal-Change LLM Port of the CLIP Pipeline
- [x] Implement a language-model counterpart to `compute_representations.py` that runs a forward pass with hooks capturing token-level residual streams and saves normalized representations, mirroring the CLI + dataloader interface (model name, dataset path, output dir) from the CLIP script (`CLIP-2OE/compute_representations.py:17`).
- [x] Create a `compute_correctness_projection.py` analog to `compute_classifier_projection.py`, but instead of zero-shot classifiers produce (a) the correctness direction vector, and (b) optional task-specific logit probes for sanity-checks (`CLIP-2OE/compute_classifier_projection.py:19`).
- [x] Port the second-order hook logic to LLMs: start from `SecondOrderPrsNeurons` but replace CLIP-specific modules (visual LN/attention) with references to the GPT-style block (`transformer.h`, `attn.c_proj`, etc.), ensuring we respect causal masking and token positions as described in the adapted formula (`CLIP-2OE/prs_second_order_neurons_hook.py:64`).
- [x] Ensure the hook manager (or PyTorch forward hooks) can register on every transformer block's LN, attention softmax, MLP activations, and unembedding projection; if needed, extend `utils/hook.py` to handle the text tower naming scheme (`CLIP-2OE/utils/hook.py:1`).
- [x] Re-create a `compute_second_order_neuron_prs.py` equivalent that:
  - Receives CLI args for `--model`, `--mlp_layer`, `--dataset`, `--output_dir`, `--device` identical to the CLIP script, to minimize new plumbing.
  - Runs the language model across batches of math problems (prompted in a consistent format) and uses the new hook to aggregate per-neuron contributions through every subsequent attention head, stopping at the unembedding matrix (`CLIP-2OE/compute_second_order_neuron_prs.py:67`).
  - Projects the effects onto the stored correctness direction (reuse the same `.npy` emission code path) to make downstream PCA/OMP unmodified.
- [x] Validate the new hook numerically by (a) checking that recombining all neuron effects plus the mean residual recovers the original logits within machine tolerance, and (b) comparing first-order vs. second-order contributions on a few hand-picked neurons to ensure signs/magnitudes look reasonable.
- [x] Update the PCA computation script to read the language-model neuron tensors and compute per-neuron rank-1 bases exactly as in CLIP (top-k norms, sign disambiguation), keeping filenames consistent to re-use the sparse decomposition code unchanged (`CLIP-2OE/compute_pcas.py:17`).
- [x] Prepare a math/logic description lexicon (start with a curated list + GSM8K rationales) and embed it through the same text encoder the LLM uses, mirroring `compute_text_set_projection.py` so that `compute_sparse_decomposition.py` can run unmodified aside from pointing at the new vocabulary file (`CLIP-2OE/compute_text_set_projection.py:21`).
- [x] Run `compute_sparse_decomposition.py` with the math lexicon and verify reconstruction quality using the `--evaluate` flag once the PCA outputs exist, ensuring reconstruction accuracy on the validation split is within 1% of the unmodified model as in the CLIP ablation benchmarks (`CLIP-2OE/compute_sparse_decomposition.py:162`).
- [x] Adapt `compute_ablations.py` to the math dataset so we can test (1) full mean ablation, (2) removing "significant" neurons (per PCA norm threshold), (3) removing "insignificant" neurons, and (4) substituting rank-1 reconstructions, mirroring the CLIP logic for a correctness metric (`CLIP-2OE/compute_ablations.py:126`).
- [ ] Generate required artifacts for Phase 2 runs:
  - [x] Run `phase1/evaluate_model.py` for each dataset split to cache prompts, representations, and initial correctness labels.
  - [x] Run `phase2/compute_representations.py` on the same splits to save token-level residual tensors.
  - [x] Run `phase2/compute_correctness_projection.py` (with `--label_source phase1` or `phase2`) to emit `*_correctness_direction.npy` and aligned labels.
  - [x] Run `phase2/compute_second_order_neuron_prs.py` per target layer/split to obtain `*_second_order_layer{L}.npy` tensors and correctness scores.
  - [x] Run `phase2/compute_pcas.py` on the training tensors to produce `*_pca.npy` / `*_norm.npy`.
  - [x] Run `phase2/compute_text_set_projection.py` once per model to embed `phase2/math_lexicon.txt`.
  - [x] Run `phase2/compute_sparse_decomposition.py --evaluate` with the math lexicon to produce sparse codes and reconstruction metrics.
  - [x] Run `phase2/compute_ablations.py` on the validation split to gather mean/ significant/insignificant/rank-1 correctness deltas.
- [ ] Repeat the artifact generation steps above for additional LLM baselines beyond GPT-2 (currently only GPT-2 has been processed).

## Phase 3 – Run the Minimal-Change Experiment and Collect Baselines
- [x] Execute the full pipeline end-to-end on the chosen baseline LLM: representations → correctness direction → second-order tensors (for selected layers/neurons) → PCA → sparse decomposition → ablation suite. *(GPT-2 baseline complete via `phase3/run_phase3_gpt2.sh` in tmux on GPU 2; see `phase3_gpt2.log` for the transcript.)*
- [ ] Record runtime, GPU memory, and intermediate tensor sizes to confirm the expected 90–150× speedups from neuron/layer sampling and single-pass hooks; if numbers deviate, log which optimization failed (`SUMMARY.md:114`).
- [ ] Produce a metrics report capturing sparsity (% tasks per neuron), layer concentration (% total second-order mass inside the selected layers), variance explained by PC1, and accuracy drops when swapping in rank-1 reconstructions vs. mean ablations (`SUMMARY.md:148`).
- [ ] Summarize qualitative findings: do OMP-derived math concepts correlate with activation triggers, and do identified neurons cleanly separate correct vs. incorrect trajectories? Save several case studies similar to the CLIP notebook visualizations but for text prompts.
- [ ] Decide whether the initial results meet the success criteria (e.g., <5% sparsity, >80% layer concentration, <1% accuracy drop) to inform how aggressive we need to be in the next iteration (`SUMMARY.md:187`).

## Phase 4 – Tailor the Approach Based on Observed Results
- [ ] If sparsity is too low/high, revisit neuron selection: tune the threshold for "active" neurons or incorporate gradient-based saliency into the scoring heuristic before re-running the second-order capture.
- [ ] If layer concentration is diffuse, expand the ablation search beyond the initial six layers or introduce per-token gating (e.g., focus on the final few generation steps) to narrow the attention path.
- [ ] When correctness-direction projections look noisy, experiment with alternative projections: e.g., logit lens directions for the correct answer tokens, or embeddings of reasoning quality labels, and compare how they change PCA/OMP outputs.
- [ ] For neurons whose sparse decompositions remain polysemantic, try swapping in a richer concept library (math textbooks, logic symbols) or increase the OMP sparsity `m` to verify whether the ambiguity is due to dictionary coverage (`SUMMARY.md:98`).
- [ ] Use the ablation tools to run small-scale editing experiments: amplify neuron effects correlated with correct answers and suppress those tied to systematic math errors; log whether this improves validation accuracy without hurting unrelated prompts (`SUMMARY.md:247`).
- [ ] Based on code health after the minimal port, plan refactors (e.g., generalize hook registration across model families, abstract dataset interfaces) only where necessary to keep iteration velocity high.

## Phase 5 – Extended Experiments and Deliverables
- [ ] Expand to a second model family (e.g., Pythia or Llama) once the baseline works, to test generalization claims from the summary (`SUMMARY.md:195`).
- [ ] Build tooling for the downstream applications sketched in the plan: automated error analysis dashboards, polysemanticity detectors, adversarial question generators, and neuron-level editing sweeps (`SUMMARY.md:167`).
- [ ] Package the full experimental log (configs, checkpoints, metrics) so future iterations can reproducibly compare against the baseline before and after tailored tweaks.
- [ ] Draft the write-up sections (methods, experiments, results, applications) in parallel with code work, using the CLIP paper structure as a template for communicating the LLM-specific findings once the tailored iteration stabilizes.
