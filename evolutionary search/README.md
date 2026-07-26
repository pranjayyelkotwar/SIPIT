# Evolutionary search with LlamaScope

This directory packages the hidden-state search from `sae_exp_1.2` behind one
entrypoint. It deliberately uses the LlamaScope SAE through `sae-lens`, in the
same way as `tests/test_llamascope_sae.py`; it does **not** load the custom SAE
checkpoint used by the old experiment.

## What it does

For one hidden-state vector, the search:

1. loads the LlamaScope SAE matching `--layer`;
2. encodes the vector with `sae.encode`;
3. takes the largest active SAE features;
4. samples sparse changes to those feature values;
5. reconstructs every candidate with `sae.decode`;
6. evaluates the candidates with an existing grounding function;
7. keeps the best candidate and returns its reconstructed hidden state.

Lower grounding scores are treated as better. By default, a candidate is
accepted only when it improves on the best state. Pass
`--accept-non-improving` to reproduce the old experiment's greedy walk, which
could move to a worse state while patience counted down.

## Exact execution commands

Run every command in this section from the SIPIT base directory:

```bash
cd /Users/pranjayyelkotwar/Desktop/Dystopian_Bench/SIPIT
```

Install the project dependencies, including `torch`, `transformers`,
`huggingface_hub`, and `sae-lens`:

```bash
python3 -m pip install -e .
```

The artifact-building commands expect the captured activation dataset at these
exact locations:

```text
activation_outs/metadata_rank0.jsonl
activation_outs/layer_22/activations_l22_idx0.pt
activation_outs/layer_22/activations_l22_idx1.pt
...
```

That dataset is not currently present in the SIPIT checkout. Create it with
`activation_capture/capture_activations.py`; complete commands are in
`activation_capture/README.md`. Do not use the artifacts under `sae_exp_1.2`:
they belong to its custom SAE, not LlamaScope.

`--input` may be:

- a tensor shaped `(d_model,)`, `(seq_len, d_model)`, or
  `(1, seq_len, d_model)`; or
- a dictionary containing such a tensor under `hidden_states`. Change the key
  with `--input-key`.

For a bundle, every field is preserved and only the selected hidden state is
replaced. Search scores, feature vectors, layer, token position, release, and
SAE ID are added under `evolutionary_search`.

The scorer must match the selected LlamaScope SAE. Use exactly one of:

```text
--perplexity-weights FILE
--avg-latents-dir DIRECTORY
```

Old `sae_exp_1.2` weights were trained against a different SAE and will usually
have a different latent width. Even if dimensions happen to agree, mixing
weights from a different SAE is not meaningful. The entrypoint checks widths
and fails with an explicit error instead of silently producing invalid scores.

LlamaScope defaults to release `llama_scope_lxr_32x` and width `32x`. The layer
is required and selects `l{layer}r_{width}`, so layer 22 loads `l22r_32x`.

## Building the required artifacts

All commands below run from the SIPIT root. Always use the same `--layer`,
`--release`, and `--width` during artifact generation and search.

### Average-latent grounding

Build LlamaScope 32x average latents for layer 22:

```bash
python "evolutionary search/build_artifacts.py" \
  --activation-dir activation_outs \
  --metadata-file activation_outs/metadata_rank0.jsonl \
  --layer 22 \
  --device cuda \
  --release llama_scope_lxr_32x \
  --width 32x \
  --dtype float32 \
  averages \
  --output-dir "evolutionary search/artifacts/llamascope_l22_32x/avg_latents" \
  --chunk-size 4096
```

This encodes every token with LlamaScope and writes files such as
`avg_latents_arc_easy.pt`, `avg_latents_hle.pt`, and
`avg_latents_mmlu.pt`. SAE grounding uses ARC-Easy minus HLE, matching the old
experiment.

### Perplexity regression

If the Llama weights are not already available locally, authenticate with
Hugging Face first:

```bash
huggingface-cli login
```

Compute layer-22 perplexity targets from every metadata prompt:

```bash
python "evolutionary search/compute_perplexity_targets.py" \
  --metadata-file activation_outs/metadata_rank0.jsonl \
  --output-jsonl "evolutionary search/artifacts/llamascope_l22_32x/grounding_perplexity.jsonl" \
  --model meta-llama/Meta-Llama-3.1-8B \
  --layer 22 \
  --max-length 192 \
  --device cuda \
  --dtype bfloat16
```

This loads the base language model and may require Hugging Face access to the
gated Llama weights. A local model directory can be passed to `--model`.

Encode the final-token hidden states with LlamaScope and train the regression:

```bash
python "evolutionary search/build_artifacts.py" \
  --activation-dir activation_outs \
  --metadata-file activation_outs/metadata_rank0.jsonl \
  --layer 22 \
  --device cuda \
  --release llama_scope_lxr_32x \
  --width 32x \
  --dtype float32 \
  regression \
  --targets-jsonl "evolutionary search/artifacts/llamascope_l22_32x/grounding_perplexity.jsonl" \
  --target-key perplexity \
  --token-pos -1 \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.01 \
  --weight-decay 0.0 \
  --normalize-targets \
  --use-bias \
  --seed 0 \
  --output-weights "evolutionary search/artifacts/llamascope_l22_32x/perplexity_regression.pt"
```

The resulting file is accepted by `run_search.py --perplexity-weights`.

### Run with average-latent grounding

This uses the existing layer-22 input bundle `sipit_id6573_l22.pt`, perturbs its
final token, and writes a new bundle:

```bash
python "evolutionary search/run_search.py" \
  --input sipit_id6573_l22.pt \
  --output "evolutionary search/results/sipit_id6573_l22_avg_grounding.pt" \
  --input-key hidden_states \
  --layer 22 \
  --device cuda \
  --release llama_scope_lxr_32x \
  --width 32x \
  --dtype float32 \
  --avg-latents-dir "evolutionary search/artifacts/llamascope_l22_32x/avg_latents" \
  --token-pos -1 \
  --num-candidates 32 \
  --min-active 2 \
  --max-active 6 \
  --active-topk 16 \
  --beta 0.5 \
  --max-iters 50 \
  --min-improvement 0.0001 \
  --patience 5 \
  --seed 0
```

Expected output:

```text
evolutionary search/results/sipit_id6573_l22_avg_grounding.pt
```

### Run with perplexity-regression grounding

```bash
python "evolutionary search/run_search.py" \
  --input sipit_id6573_l22.pt \
  --output "evolutionary search/results/sipit_id6573_l22_regression.pt" \
  --input-key hidden_states \
  --layer 22 \
  --device cuda \
  --release llama_scope_lxr_32x \
  --width 32x \
  --dtype float32 \
  --perplexity-weights "evolutionary search/artifacts/llamascope_l22_32x/perplexity_regression.pt" \
  --token-pos -1 \
  --num-candidates 32 \
  --min-active 2 \
  --max-active 6 \
  --active-topk 16 \
  --beta 0.5 \
  --max-iters 50 \
  --min-improvement 0.0001 \
  --patience 5 \
  --seed 0
```

Expected output:

```text
evolutionary search/results/sipit_id6573_l22_regression.pt
```

For CPU-only execution, replace `--device cuda` with `--device cpu`. For the
perplexity command, also replace `--dtype bfloat16` with `--dtype float32`.

### Smoke-test commands

Limit target computation and regression training to ten records:

```bash
python "evolutionary search/compute_perplexity_targets.py" \
  --metadata-file activation_outs/metadata_rank0.jsonl \
  --output-jsonl /tmp/grounding_perplexity_smoke.jsonl \
  --model meta-llama/Meta-Llama-3.1-8B \
  --layer 22 \
  --max-length 192 \
  --max-records 10 \
  --device cuda \
  --dtype bfloat16
```

```bash
python "evolutionary search/build_artifacts.py" \
  --activation-dir activation_outs \
  --metadata-file activation_outs/metadata_rank0.jsonl \
  --layer 22 \
  --device cuda \
  --release llama_scope_lxr_32x \
  --width 32x \
  --dtype float32 \
  regression \
  --targets-jsonl /tmp/grounding_perplexity_smoke.jsonl \
  --target-key perplexity \
  --token-pos -1 \
  --max-samples 10 \
  --epochs 2 \
  --batch-size 2 \
  --lr 0.01 \
  --seed 0 \
  --output-weights /tmp/perplexity_regression_smoke.pt
```

## Python API

Because this folder name contains a space, add it to `sys.path` before importing:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("evolutionary search").resolve()))

from grounding import load_perplexity_regression_weights
from llamascope import load_llamascope_sae
from search import SearchConfig, perturb_hidden_state

loaded = load_llamascope_sae(layer=22, device="cuda")
scorer = load_perplexity_regression_weights(
    Path("weights.pt"), device=loaded.sae.device
)
result = perturb_hidden_state(
    hidden_state,
    sae=loaded.sae,
    scorer=scorer,
    config=SearchConfig(num_candidates=32, max_iters=20, beta=0.1),
)
reconstructed = result.reconstructed_hidden_state
```

A custom scorer is also supported. It receives `(features, reconstructed)` and
must return one scalar tensor; the search minimizes it.

## Grounding functions

The full set is now split according to what it actually requires:

- `grounding.py`: ARC-Easy/HLE SAE grounding, perplexity regression, and the
  Fisher stability delta formula.
- `model_grounding.py`: Fisher-diagonal estimation and pseudo-curvature.

Fisher estimation and pseudo-curvature are not enabled in the lone-hidden-state
CLI because they cannot be calculated from a vector alone. They require a
loaded Llama model, prompt tokens, the complete activation sequence, override
layer, and token position. `compute_fisher_diag` supports the original custom
Llama override interface. `compute_pseudo_curv` accepts a `logits_fn` adapter so
it can be connected to another Llama runtime while still using LlamaScope
feature perturbations and decoding.

The output is an SAE reconstruction, so it may differ from the input even when
the search performs zero steps. This is the expected LlamaScope reconstruction
error.
