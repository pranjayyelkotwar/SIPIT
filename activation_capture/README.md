# LlamaScope-aligned activation capture

This package creates the `activation_outs` directory consumed by
`evolutionary search/build_artifacts.py`. It ports the QA and OpenWebText
dataset behavior from `sae_exp_1.2`, but uses Hugging Face Llama and captures
the output of decoder block `N` for `--layers N`. This is residual-post
activation data for the corresponding LlamaScope `lNr_*` SAE.

The supported sources are:

- `arc_easy`: `allenai/ai2_arc`, ARC-Easy train split;
- `mmlu`: `cais/mmlu`, all subjects, test split;
- `hle`: `cais/hle`, test split, text-only examples;
- `openwebtext`: `paulpauls/openwebtext-sentences`, train split.

QA prompts preserve the old format:

```text
Answer the following question.

Question: ...
Choices:
A. ...
B. ...
```

## Output contract

For layer 22, capture creates:

```text
activation_outs/
├── metadata_rank0.jsonl
└── layer_22/
    ├── activations_l22_idx0.pt
    ├── activations_l22_idx1.pt
    └── ...
```

Each activation tensor is `(sequence_length, hidden_size)`, with padding
removed. Every metadata line includes the normalized dataset record plus:

```text
activation_path
capture_dataset_idx
capture_layer
prompt_text
token_count
source_dataset
```

The output filenames and metadata fields match the consumers in
`evolutionary search`.

## Exact commands

Run from the SIPIT root:

```bash
cd /Users/pranjayyelkotwar/Desktop/Dystopian_Bench/SIPIT
```

Install dependencies:

```bash
python3 -m pip install -e .
```

Authenticate before downloading gated Meta Llama weights:

```bash
huggingface-cli login
```

### Smoke test: ten samples from each QA dataset

```bash
python activation_capture/capture_activations.py \
  --model meta-llama/Meta-Llama-3.1-8B \
  --output-dir activation_outs_smoke \
  --layers 22 \
  --dataset-source qa \
  --qa-datasets arc_easy,mmlu,hle \
  --qa-num-samples arc_easy:10,mmlu:10,hle:10 \
  --max-token-length 192 \
  --batch-size 2 \
  --num-workers 2 \
  --add-bos-token \
  --include-choices \
  --dtype bfloat16
```

### Full QA capture for layer 22

This is the exact capture needed to build the default ARC-Easy/HLE grounding
artifacts:

```bash
python activation_capture/capture_activations.py \
  --model meta-llama/Meta-Llama-3.1-8B \
  --output-dir activation_outs \
  --layers 22 \
  --dataset-source qa \
  --qa-datasets arc_easy,mmlu,hle \
  --max-token-length 192 \
  --batch-size 8 \
  --num-workers 4 \
  --add-bos-token \
  --include-choices \
  --dtype bfloat16
```

### Limited QA capture

```bash
python activation_capture/capture_activations.py \
  --model meta-llama/Meta-Llama-3.1-8B \
  --output-dir activation_outs \
  --layers 22 \
  --dataset-source qa \
  --qa-datasets arc_easy,mmlu,hle \
  --qa-num-samples arc_easy:1000,mmlu:2200,hle:1000 \
  --max-token-length 192 \
  --batch-size 8 \
  --num-workers 4 \
  --add-bos-token \
  --include-choices \
  --dtype bfloat16
```

### Capture layers 16 and 22

```bash
python activation_capture/capture_activations.py \
  --model meta-llama/Meta-Llama-3.1-8B \
  --output-dir activation_outs \
  --layers 16,22 \
  --dataset-source qa \
  --qa-datasets arc_easy,mmlu,hle \
  --max-token-length 192 \
  --batch-size 8 \
  --num-workers 4 \
  --add-bos-token \
  --include-choices \
  --dtype bfloat16
```

### OpenWebText capture

```bash
python activation_capture/capture_activations.py \
  --model meta-llama/Meta-Llama-3.1-8B \
  --output-dir activation_outs_openwebtext \
  --layers 22 \
  --dataset-source openwebtext \
  --num-samples 10000 \
  --max-token-length 192 \
  --batch-size 8 \
  --num-workers 4 \
  --shuffle \
  --seed 42 \
  --add-bos-token \
  --dtype bfloat16
```

### Multi-GPU capture

The same CLI supports `torchrun`. Each rank receives a non-overlapping
deterministic dataset shard. Rank zero merges all metadata parts into
`metadata_rank0.jsonl`.

```bash
torchrun --standalone --nproc-per-node=4 \
  activation_capture/capture_activations.py \
  --model meta-llama/Meta-Llama-3.1-8B \
  --output-dir activation_outs \
  --layers 22 \
  --dataset-source qa \
  --qa-datasets arc_easy,mmlu,hle \
  --max-token-length 192 \
  --batch-size 8 \
  --num-workers 4 \
  --add-bos-token \
  --include-choices \
  --dtype bfloat16
```

`--batch-size` is per GPU. Four processes with batch size eight process up to
32 prompts concurrently.

### Local model and CPU

Replace `--model` with a local Hugging Face model directory:

```text
--model /absolute/path/to/Meta-Llama-3.1-8B
```

CPU capture is selected automatically when CUDA is unavailable. In that case,
use:

```text
--dtype float32 --batch-size 1
```

## Continue into artifact generation

After full QA capture completes, run:

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

See `evolutionary search/README.md` for regression artifacts and search.

## Operational notes

- Existing files with the same layer/index names are overwritten. Use a new
  output directory when changing dataset order or configuration.
- Capturing multiple layers writes one tensor file per layer and one metadata
  record per layer.
- The model is replicated once per `torchrun` process; ensure each GPU has
  enough memory for one Llama-3.1-8B copy.
- The asynchronous writer moves completed activations to CPU and writes them
  while the next batches execute.

