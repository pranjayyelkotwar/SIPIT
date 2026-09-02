# SIPIT GPU Restart and Experiment Runbook

This document records the complete workflow for rebuilding a temporary GPU
instance, capturing Llama activations, running the SAE experiments, and copying
the results back to the local machine before deleting the instance.

The commands assume:

- the GPU workspace is `/workspace`;
- the repository should be `/workspace/SIPIT`;
- the Git branch containing the experiment code is `siddhant/dev`;
- the model is `meta-llama/Llama-3.1-8B`;
- the capture and LlamaScope SAE layer is 22;
- captures use `bfloat16`;
- the local repository is
  `/Users/siddhantmahajan/Desktop/SIPIT`.

Replace `<SSH_PORT>`, `<GPU_HOST>`, and `<PRIVATE_KEY>` with the values for the
new instance. SSH uses lowercase `-p`; SCP uses uppercase `-P`.

## 1. Connect to the GPU instance

From the local Mac:

```bash
ssh \
  -i <PRIVATE_KEY> \
  -o IdentitiesOnly=yes \
  -p <SSH_PORT> \
  root@<GPU_HOST> \
  -L 8080:localhost:8080
```

The port-forwarding argument is optional for these experiments. A minimal SSH
command is:

```bash
ssh -i <PRIVATE_KEY> -o IdentitiesOnly=yes -p <SSH_PORT> root@<GPU_HOST>
```

## 2. Verify the instance

Run on the GPU:

```bash
nvidia-smi
python --version
df -h /workspace
```

Python 3.10 or 3.11 is preferred for this repository.

## 3. Clone SIPIT and select the experiment branch

Run on the GPU:

```bash
cd /workspace

git clone \
  --branch siddhant/dev \
  --single-branch \
  https://github.com/pranjayyelkotwar/SIPIT.git

cd /workspace/SIPIT
git branch --show-current
git log -1 --oneline
```

The displayed branch should be:

```text
siddhant/dev
```

If the repository already exists on a persistent volume:

```bash
cd /workspace/SIPIT
git status --short
git pull --ff-only origin siddhant/dev
```

Do not run `git pull` over uncommitted remote-instance changes without first
checking `git status`.

## 4. Upload local-only files when necessary

The branch normally contains:

```text
experiments/compare_last_token_features.py
experiments/inspect_token_feature_activations.py
experiments/rank_questions_by_feature.py
experiments/analyze_top_questions_tokens.py
```

Verify on the GPU:

```bash
cd /workspace/SIPIT
ls -l experiments
```

If a script exists only on the local Mac, open a second local terminal and copy
the complete experiments directory:

```bash
scp \
  -i <PRIVATE_KEY> \
  -o IdentitiesOnly=yes \
  -P <SSH_PORT> \
  -r /Users/siddhantmahajan/Desktop/SIPIT/experiments \
  root@<GPU_HOST>:/workspace/SIPIT/
```

To upload only one file:

```bash
scp \
  -i <PRIVATE_KEY> \
  -o IdentitiesOnly=yes \
  -P <SSH_PORT> \
  /Users/siddhantmahajan/Desktop/SIPIT/experiments/analyze_top_questions_tokens.py \
  root@<GPU_HOST>:/workspace/SIPIT/experiments/
```

## 5. Create the Python environment and install dependencies

Run on the GPU:

```bash
cd /workspace/SIPIT

python -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Whenever reconnecting to the same instance, reactivate the environment:

```bash
cd /workspace/SIPIT
source .venv/bin/activate
```

Verify the installation and CUDA:

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -c "import datasets, transformers, sae_lens; print('experiment dependencies imported successfully')"
```

If `torch.cuda.is_available()` is false, stop and fix the instance/PyTorch setup
before starting a long capture.

## 6. Authenticate with Hugging Face

The Meta Llama model is gated. The Hugging Face account must already have
access to `meta-llama/Llama-3.1-8B`.

Run interactively on the GPU:

```bash
hf auth login
hf auth whoami
```

The current Hugging Face CLI documents `hf auth login` as the standard login
command:
<https://huggingface.co/docs/huggingface_hub/guides/cli>.

Do not commit a Hugging Face token to Git. If environment-variable login is
preferred, enter it without placing the literal token in this document:

```bash
read -s HF_TOKEN
export HF_TOKEN
hf auth login --token "$HF_TOKEN"
unset HF_TOKEN
```

## 7. Optional 10 + 10 smoke capture

Run this first on a new image to verify model access, dataset access, CUDA, and
disk writes without paying for the full capture:

```bash
cd /workspace/SIPIT
source .venv/bin/activate

python activation_capture/capture_activations.py \
  --model meta-llama/Llama-3.1-8B \
  --output-dir activation_outs_hle_arc_smoke \
  --layers 22 \
  --dataset-source qa \
  --qa-datasets hle,arc_easy \
  --qa-num-samples hle:10,arc_easy:10 \
  --max-token-length 192 \
  --batch-size 4 \
  --num-workers 2 \
  --add-bos-token \
  --include-choices \
  --dtype bfloat16
```

Expected capture structure:

```text
activation_outs_hle_arc_smoke/
├── layer_22/
│   └── activations_l22_idx*.pt
├── metadata.part_rank0.jsonl
└── metadata_rank0.jsonl
```

Verify counts:

```bash
find activation_outs_hle_arc_smoke/layer_22 -name '*.pt' | wc -l
wc -l activation_outs_hle_arc_smoke/metadata_rank0.jsonl
```

Both counts should be 20.

## 8. Optional smoke comparison

```bash
python experiments/compare_last_token_features.py \
  --activation-dir activation_outs_hle_arc_smoke \
  --metadata-file activation_outs_hle_arc_smoke/metadata_rank0.jsonl \
  --layer 22 \
  --device cuda \
  --dtype bfloat16 \
  --activation-threshold 0 \
  --min-prevalence 0.10 \
  --output-dir experiment_outputs/exp1_smoke
```

Expected files:

```text
experiment_outputs/exp1_smoke/
├── feature_sets.json
├── feature_statistics.pt
├── feature_stats.csv
└── summary.json
```

## 9. Production capture: 100 HLE + 100 ARC-Easy

Use a new output directory. Reusing an existing capture directory can overwrite
metadata or activation filenames.

```bash
cd /workspace/SIPIT
source .venv/bin/activate

python activation_capture/capture_activations.py \
  --model meta-llama/Llama-3.1-8B \
  --output-dir activation_outs_hle_arc_100 \
  --layers 22 \
  --dataset-source qa \
  --qa-datasets hle,arc_easy \
  --qa-num-samples hle:100,arc_easy:100 \
  --max-token-length 192 \
  --batch-size 4 \
  --num-workers 2 \
  --add-bos-token \
  --include-choices \
  --dtype bfloat16
```

Verify that 200 tensors and metadata records exist:

```bash
find activation_outs_hle_arc_100/layer_22 -name '*.pt' | wc -l
wc -l activation_outs_hle_arc_100/metadata_rank0.jsonl
du -sh activation_outs_hle_arc_100
```

If capture runs out of GPU memory, retry in a fresh output directory with:

```text
--batch-size 2
```

or, if needed:

```text
--batch-size 1
```

## 10. Experiment 1: compare final-token HLE and ARC SAE features

```bash
python experiments/compare_last_token_features.py \
  --activation-dir activation_outs_hle_arc_100 \
  --metadata-file activation_outs_hle_arc_100/metadata_rank0.jsonl \
  --layer 22 \
  --device cuda \
  --width 32x \
  --dtype bfloat16 \
  --activation-threshold 0 \
  --min-prevalence 0.10 \
  --min-mean 0 \
  --top-k 100 \
  --output-dir experiment_outputs/exp1_100
```

Expected outputs:

```text
experiment_outputs/exp1_100/
├── feature_sets.json
├── feature_statistics.pt
├── feature_stats.csv
└── summary.json
```

Check the processed counts:

```bash
python -m json.tool experiment_outputs/exp1_100/summary.json
```

The summary should report 100 HLE questions, 100 ARC questions, and zero missing
files.

## 11. Optional aggregate-statistics helper

`experiments/analysis.py` is currently a local helper. Upload it with the
experiments directory if it is needed, then run:

```bash
python experiments/analysis.py
```

It reads:

```text
experiment_outputs/exp1_100/feature_statistics.pt
```

and writes filtered HLE and ARC feature-statistics text files.

## 12. Rank the top HLE questions for the selected features

The default feature IDs are:

```text
2548, 11089, 14847, 36468, 36484,
57138, 66188, 69504, 106542, 121471
```

Run:

```bash
python experiments/rank_questions_by_feature.py \
  --activation-dir activation_outs_hle_arc_100 \
  --metadata-file activation_outs_hle_arc_100/metadata_rank0.jsonl \
  --layer 22 \
  --device cuda \
  --width 32x \
  --dtype bfloat16 \
  --top-k 10 \
  --minimum-activation 0 \
  --output-dir experiment_outputs/exp1_100/top_questions
```

Expected outputs:

```text
experiment_outputs/exp1_100/top_questions/
├── top_hle_questions_by_feature.csv
└── top_hle_questions_by_feature.json
```

To replace the default candidate list, repeat `--feature-id`:

```bash
python experiments/rank_questions_by_feature.py \
  --activation-dir activation_outs_hle_arc_100 \
  --metadata-file activation_outs_hle_arc_100/metadata_rank0.jsonl \
  --layer 22 \
  --device cuda \
  --dtype bfloat16 \
  --feature-id 2548 \
  --feature-id 69504 \
  --top-k 10 \
  --output-dir experiment_outputs/custom_top_questions
```

## 13. Experiment 2: analyze the top five questions token by token

The batch script loads the tokenizer and SAE once. It omits token positions
whose selected-feature activation is zero.

### Feature 2548

```bash
python experiments/analyze_top_questions_tokens.py \
  --ranked-json experiment_outputs/exp1_100/top_questions/top_hle_questions_by_feature.json \
  --feature-id 2548 \
  --top-k-questions 5 \
  --top-k-tokens 20 \
  --activation-dir activation_outs_hle_arc_100 \
  --device cuda \
  --width 32x \
  --dtype bfloat16 \
  --output-dir experiment_outputs/exp2/feature_2548_top5
```

### Feature 36484

```bash
python experiments/analyze_top_questions_tokens.py \
  --ranked-json experiment_outputs/exp1_100/top_questions/top_hle_questions_by_feature.json \
  --feature-id 36484 \
  --top-k-questions 5 \
  --top-k-tokens 20 \
  --activation-dir activation_outs_hle_arc_100 \
  --device cuda \
  --width 32x \
  --dtype bfloat16 \
  --output-dir experiment_outputs/exp2/feature_36484_top5
```

### Feature 69504

```bash
python experiments/analyze_top_questions_tokens.py \
  --ranked-json experiment_outputs/exp1_100/top_questions/top_hle_questions_by_feature.json \
  --feature-id 69504 \
  --top-k-questions 5 \
  --top-k-tokens 20 \
  --activation-dir activation_outs_hle_arc_100 \
  --device cuda \
  --width 32x \
  --dtype bfloat16 \
  --output-dir experiment_outputs/exp2/feature_69504_top5
```

Each command produces:

```text
experiment_outputs/exp2/feature_<ID>_top5/
├── token_feature_rankings.csv
└── token_feature_rankings.json
```

To analyze a different feature, change both `--feature-id` and the output
directory.

## 14. Experiment 2 for one manually selected question

Use this when inspecting one activation tensor with an exact saved prompt:

```bash
python experiments/inspect_token_feature_activations.py \
  --activation activation_outs_hle_arc_100/layer_22/activations_l22_idx0.pt \
  --prompt-file question_prompt.txt \
  --feature-id 2548 \
  --layer 22 \
  --model meta-llama/Meta-Llama-3.1-8B \
  --device cuda \
  --width 32x \
  --dtype bfloat16 \
  --top-k-tokens 20 \
  --output experiment_outputs/exp2/manual_question_feature_2548.csv
```

The contents of `question_prompt.txt` must exactly match the prompt used during
capture, including the template and choices. The BOS-token setting defaults to
enabled and must also match capture.

## 15. Inspect outputs before copying them

```bash
find experiment_outputs -maxdepth 5 -type f -print | sort
du -sh experiment_outputs

python -m json.tool experiment_outputs/exp1_100/summary.json
head -n 20 experiment_outputs/exp1_100/feature_stats.csv
head -n 20 experiment_outputs/exp1_100/top_questions/top_hle_questions_by_feature.csv
head -n 20 experiment_outputs/exp2/feature_2548_top5/token_feature_rankings.csv
```

## 16. Package results before deleting the instance

Run on the GPU:

```bash
cd /workspace/SIPIT

tar -czf /workspace/sipit_experiment_results.tar.gz \
  experiment_outputs

sha256sum /workspace/sipit_experiment_results.tar.gz
ls -lh /workspace/sipit_experiment_results.tar.gz
```

Copy the archive from the GPU to the local Mac:

```bash
scp \
  -i <PRIVATE_KEY> \
  -o IdentitiesOnly=yes \
  -P <SSH_PORT> \
  root@<GPU_HOST>:/workspace/sipit_experiment_results.tar.gz \
  /Users/siddhantmahajan/Desktop/SIPIT/
```

Verify locally:

```bash
cd /Users/siddhantmahajan/Desktop/SIPIT
shasum -a 256 sipit_experiment_results.tar.gz
tar -tzf sipit_experiment_results.tar.gz | sed -n '1,80p'
```

The local SHA-256 hash must match the remote `sha256sum` value.

Extract locally if desired:

```bash
cd /Users/siddhantmahajan/Desktop/SIPIT
mkdir -p restored_gpu_results
tar -xzf sipit_experiment_results.tar.gz -C restored_gpu_results
```

## 17. Optional: preserve capture tensors

The experiment result archive does not contain the raw activation tensors.
Without them, token-level analyses cannot be rerun after deleting the GPU
instance. If future reanalysis is likely, package and download the capture too:

```bash
cd /workspace/SIPIT

tar -czf /workspace/sipit_activation_outs_hle_arc_100.tar.gz \
  activation_outs_hle_arc_100

sha256sum /workspace/sipit_activation_outs_hle_arc_100.tar.gz
ls -lh /workspace/sipit_activation_outs_hle_arc_100.tar.gz
```

Then, from the local Mac:

```bash
scp \
  -i <PRIVATE_KEY> \
  -o IdentitiesOnly=yes \
  -P <SSH_PORT> \
  root@<GPU_HOST>:/workspace/sipit_activation_outs_hle_arc_100.tar.gz \
  /Users/siddhantmahajan/Desktop/SIPIT/
```

Capture tensors are much larger than the aggregate experiment results, so this
step is optional.

## 18. Final deletion checklist

Do not delete the instance until all applicable items are confirmed:

- [ ] `summary.json` reports the expected HLE and ARC counts.
- [ ] `feature_statistics.pt`, `feature_stats.csv`, and `feature_sets.json` exist.
- [ ] Top-question JSON and CSV exist.
- [ ] Required Experiment 2 JSON and CSV files exist.
- [ ] `sipit_experiment_results.tar.gz` was downloaded locally.
- [ ] Local and remote SHA-256 hashes match.
- [ ] The archive file listing was inspected locally.
- [ ] Raw capture tensors were downloaded if future token-level reruns are needed.
- [ ] Any unpushed code changes were copied back to the local repository.

Only after completing this checklist should the temporary GPU instance be
deleted.

## 19. Important interpretation caveats

The current HLE/ARC experiment is the original exploratory run. It has known
format confounds:

- HLE exact-answer prompts often end in punctuation.
- ARC prompts with included choices end in the final choice.
- Features were selected using final-token activations.
- The first 100 examples were selected rather than a shuffled, stratified
  sample.
- Prompts may be truncated at 192 tokens.

The proposed common `Answer:` suffix, random/stratified sampling, SimpleQA,
GPQA, and within-HLE subject comparisons have not yet been implemented in the
current capture pipeline. Do not assume the commands in this runbook perform
those future controls.
