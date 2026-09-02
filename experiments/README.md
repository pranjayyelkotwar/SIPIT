# SAE experiments

These scripts reuse the residual-stream captures produced by
`activation_capture/capture_activations.py` and the LlamaScope loader in
`evolutionary search/llamascope.py`. They do not load the language model or
download the question datasets again.

## 1. HLE versus ARC at the final prompt token

First capture both datasets at the desired SAE layer:

```bash
python activation_capture/capture_activations.py \
  --dataset-source qa \
  --qa-datasets arc_easy,hle \
  --layers 22 \
  --output-dir activation_outs/hle_arc_l22
```

Then compare only the last token state from every captured question:

```bash
python experiments/compare_last_token_features.py \
  --activation-dir activation_outs/hle_arc_l22 \
  --metadata-file activation_outs/hle_arc_l22/metadata_rank0.jsonl \
  --layer 22 \
  --activation-threshold 0 \
  --min-prevalence 0.10 \
  --output-dir experiment_outputs/exp1
```

The output directory contains:

- `feature_sets.json`: HLE, ARC, set differences, and intersection.
- `feature_stats.csv`: the largest absolute differences in mean activation.
- `feature_statistics.pt`: complete per-feature means, standard deviations, and
  activation prevalence for subsequent analysis.
- `summary.json`: settings, sample counts, missing-file counts, and set sizes.

The feature-set filter is:

```text
mean activation > --min-mean
and activation prevalence >= --min-prevalence
```

Activation prevalence is the fraction of questions whose final-position
activation exceeds `--activation-threshold`.

For the cleanest comparison, capture HLE and ARC with the same prompt template,
including the same suffix. Otherwise, final-position features can reflect
dataset-specific formatting or final-token identity.

## 2. Tokens activating a selected feature

Supply one captured question tensor, its exact prompt, and one or more features:

```bash
python experiments/inspect_token_feature_activations.py \
  --activation activation_outs/hle_arc_l22/layer_22/activations_l22_idx0.pt \
  --prompt-file question_prompt.txt \
  --layer 22 \
  --feature-id 1234 \
  --feature-id 5678 \
  --output experiment_outputs/exp2/token_feature_rankings.csv
```

The CSV ranks token positions separately for each feature and includes token
IDs, decoded token pieces, activations, and short contexts. Its neighboring
JSON file records the model, SAE, layer, and feature IDs used.

The prompt and `--add-bos-token` setting must match the original capture. If the
capture truncated a long prompt, the script applies the same prefix length from
the activation tensor.

## Rank questions for selected final-token features

Rank HLE questions by the ten candidate features selected from Experiment 1:

```bash
python experiments/rank_questions_by_feature.py \
  --activation-dir activation_outs_hle_arc_100 \
  --metadata-file activation_outs_hle_arc_100/metadata_rank0.jsonl \
  --layer 22 \
  --device cuda \
  --dtype bfloat16 \
  --output-dir experiment_outputs/exp1_100/top_questions
```

The default feature IDs are `2548`, `11089`, `14847`, `36468`, `36484`,
`57138`, `66188`, `69504`, `106542`, and `121471`. Override them by repeating
`--feature-id`. The output JSON and CSV contain the ten strongest HLE questions
for each feature, their final-token activations, question metadata, exact prompt
text, and activation paths.

## Token analysis for the top questions of one feature

Feature `2548` is the default because its top questions form a relatively
coherent advanced-mathematics/theoretical-physics cluster. Run Experiment 2 on
its five strongest questions with:

```bash
python experiments/analyze_top_questions_tokens.py \
  --ranked-json experiment_outputs/exp1_100/top_questions/top_hle_questions_by_feature.json \
  --feature-id 2548 \
  --top-k-questions 5 \
  --top-k-tokens 20 \
  --activation-dir activation_outs_hle_arc_100 \
  --device cuda \
  --dtype bfloat16 \
  --output-dir experiment_outputs/exp2/feature_2548_top5
```

The script loads the SAE once and writes combined token rankings for all five
questions to `token_feature_rankings.csv` and `token_feature_rankings.json`.
Tokens whose selected-feature activation is zero are omitted.
