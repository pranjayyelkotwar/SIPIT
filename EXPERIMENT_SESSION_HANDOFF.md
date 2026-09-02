# SAE Experiment Session Handoff

## Purpose

This document gives a new session enough context to continue the SAE analysis
without reconstructing the full conversation. It records:

- the research goal;
- the code that was added;
- the exact experimental configuration;
- quantitative results;
- feature-level and token-level findings;
- the main confound discovered;
- retained and missing artifacts;
- recommended next experiments.

For complete GPU setup and execution commands, see
[`GPU_RESTART_RUNBOOK.md`](GPU_RESTART_RUNBOOK.md).

## Research goal

The original goal was to identify LlamaScope SAE features that distinguish
Humanity's Last Exam (HLE) questions from ARC-Easy questions, then inspect which
tokens activate those features and infer possible feature-to-concept mappings.

Experiment 1 intentionally examined only the **final prompt-token hidden state**
for each question. Experiment 2 examined every token position for selected
features and selected questions.

The desired long-term outcome is to find features representing substantive
concepts, subjects, or reasoning patterns—not punctuation, special tokens, or
prompt formatting.

## Repository state

- Repository: `https://github.com/pranjayyelkotwar/SIPIT.git`
- Working branch: `siddhant/dev`
- Latest observed commit: `cea7b06` (`only include postive tokens with +ve activation`)
- Model: `meta-llama/Llama-3.1-8B`
- SAE: LlamaScope `llama_scope_lxr_32x`, SAE ID `l22r_32x`
- Capture/SAE layer: 22
- SAE feature count: 131,072

The layer convention follows the repository's current activation-capture code:
requested Hugging Face hidden-state index 22 is captured from decoder block 21.
Do not mix these captures with older captures made before the layer-alignment
fix.

## Experiment code

### `experiments/compare_last_token_features.py`

Consumes activation tensors and metadata produced by
`activation_capture/capture_activations.py`.

For each question it:

1. Loads the residual-stream tensor.
2. Selects only `selected_state(..., -1)`, the final captured token.
3. Encodes that state through the matching LlamaScope SAE.
4. Accumulates per-feature mean, standard deviation, and prevalence separately
   for HLE and ARC.
5. Creates HLE, ARC, set-difference, and intersection feature sets.

Feature-set membership is:

```text
mean activation > min_mean
and
fraction of questions with activation > activation_threshold >= min_prevalence
```

The default values are:

```text
activation_threshold = 0
min_mean = 0
min_prevalence = 0.10
```

The script also ranks features by the absolute HLE-minus-ARC mean difference
and reports a standardized difference.

### `experiments/rank_questions_by_feature.py`

For selected features, this script re-encodes the final token of every HLE
question and ranks the strongest questions. It writes a grouped JSON and flat
CSV containing activation values, question text, prompt text, metadata, and
activation paths.

Its built-in candidate feature list is:

```text
2548, 11089, 14847, 36468, 36484,
57138, 66188, 69504, 106542, 121471
```

### `experiments/inspect_token_feature_activations.py`

Runs token-level SAE analysis for one activation tensor and one or more feature
IDs. It tokenizes the exact captured prompt, aligns tokens with saved states,
encodes every token, and ranks token positions by feature activation.

It now omits tokens whose selected-feature activation is zero.

### `experiments/analyze_top_questions_tokens.py`

Batch version of Experiment 2. It reads the ranked-question JSON, loads the SAE
once, and analyzes the top N questions for one feature. It writes a combined CSV
and JSON. Zero-activation token positions are omitted.

### `experiments/analysis.py`

Local helper that reads `feature_statistics.pt`, writes filtered HLE/ARC text
files, and intersects HLE-high-prevalence features with ARC-low-prevalence
features. This helper is currently untracked and may need to be copied manually
to a new GPU instance.

## Activation capture configuration

The main capture used:

```text
model: meta-llama/Llama-3.1-8B
datasets: HLE and ARC-Easy
samples: 100 HLE + 100 ARC
layer: 22
maximum prompt length: 192 tokens
batch size: 4
BOS token: included
choices: included
capture dtype: bfloat16
```

The capture selected the first 100 examples from each dataset rather than a
random or stratified sample. This is an important limitation.

The prompt template was approximately:

```text
Answer the following question.

Question: <question text>
Choices:
A. ...
B. ...
...
```

Choices were added only when present. Consequently:

- HLE exact-answer prompts usually ended directly in question punctuation.
- ARC prompts ended in the text of the final answer choice.

This format mismatch became the experiment's main confound.

## Experiment 1 results

The 100-HLE/100-ARC run reported:

```text
HLE questions: 100
ARC questions: 100
missing HLE files: 0
missing ARC files: 0

HLE set: 36 features
ARC set: 40 features
HLE minus ARC: 16 features
ARC minus HLE: 20 features
intersection: 20 features
```

The initial 10+10 smoke run had produced far more apparent candidates because
10% prevalence meant activation on only one question:

```text
HLE set: 159
ARC set: 121
HLE minus ARC: 132
ARC minus HLE: 94
intersection: 27
```

This demonstrated why the 10-question run was too permissive.

## Candidate-feature filter

The ten features selected for interpretation satisfied approximately:

```text
HLE prevalence >= 0.30
ARC prevalence <= 0.05
```

Their aggregate results were:

| Feature | HLE mean | ARC mean | HLE prevalence | ARC prevalence | Standardized difference |
|---:|---:|---:|---:|---:|---:|
| 2548 | 1.321 | 0.000 | 0.40 | 0.00 | 1.079 |
| 11089 | 1.472 | 0.000 | 0.33 | 0.00 | 0.981 |
| 14847 | 2.063 | 0.000 | 0.43 | 0.00 | 1.152 |
| 36468 | 1.015 | 0.000 | 0.32 | 0.00 | 0.934 |
| 36484 | 0.886 | 0.000 | 0.32 | 0.00 | 0.872 |
| 57138 | 1.623 | 0.000 | 0.38 | 0.00 | 1.014 |
| 66188 | 1.011 | 0.000 | 0.30 | 0.00 | 0.850 |
| 69504 | 2.004 | 0.040 | 0.50 | 0.02 | 1.255 |
| 106542 | 0.811 | 0.018 | 0.32 | 0.01 | 0.887 |
| 121471 | 2.002 | 0.000 | 0.47 | 0.00 | 1.254 |

These are descriptive discovery statistics. No confidence intervals,
multiple-comparison correction, or held-out validation was performed.

## Top-question comparison

The top ten HLE questions were retrieved for each candidate feature, producing
100 feature/question slots.

### Overlap

- 100 top-ten slots contained only **41 unique questions**.
- Restricting to the top five per feature produced 50 slots containing **29
  unique questions**.
- Ten independent random five-question selections from a pool of 100 would have
  an expected union of approximately 40 questions, so 29 indicates overlap
  beyond the finite-pool effect.

Frequently repeated questions included:

- the Wasserstein regular-subgradient question: six feature lists;
- the concentric-circle prototype-classifier question: six lists;
- the Lie algebra/Poincaré-polynomial question: five lists;
- the higher-central-charge question: five lists;
- the transformer circuit-complexity question: five lists;
- the soft-label kNN prototype-count question: five lists.

This suggests correlated/co-activating features rather than ten cleanly
separated semantic concepts.

### Tentative question-level trends

Before token-level inspection, the top-question lists suggested:

- `2548`: formal mathematics/theoretical physics and dense notation;
- `11089`: exact quantity, bound, rank, or class;
- `14847`: formal-model classification or invariant;
- `36468`: structural property of an abstract system;
- `36484`: explicit response-format constraints;
- `57138`: long, context-heavy technical prompts;
- `66188`: algebraic/topological/combinatorial invariants;
- `69504`: discrete counts, constructions, and minimum-size problems;
- `106542`: constraint-heavy exact calculations;
- `121471`: canonical descriptor or classification.

These semantic hypotheses weakened substantially after token-level inspection.

The detailed question-level report is retained at:

```text
experiment_outputs/exp1_100/top_questions/feature_question_analysis.md
```

## Experiment 2: token-level results

Token-level analysis was completed for features `2548`, `36484`, and `69504`,
using the top five HLE questions for each feature.

### Feature 2548

Nine positive token positions were found across five questions:

| Question | Positive activations |
|---:|---|
| 1 | `?`: 5.156 |
| 2 | `?`: 5.094; two `ld` fragments from `\\ldots`: 2.109 and 2.000 |
| 3 | `$.`: 4.875; `_`: 3.297; `fr` from `\\mathfrak`: 2.375 |
| 4 | `?`: 4.781 |
| 5 | `?`: 4.719 |

Interpretation:

- the strongest activation was the final token in all five questions;
- four of five strongest tokens were question marks;
- the fifth was a combined math-closing/period token;
- weaker activations appeared on fragments of mathematical notation.

Tentative label: **formal-question boundary, especially question marks, with
some mathematical-notation sensitivity**.

### Feature 36484

Eight positive token positions were found, all punctuation:

| Question | Positive activations |
|---:|---|
| 1 | final `.").`: 6.875; internal `?`: 2.031 |
| 2 | internal `.`: 6.344; final `.`: 5.188 |
| 3 | final `.`: 4.531 |
| 4 | final `$.`: 4.031 |
| 5 | final `.`: 3.781; internal `?`: 2.781 |

Interpretation:

- every positive activation occurred on sentence-ending punctuation;
- an internal period activated more strongly than the final period in one
  question, so this was not merely an absolute-final-position effect;
- periods were more characteristic than question marks.

Tentative label: **sentence boundary/period feature**.

### Feature 69504

Fourteen positive positions were found:

- five `<|begin_of_text|>` activations, all exactly `29.75`;
- four question marks;
- four periods;
- one `by` token in “one by one.”

The BOS state was identical across all questions because a causal transformer's
position-zero state cannot attend to later question content. With the same BOS
token and evaluation-mode model, the layer-22 state and SAE activation are
deterministic.

Thirteen of fourteen positive activations were BOS or punctuation boundaries.

Tentative label: **general text/sentence-boundary feature, with especially
strong BOS activation**.

The BOS activation did not itself cause HLE/ARC discrimination because
Experiment 1 examined only final tokens. Final punctuation likely caused the
HLE-selective prevalence.

## Principal outcome

The main result of this session is that the initial filter predominantly found
features related to **prompt boundaries, punctuation, BOS, sentence structure,
or formatting**, not deep HLE subject concepts.

The reasons are:

1. HLE and ARC prompts had different ending structures.
2. Many HLE exact-answer prompts ended directly in `?` or `.`.
3. ARC prompts with included choices ended in the last choice text.
4. Experiment 1 intentionally analyzed only final-token activations.
5. Experiment 2 selected questions by final-token activation, so observing high
   final-token activation was partly guaranteed by construction.

The experiment was still useful: it exposed a strong format confound that must
be controlled before dataset-selective SAE features can be interpreted as
conceptual features.

## Recommended next experiment

### 1. Standardize the final suffix

Every prompt should end in exactly the same token sequence, for example:

```text
Question:
<question text>

Answer:
```

For multiple choice:

```text
Question:
<question text>

Choices:
A. ...
B. ...
C. ...
D. ...

Answer:
```

The final analyzed token should be the same `:` token ID for every prompt.

The suffix must be preserved under truncation. Tokenize/truncate the question
body to `max_length - suffix_length`, then append the suffix tokens. Do not
append the suffix and subsequently right-truncate it away.

### 2. Separate HLE answer formats

Analyze independently:

- HLE exact/short-answer questions;
- HLE multiple-choice questions.

Only compare groups with matched prompt and choice formatting.

### 3. Replace or supplement ARC controls

Recommended controls:

- **SimpleQA:** structurally closer short, open-ended factual questions;
- **GPQA:** difficult expert science control for matched HLE science questions;
- **OpenWebText sentences:** nuisance baseline for punctuation/sentence-boundary
  features;
- **MMLU-Pro or MMLU:** broad academic control, but only with matched
  multiple-choice formatting.

SimpleQA is the best primary comparison for HLE exact-answer questions. GPQA is
limited to roughly 448 science questions, so it should be used as a matched
science control rather than a 1,000-question broad comparison.

### 4. Use within-HLE contrasts

Within-dataset comparisons may be more effective for concept discovery:

```text
HLE Math vs HLE non-Math
HLE Physics vs HLE non-Physics
HLE Humanities vs HLE STEM
HLE multiple-choice vs HLE short-answer
```

This reduces dataset-style confounds and uses HLE's subject/category metadata.

### 5. Add all-token concept discovery

Final-token analysis may miss features that activate where a concept is
actually mentioned. Add an analysis that:

1. encodes every content-token state;
2. masks BOS, template, suffix, padding, and punctuation-only tokens;
3. ranks features or contexts by positive activation;
4. aggregates across questions using maximum activation or activation
   prevalence over content positions;
5. retrieves top contexts for interpretation.

This is more likely to reveal mathematics, topology, physics, linguistics, or
other concept features.

### 6. Improve sampling and statistics

- Shuffle with a fixed seed instead of selecting the first N examples.
- Stratify by subject, answer type, length, and mathematical notation.
- Length-match comparison datasets.
- Use separate discovery and validation samples.
- Report mean difference, prevalence difference, standardized effect size, and
  bootstrap confidence intervals.
- Correct for testing 131,072 features, for example with false-discovery-rate
  control.
- Increasing from 100 to 1,000 reduces sampling variance but does not remove
  systematic formatting bias.

### 7. Directly identify nuisance features

For a mixed calibration corpus, calculate:

```text
boundary_score(feature) =
    activation mass on BOS/punctuation/template tokens
    --------------------------------------------------
                 total activation mass
```

Features with a high boundary score can be excluded from semantic analysis.

## Retained artifacts

The following outputs are currently present locally:

```text
experiment_outputs/exp1_100/top_questions/
├── feature_question_analysis.md
├── top_hle_questions_by_feature.csv
└── top_hle_questions_by_feature.json

exp2/feature_2548_top5/
├── token_feature_rankings.csv
└── token_feature_rankings.json

exp2/feature_36484_top5/
├── token_feature_rankings.csv
└── token_feature_rankings.json

exp2/feature_69504_top5/
├── token_feature_rankings.csv
└── token_feature_rankings.json
```

At the time of this handoff, the following aggregate Experiment 1 artifacts are
not present locally under `experiment_outputs/exp1_100/`:

```text
feature_sets.json
feature_statistics.pt
feature_stats.csv
summary.json
```

They must be restored from an archive or regenerated on a GPU if needed. Their
key observed values are preserved in this document.

Raw activation captures are also not currently present locally. Token-level
analysis cannot be rerun without recapturing or restoring them.

## Operational continuation

To restart on a temporary GPU instance:

1. Follow `GPU_RESTART_RUNBOOK.md`.
2. Clone branch `siddhant/dev`.
3. Run the optional 10+10 smoke workflow.
4. Recapture the desired datasets.
5. Before repeating the original analysis, decide whether to first implement
   the standardized `Answer:` suffix and randomized/stratified sampling.
6. Download result archives—and optionally raw captures—before deleting the
   instance.

## Suggested first task for the next session

Do **not** immediately rerun the same HLE/ARC final-token experiment at a larger
sample size. The best next implementation task is:

> Add a capture mode that guarantees a shared final `Answer:` token sequence,
> preserves that suffix during truncation, supports random/stratified sampling,
> and separates HLE exact-answer from multiple-choice questions.

After that, add SimpleQA as the matched open-ended control and rerun a small
smoke comparison before scaling.
