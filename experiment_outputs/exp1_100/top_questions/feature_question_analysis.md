# Analysis of Top HLE Questions by SAE Feature

## Scope

This report analyzes the top ten HLE questions ranked by **final-token SAE
activation** for features `2548`, `11089`, `14847`, `36468`, `36484`, `57138`,
`66188`, `69504`, `106542`, and `121471`.

The source is `top_hle_questions_by_feature.json`, produced from 100 HLE
questions at LlamaScope layer 22 (`l22r_32x`). The observations below are
descriptive hypotheses, not established feature-to-concept mappings.

## Main findings

1. **The features are not cleanly separated.** The 100 feature/question slots
   contain only 41 unique questions. Several questions occur in five or six
   feature lists, suggesting correlated or co-activating features rather than
   ten independent concepts.
2. **A broad shared signal is visible.** Most top questions request a precise,
   exact answer about a highly technical formal object. Mathematical notation,
   abstract structures, classification, invariants, ranks, dimensions, and
   exact counts are common.
3. **Some narrower tendencies exist.** Features `2548` and `66188` are especially
   mathematical; `69504` leans toward discrete counting or construction;
   `36484` appears sensitive to explicit response-format constraints; and
   `57138` favors longer, context-heavy prompts.
4. **Formatting remains a plausible explanation.** These activations were taken
   only at the final captured token. Ending punctuation, answer-format language,
   truncation at 192 tokens, and the `exactMatch` HLE format may affect the
   result as much as question semantics.

All 100 ranked entries have HLE question type `exactMatch`, so question type
cannot explain differences among these ten features, but it can still explain
part of their separation from ARC.

## Cross-feature overlap

The strongest repeated questions are:

| Question theme | Number of feature lists | Features |
|---|---:|---|
| Wasserstein regular subgradient | 6 | 2548, 11089, 14847, 36468, 57138, 66188 |
| Concentric-circle prototype classifier | 6 | 11089, 14847, 36468, 57138, 69504, 121471 |
| Six-dimensional Lie algebra/Poincaré polynomial | 5 | 2548, 36468, 36484, 66188, 106542 |
| Higher central charge of an Abelian theory | 5 | 2548, 11089, 36468, 66188, 121471 |
| Transformer circuit-complexity upper bound | 5 | 11089, 14847, 36484, 57138, 121471 |
| Soft-label kNN prototype count | 5 | 14847, 36484, 57138, 69504, 121471 |

The following feature pairs share five of their ten top questions (Jaccard
similarity approximately 0.33):

- `14847`–`36468`
- `14847`–`57138`
- `36468`–`57138`
- `36468`–`66188`
- `36468`–`121471`
- `57138`–`121471`
- `66188`–`106542`

This overlap argues against assigning sharply distinct semantic labels from
these results alone.

## Feature-level summary

| Feature | Subject tendency in top 10 | Tentative interpretation | Confidence |
|---:|---|---|---|
| 2548 | 8 Math, 2 Physics | Dense formal mathematics and theoretical-physics notation | Moderate |
| 11089 | 6 Math, 2 CS/AI, 1 Physics, 1 Other | Direct request for an exact quantity, bound, rank, or class | Low–moderate |
| 14847 | Math/Physics/CS mixture | Classification or invariant of a formal model | Low–moderate |
| 36468 | 5 Math plus Physics/CS/Other | Structural property of an abstract system | Low |
| 36484 | Highly mixed subjects | Explicit answer constraints or response-format language | Moderate |
| 57138 | Highly mixed, longest prompts | Long technical setup followed by a precise conclusion | Low–moderate |
| 66188 | 8 Math | Algebraic/topological/combinatorial invariant | Moderate |
| 69504 | 5 Math plus CS, puzzles, linguistics | Discrete counting, construction, or minimum-size problem | Moderate |
| 106542 | 7 Math plus Chemistry/Biology/Other | Constraint-heavy exact calculation or invariant | Low–moderate |
| 121471 | Math/Physics/CS mixture | Canonical descriptor or classification of a formal system | Low–moderate |

## Feature 2548

### Observed similarities

- Eight of the ten questions are labeled Math and the other two Physics.
- All ten contain mathematical notation.
- The questions concern conormal spaces, gamma matrices, Lie algebras,
  irreducibility modulo primes, stochastic processes, Wasserstein geometry,
  functional equations, central charges, Artin groups, and homotopy groups.
- Most prompts begin with “Let,” “Consider,” or “Recall,” which is characteristic
  of formal mathematical problem statements.

### Tentative interpretation

`2548` may be associated with **advanced symbolic mathematical formalism or
abstract mathematical objects**. It does not appear specific to one branch of
mathematics: analysis, algebra, probability, geometry, and theoretical physics
are all represented.

### Alternative explanation

The feature may respond to LaTeX density, formal variable introduction, or the
general HLE exact-answer style rather than mathematical abstraction itself.

## Feature 11089

### Observed similarities

- All ten top prompts end in a question mark.
- Questions repeatedly ask for an exact size, minimum, upper bound, rank,
  probability, set count, torsion order, charge, or function space.
- The subjects are broader than feature `2548`, including knot coloring,
  nearest-neighbor prototypes, transformer complexity, flag-matrix rank, and
  finite group representations.

### Tentative interpretation

`11089` may encode a **direct exact-value interrogative**: “what/how many/what
class does this formal object have?” The commonality seems more related to the
requested output than to a specific topic.

### Alternative explanation

Because every example ends in `?` and activation is measured at the final
token, the feature could partly reflect interrogative punctuation or syntax.

## Feature 14847

### Observed similarities

- All ten prompts end in a question mark and eight contain mathematical
  notation.
- Common requested outputs include a K-matrix, complexity class, prototype
  count, matrix rank, topological-invariant group, information-theoretic
  maximum, particle mass, and homotopy rank.
- Physics, CS/AI, and Math are all strongly represented.

### Tentative interpretation

`14847` may be associated with **classifying a formal model or extracting a
canonical invariant from it**.

### Alternative explanation

Its five-question overlap with both `36468` and `57138` suggests this may be a
member of a broader correlated technical-question cluster rather than a unique
classification feature.

## Feature 36468

### Observed similarities

- Five questions are Math; the remainder span Physics, CS/AI, and a simulation
  question.
- Repeated outputs include K-matrix, simulation validity, subgradient,
  homotopy rank, central charge, homology, f-vector, circuit class, and
  Poincaré polynomial.
- It has five-question overlaps with `14847`, `57138`, `66188`, and `121471`.

### Tentative interpretation

The broadest plausible label is **a requested structural property or invariant
of a described system**.

### Confidence assessment

Low. The topics and answer types are heterogeneous, and the overlap with four
other features is unusually high. Token-level and controlled-prompt evidence is
needed before giving this feature a semantic label.

## Feature 36484

### Observed similarities

- This is the most subject-diverse list: humanities, linguistics, chemistry,
  mathematics, flags, Rubik’s Cube, CS/AI, and physics.
- Several prompts explicitly constrain the required response:
  - answer yes or no;
  - return one IPA string;
  - give a value in `mol L^-1`;
  - determine an exponent `alpha`;
  - return a count, rank, class, or squared mass.
- Six of ten question texts end in a period rather than a question mark.

### Tentative interpretation

`36484` may be related to **explicit answer-format or exact-response
specification**, rather than a subject-area concept.

### Best validation test

Take the same questions and remove phrases such as “Your final answer should…”
or “giving your answer in…”. If activation falls while semantic content stays
fixed, the response-format hypothesis becomes much stronger.

## Feature 57138

### Observed similarities

- This group has one of the longest average question lengths, approximately 68
  words.
- It includes long contextual setups: Kant’s theory of judgment, Wasserstein
  geometry, a fluid simulation, prototype-classifier constructions, historical
  riddles, and historical linguistics.
- Subjects are highly mixed, and only four of ten questions contain the detected
  LaTeX-style notation.

### Tentative interpretation

`57138` may respond to a **long, specialized setup that must be compressed into
a precise conclusion**.

### Alternative explanation

The feature may encode prompt length, discourse structure, or proximity to the
192-token truncation boundary. This should be checked before proposing a
semantic meaning.

## Feature 66188

### Observed similarities

- Eight of the ten questions are Math.
- Topics are concentrated in Lie algebra, moduli spaces, VC dimension, central
  charge, spin bordism, polytopes, conormal spaces, Wasserstein geometry,
  configuration-space genus, and polynomial asymptotics.
- Many ask for a named invariant: Poincaré polynomial, homology, VC dimension,
  central charge, bordism group, f-vector, conormal order, genus, or exponent.

### Tentative interpretation

`66188` may represent **computing an algebraic, topological, or combinatorial
invariant**. This is one of the more coherent feature lists.

### Relationship to feature 2548

Both are strongly mathematical. `2548` appears broader and more notation-heavy,
whereas `66188` is somewhat more concentrated on named structural invariants.

## Feature 69504

### Observed similarities

- Several questions concern finite or discrete systems: cyclic hotel lights,
  Conway’s Game of Life, Rubik’s Cube stickers, integer solutions, and prototype
  counts.
- Common requested outputs are an exact count, minimum, probability, rational
  value, or reconstructed string.
- Compared with `2548` and `66188`, fewer examples are purely abstract symbolic
  mathematics.

### Tentative interpretation

`69504` may be associated with **discrete construction, counting, or minimum-size
problems requiring a single exact answer**.

### Alternative explanation

The linguistics examples indicate that “construct/recover one exact object” may
fit better than “combinatorics” alone.

## Feature 106542

### Observed similarities

- Seven questions are Math; the remainder are Chemistry, Biology/Medicine, and
  a flag-rank problem.
- The list includes rank, solubility, Poincaré polynomial, receptor-chain cause
  lists, asymptotic degree, f-vector, genus, eigenvalue-set count, homology, and
  Game-of-Life growth.
- The average prompt length is approximately 71 words, the highest of the ten
  lists, although this is heavily influenced by the long biology question.
- Seven question texts end in a period.

### Tentative interpretation

`106542` may encode a **constraint-heavy calculation that yields a precise
numerical or algebraic descriptor**.

### Confidence assessment

Low–moderate. Its five-question overlap with `66188` indicates that the apparent
meaning may largely be the same mathematical-invariant signal.

## Feature 121471

### Observed similarities

- The list is balanced across Math, Physics, and CS/AI, with one simulation
  question.
- Requested outputs include f-vector, system validity, central charge,
  K-matrix, VC dimension, prototype count, gamma-matrix factor, torsion count,
  and conormal space.
- Seven questions contain mathematical notation.

### Tentative interpretation

`121471` may represent **requesting the canonical descriptor, classification,
or invariant of a formal system**.

### Relationship to other features

It is close to the proposed meanings of `14847` and `36468`, and shares five top
questions with both `36468` and `57138`. A distinct interpretation is not yet
justified.

## Provisional feature groups

The ten candidates may be more usefully treated as overlapping groups:

### Formal mathematical abstraction

- `2548`
- `66188`
- possibly `36468` and `106542`

### Classification, invariant, or canonical output

- `11089`
- `14847`
- `36468`
- `121471`

### Exact discrete construction or count

- `69504`
- possibly `11089`

### Prompt structure or answer-format sensitivity

- `36484`
- `57138`
- possibly `106542`

These groups overlap substantially and should not be treated as mutually
exclusive.

## Recommended next tests

1. **Inspect token-level activations.** For each feature, run Experiment 2 on at
   least five top questions. Check whether the maximum occurs on mathematical
   terms, answer-format language, punctuation, or the final token.
2. **Use minimal pairs.** Preserve meaning while changing notation, question
   syntax, length, or response instructions.
3. **Fix the final token.** Recapture prompts with the same suffix, such as
   `Answer:`, so final-token identity is controlled.
4. **Check truncation.** Determine which prompts reached the 192-token limit.
   Their “final token” is a truncation-boundary token, not necessarily the end of
   the question.
5. **Compare matched formats.** Compare HLE exact-answer questions to an ARC or
   other control set with the same prompt and answer format.
6. **Inspect ARC counterexamples.** For each feature, examine the strongest ARC
   activations, even when ARC prevalence is below 10%. Those examples help
   distinguish semantic selectivity from HLE-specific formatting.
7. **Validate on fresh questions.** The same 100 HLE questions were used to
   select and interpret these features. Test the hypotheses on a new sample.

## Bottom line

The most defensible current conclusion is that these features form a correlated
cluster associated with **technical exact-answer prompts involving formal
objects, invariants, classifications, or discrete constructions**. Features
`2548`, `66188`, and `69504` show the clearest narrower tendencies. Features
`36468`, `57138`, and `121471` remain too overlapping to support distinct
semantic labels. Feature `36484` is the strongest candidate for a non-semantic
prompt/answer-format signal.
