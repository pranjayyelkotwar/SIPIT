# SIPIT

Official implementation of **SIPIT** from the paper:

> **Language Models are Injective and Hence Invertible**
> ICLR 2026

SIPIT is an algorithm for exact prompt inversion on causal language models. Given access to the hidden states at any intermediate layer of a transformer, SIPIT recovers the original input tokens by alternating between continuous gradient-based optimization and discrete projection onto the vocabulary.

This repository also includes two baseline methods — **BruteForce** (exhaustive vocabulary search) and **HardPrompts** (joint sequence optimization).

## Setup

Requires Python 3.10+ and CUDA.

```bash
git clone https://github.com/giorgosnikolaou/SIPIT.git
cd SIPIT
pip install -e .
```

For gated models (e.g. Llama, Mistral), copy the example env file and add your HuggingFace token:

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_...
```

## Usage

### Invert a single prompt

```bash
sipit \
    --command invert-single \
    --method SIPIT \
    --model-id "openai-community/gpt2" \
    --layer-idx -1 \
    --step-size 1.0 \
    --prompt "SIPIT should be able to invert this prompt!"
```

### Invert a dataset

First, create a dataset, then run inversion on it:

```bash
# Create a dataset from real text sources
# --dataset-config points to a JSON file defining the sources to sample from.
# The default config (src/data/datasets.json) includes Wikipedia, C4, ArXiv, and GitHub Python code.
sipit \
    --command create-dataset-collection \
    --dataset-config src/data/datasets.json \
    --dataset-name SIPIT-Collection \
    --tokens 10

# Or create a random token dataset
sipit \
    --command create-dataset-random \
    --dataset-name SIPIT-Random \
    --prompts 100 \
    --tokens 20

# Run inversion
sipit \
    --command invert-dataset \
    --method SIPIT \
    -i data/SIPIT-Collection/gpt2 \
    -o results.csv \
    --model-id "openai-community/gpt2"
```

### Key options

| Flag | Description | Default |
|------|-------------|---------|
| `--method` | `SIPIT`, `BruteForce`, or `HardPrompts` | `SIPIT` |
| `--model-id` | HuggingFace model identifier | `openai-community/gpt2` |
| `--layer-idx` | Layer to invert (`-1` = last) | `-1` |
| `--precision` | Weight precision: `4`, `8`, `16`, or `32` bit | `32` |
| `--step-size` | Step Size | `1.0` |
| `--special-start-token` | BOS token ID (needed for e.g. Llama-3.1) | `None` |

See `scripts/` for more complete examples.

## Testing

Install test dependencies and run the suite with pytest:

```bash
pip install -e ".[test]"
pytest
```

Tests use GPT-2 and cover exact inversion (SIPIT, BruteForce), algorithm completion (HardPrompts), dataset creation, and end-to-end dataset inversion. Add new test files to the `tests/` directory.

## Citation

```bibtex
@article{
    nikolaou2025injective,
    title={Language Models are Injective and Hence Invertible},
    author={Nikolaou*, Giorgos and Mencattini*, Tommaso and Crisostomi, Donato and Santilli, Andrea and Panagakis, Yannis and Rodol{\`a}, Emanuele},
    journal={The Fourteenth International Conference on Learning Representations (ICLR 2026)},
    year={2026},
    url = {https://arxiv.org/abs/2510.15511}
}
```