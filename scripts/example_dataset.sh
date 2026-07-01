#!/bin/bash -l

seed=8

sipit \
    --command create-dataset-collection \
    --dataset-config src/data/datasets.json \
    --dataset-name SIPIT-Collection \
    --overwrite \
    --seed $seed \
    --tokens 10

sipit \
    --command create-dataset-random \
    --dataset-name SIPIT-Random \
    --seed $seed \
    --prompts 10 \
    --tokens 10


method="SIPIT" # Choose between "SIPIT", "BruteForce" or "HardPrompts"
layer_idx=-1
precision=32
step_size=1.0

######################
# Collection Dataset #
######################

sipit \
    --command invert-dataset \
    --method $method \
    -i data/SIPIT-Collection/gpt2 \
    -o data/experiments/gpt_collection.csv \
    --model-id "openai-community/gpt2" \
    --seed $seed \
    --layer-idx $layer_idx \
    --precision $precision \
    --step-size $step_size

sipit \
    --command invert-dataset \
    --method $method \
    -i data/SIPIT-Collection/gpt2 \
    -o data/experiments/mistral_collection.csv \
    --model-id "mistralai/Mistral-7B-v0.1" \
    --seed $seed \
    --layer-idx $layer_idx \
    --precision $precision \
    --step-size $step_size

sipit \
    --command invert-dataset \
    --method $method \
    -i data/SIPIT-Collection/gpt2 \
    -o data/experiments/llama_collection.csv \
    --model-id "meta-llama/Meta-Llama-3-8B" \
    --seed $seed \
    --layer-idx $layer_idx \
    --precision $precision \
    --step-size $step_size

##################
# Random Dataset #
##################

sipit \
    --command invert-dataset \
    --method $method \
    -i data/SIPIT-Random \
    -o data/experiments/gpt_random.csv \
    --model-id "openai-community/gpt2" \
    --seed $seed \
    --layer-idx $layer_idx \
    --precision $precision \
    --step-size $step_size

sipit \
    --command invert-dataset \
    --method $method \
    -i data/SIPIT-Random \
    -o data/experiments/mistral_random.csv \
    --model-id "mistralai/Mistral-7B-v0.1" \
    --seed $seed \
    --layer-idx $layer_idx \
    --precision $precision \
    --step-size $step_size

sipit \
    --command invert-dataset \
    --method $method \
    -i data/SIPIT-Random \
    -o data/experiments/llama_random.csv \
    --model-id "meta-llama/Meta-Llama-3-8B" \
    --seed $seed \
    --layer-idx $layer_idx \
    --precision $precision \
    --step-size $step_size
