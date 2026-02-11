#!/bin/bash -l

seed=8
layer_idx=-1
precision=32
step_size=1.0
prompt="Hello there, SIPIT should be able to efficiently and exactly invert this prompt!"


method="SIPIT"
sipit \
    --command invert-single \
    --method $method \
    --model-id "openai-community/gpt2" \
    --seed $seed \
    --layer-idx $layer_idx \
    --precision $precision \
    --step-size $step_size \
    --prompt "$prompt"
    

sipit \
    --command invert-single \
    --method $method \
    --model-id "mistralai/Mistral-7B-v0.1" \
    --seed $seed \
    --layer-idx $layer_idx \
    --precision $precision \
    --step-size $step_size \
    --prompt "$prompt"

sipit \
    --command invert-single \
    --method $method \
    --model-id "meta-llama/Llama-3.1-8B" \
    --special-start-token 128000 \
    --seed $seed \
    --layer-idx $layer_idx \
    --precision $precision \
    --step-size $step_size \
    --prompt "$prompt"


method="BruteForce"
sipit \
    --command invert-single \
    --method $method \
    --model-id "openai-community/gpt2" \
    --seed $seed \
    --layer-idx $layer_idx \
    --step-size $step_size \
    --prompt "$prompt"

method="HardPrompts"
sipit \
    --command invert-single \
    --method $method \
    --model-id "openai-community/gpt2" \
    --seed $seed \
    --layer-idx $layer_idx \
    --precision $precision \
    --step-size $step_size \
    --vocab-scale-factor 7250 \
    --scheduler \
    --prompt "$prompt"