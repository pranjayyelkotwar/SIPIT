import torch

from src.algorithm.SIPIT import SIPIT
from src.utils.model import setup

MODEL_ID = "meta-llama/Meta-Llama-3-8B"  # or exact model that produced the tensor
LAYER_IDX = 22
STEP_SIZE = 1.0
TARGET_PATH = "activations_l22_idx6573.pt"

model, tokenizer, model_name, layer_idx = setup(
    model_id=MODEL_ID,
    precision=16,
    layer_idx=LAYER_IDX,
)

target_hidden_states = torch.load(TARGET_PATH, map_location="cpu")

# Make sure shape is (seq_len, hidden_dim), not (1, seq_len, hidden_dim)
if target_hidden_states.dim() == 3:
    target_hidden_states = target_hidden_states.squeeze(0)

algo = SIPIT(
    log_dir="logs",
    log_name=None,
    special_start_token_id=None,  # set this only if the captured activations assume an extra BOS token
)

inversion_time, recovered_ids, timesteps, times = algo.find_prompt(
    model=model,
    tokenizer=tokenizer,
    layer_idx=layer_idx,
    target_hidden_states=target_hidden_states,
    step_size=STEP_SIZE,
)

print("Recovered ids:", recovered_ids)
print("Recovered text:")
print(tokenizer.decode(recovered_ids, skip_special_tokens=False))
print("Time:", inversion_time)