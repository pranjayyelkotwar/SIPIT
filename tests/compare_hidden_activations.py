import argparse

import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument(
    "--original",
    default="/Users/pranjayyelkotwar/Desktop/Dystopian_Bench/SIPIT/llama31_8b_l19/original_bundle/tensors/prompt.pt",
)
parser.add_argument(
    "--reconstructed",
    default="/Users/pranjayyelkotwar/Desktop/Dystopian_Bench/SIPIT/llama31_8b_l19/prompt_sae_reconstructed.pt",
)
parser.add_argument("--tokens", action="store_true")
args = parser.parse_args()

orig_path = args.original
recon_path = args.reconstructed

orig_obj = torch.load(orig_path, map_location="cpu")
recon_obj = torch.load(recon_path, map_location="cpu")

x = orig_obj["hidden_states"].detach().float().cpu()
y = recon_obj["hidden_states"].detach().float().cpu()

print("original:", x.shape, x.dtype)
print("reconstructed:", y.shape, y.dtype)

assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"

diff = y - x

flat_x = x.flatten()
flat_y = y.flatten()
flat_diff = diff.flatten()

cosine = F.cosine_similarity(flat_x, flat_y, dim=0)
mse = torch.mean(flat_diff ** 2)
rmse = torch.sqrt(mse)
mae = torch.mean(torch.abs(flat_diff))
max_abs = torch.max(torch.abs(flat_diff))

l2_error = torch.linalg.vector_norm(flat_diff)
relative_l2 = l2_error / torch.linalg.vector_norm(flat_x)

signal_power = torch.mean(flat_x ** 2)
snr_db = 10 * torch.log10(signal_power / mse)

print("\nGlobal similarity")
print(f"cosine similarity: {cosine.item():.8f}")
print(f"relative L2 error: {relative_l2.item():.8f}")
print(f"L2 error:          {l2_error.item():.8f}")
print(f"MSE:               {mse.item():.8g}")
print(f"RMSE:              {rmse.item():.8g}")
print(f"MAE:               {mae.item():.8g}")
print(f"max abs error:     {max_abs.item():.8g}")
print(f"SNR dB:            {snr_db.item():.4f}")

# If hidden_states shape is like [batch, seq, hidden_dim],
# this compares each hidden vector independently.
if x.ndim >= 2:
    per_vec_cos = F.cosine_similarity(x, y, dim=-1)
    per_vec_rmse = torch.sqrt(torch.mean(diff ** 2, dim=-1))

    print("\nPer-vector over last dim")
    print(f"mean cosine: {per_vec_cos.mean().item():.8f}")
    print(f"min cosine:  {per_vec_cos.min().item():.8f}")
    print(f"mean RMSE:   {per_vec_rmse.mean().item():.8g}")
    print(f"max RMSE:    {per_vec_rmse.max().item():.8g}")

    print("\nCosine quantiles")
    print(torch.quantile(per_vec_cos.flatten(), torch.tensor([0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])))

    print("\nRMSE quantiles")
    print(torch.quantile(per_vec_rmse.flatten(), torch.tensor([0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])))

    if args.tokens:
        input_ids = orig_obj.get("input_ids")
        prompt = orig_obj.get("prompt")
        print("\nPer-token diagnostics")
        if prompt is not None:
            print(f"prompt: {prompt!r}")
        for idx, (token_cos, token_rmse) in enumerate(zip(per_vec_cos.flatten(), per_vec_rmse.flatten())):
            token_id = None
            if torch.is_tensor(input_ids) and input_ids.ndim == 1 and idx < input_ids.numel():
                token_id = int(input_ids[idx])
            token_label = f" token_id={token_id}" if token_id is not None else ""
            print(
                f"{idx:04d}{token_label} "
                f"cos={token_cos.item():.8f} "
                f"rmse={token_rmse.item():.8g} "
                f"orig_norm={torch.linalg.vector_norm(x[idx].float()).item():.8g} "
                f"recon_norm={torch.linalg.vector_norm(y[idx].float()).item():.8g}"
            )
