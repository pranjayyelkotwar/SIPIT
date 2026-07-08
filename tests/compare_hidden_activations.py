import torch
import torch.nn.functional as F

orig_path = "/Users/pranjayyelkotwar/Downloads/hidden-state-bundle/tensors/prompt.pt"
recon_path = "/Users/pranjayyelkotwar/Downloads/hidden-state-bundle/tensors/prompt_sae_reconstructed.pt"

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