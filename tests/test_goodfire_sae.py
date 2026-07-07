import torch
from huggingface_hub import hf_hub_download

# Download SAE
sae_path = hf_hub_download(
    repo_id="Goodfire/Llama-3.1-8B-Instruct-SAE-l19",
    filename="Llama-3.1-8B-Instruct-SAE-l19.pth",
)

class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, d_in, expansion_factor):
        super().__init__()

        d_hidden = d_in * expansion_factor

        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)

    def encode(self, x):
        return torch.relu(self.encoder_linear(x))

    def decode(self, f):
        return self.decoder_linear(f)

sae = SparseAutoEncoder(
    d_in=4096,
    expansion_factor=16
)

state = torch.load(sae_path, map_location="cpu", weights_only=True)
sae.load_state_dict(state)
sae.eval()