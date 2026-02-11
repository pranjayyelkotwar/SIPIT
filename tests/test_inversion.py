import torch

from src.algorithm import SIPIT, BruteForce, HardPrompts
SEED = 1234
PROMPT = "Hello world"


def _tokenize(prompt, tokenizer, device):
    enc = tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)
    return torch.tensor(enc['input_ids'], dtype=torch.long, device=device)


def _run_inversion(algorithm, input_ids, model, tokenizer, layer_idx):
    match, time_taken, timesteps, times = algorithm.inversion_attack(
        input_ids=input_ids,
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        step_size=1.0,
        seed=SEED,
    )
    assert match is not None, "Inversion returned None"
    assert time_taken > 0
    return match


class TestSIPIT:
    def test_exact_inversion(self, model):
        m, tokenizer, device, layer_idx, model_id, special_start_token = model
        input_ids = _tokenize(PROMPT, tokenizer, device)

        algo = SIPIT(log_dir='logs', log_name=None, special_start_token_id=special_start_token)
        assert _run_inversion(algo, input_ids, m, tokenizer, layer_idx)


class TestBruteForce:
    def test_exact_inversion(self, gpt2):
        m, tokenizer, device, layer_idx = gpt2
        input_ids = _tokenize(PROMPT, tokenizer, device)

        algo = BruteForce(log_dir='logs', log_name=None)
        assert _run_inversion(algo, input_ids, m, tokenizer, layer_idx)


class TestHardPrompts:
    def test_completes(self, gpt2):
        m, tokenizer, device, layer_idx = gpt2
        input_ids = _tokenize(PROMPT, tokenizer, device)

        algo = HardPrompts(log_dir='logs', log_name=None, use_scheduler=True)
        match = _run_inversion(algo, input_ids, m, tokenizer, layer_idx)
        # HardPrompts may not always achieve exact inversion on short prompts,
        # so we only assert it completes without error.
