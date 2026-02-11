import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dotenv import load_dotenv
load_dotenv()

import transformers
transformers.logging.set_verbosity_error()

import pytest

from src.utils.model import setup
from src.utils.utils import set_seed


SEED = 1234


MODEL_CONFIGS = [
    ('openai-community/gpt2', None),
    ('mistralai/Mistral-7B-v0.1', None),
    ('meta-llama/Llama-3.1-8B', 128000),
]


@pytest.fixture(scope='session')
def gpt2():
    """Load GPT-2 once and share across all tests."""
    m, tokenizer, device, layer_idx = setup(
        model_id='openai-community/gpt2',
        precision=32,
        layer_idx=-1,
        print_stats=False,
    )
    return m, tokenizer, device, layer_idx


@pytest.fixture(scope='session', params=MODEL_CONFIGS, ids=lambda c: c[0].split('/')[-1])
def model(request):
    """Load a model once and share across all tests."""
    model_id, special_start_token = request.param
    m, tokenizer, device, layer_idx = setup(
        model_id=model_id,
        precision=32,
        layer_idx=-1,
        print_stats=False,
    )
    return m, tokenizer, device, layer_idx, model_id, special_start_token


@pytest.fixture(autouse=True)
def seed():
    """Reset random state before every test for reproducibility."""
    set_seed(SEED)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary directory for dataset tests."""
    return tmp_path
