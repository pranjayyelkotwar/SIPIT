import random
from time import gmtime, strftime

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel


def format_time_minutes(seconds: float) -> str:
    return strftime("%M:%S", gmtime(seconds))

def format_time_hours(seconds: float) -> str:
    return strftime("%H:%M:%S", gmtime(seconds))


def terminal_safe_repr(token: str) -> str:
    result = []

    for ch in token:
        code = ord(ch)

        # Printable ASCII
        if 32 <= code <= 126:
            result.append(ch)

        # Common control characters with named escapes
        elif ch == '\n':
            result.append('\\n')
        elif ch == '\r':
            result.append('\\r')
        elif ch == '\t':
            result.append('\\t')
        elif ch == '\b':
            result.append('\\b')
        elif ch == '\f':
            result.append('\\f')
        elif ch == '\v':
            result.append('\\v')
        elif ch == '\a':
            result.append('\\a')
        elif ch == '\0':
            result.append('\\0')

        # Other ASCII control chars or DEL (0–31, 127)
        elif code < 128:
            result.append(f'\\x{code:02x}')

        # Non-ASCII → UTF-8 hex escape
        else:
            result.append(''.join(f'\\x{b:02x}' for b in ch.encode('utf-8')))

    return ''.join(result)


def format_token(token: str, length: int = 15) -> str:
    token = terminal_safe_repr(token)
    if len(token) > length:
        token = token[:length - 3] + '...'
    return token


def set_seed(seed: int = 8):
    '''Set seed for reproducibility.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_indent(string: str, to_replace: str = '\n', tab_count: int = 1):
    tabs = '\t' * tab_count
    print(f'{tabs}{string.replace(to_replace, to_replace + tabs)}')


def all_token_ids_strict(
    tokenizer,
    include_special: bool = True,
) -> list[int]:
    '''
    Return the sorted set of valid token IDs.

    - Base vocab  : [0, tokenizer.vocab_size)
    - Added tokens: tokenizer.get_added_vocab().values()
    - Special     : tokenizer.all_special_ids
    '''
    base = set(range(tokenizer.vocab_size))

    special_ids = set(getattr(tokenizer, 'all_special_ids', []))

    ids = set(base)
    if include_special:
        ids |= special_ids
    else:
        ids -= special_ids  # ensure specials excluded even if include_added=True

    return sorted(ids)


def _get_attr_by_path(root, path: str):
    """Get attribute by a dotted path, or raise AttributeError."""
    obj = root
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj

def replace_last_norm(model_id: str, model: PreTrainedModel):
    attr_name = {
        'roneneldan/TinyStories-1M': 'transformer.ln_f',
        'roneneldan/TinyStories-8M': 'transformer.ln_f',
        'roneneldan/TinyStories-33M': 'transformer.ln_f',
        
        'openai-community/gpt2': 'transformer.ln_f',
        'openai-community/gpt2-medium': 'transformer.ln_f',
        'openai-community/gpt2-large': 'transformer.ln_f',

        'google/gemma-3-1b-pt': 'model.norm',
        'google/gemma-3-4b-pt': 'model.norm',
        'google/gemma-3-12b-pt': 'model.norm',
        
        'microsoft/Phi-4-mini-instruct': 'model.norm',
        'mistralai/Mistral-7B-v0.1': 'model.norm',
        'meta-llama/Meta-Llama-3-8B': 'model.norm',
        'meta-llama/Llama-3.1-8B': 'model.norm',
        'locuslab/tofu_ft_llama2-7b': 'model.norm',

        'Qwen/Qwen2.5-0.5B': 'model.norm',
    }

    if model_id not in attr_name:
        raise NotImplementedError(f'Model ID `{model_id}` is not supported!')

    parts = attr_name[model_id].split('.')
    parent_path, leaf = '.'.join(parts[:-1]), parts[-1]

    parent = _get_attr_by_path(model, parent_path) if parent_path else model

    if not hasattr(model, 'norm_bk'):
        model.norm_bk = getattr(parent, leaf) # type: ignore
        
    setattr(parent, leaf, nn.Identity())