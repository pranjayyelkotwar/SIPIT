from collections.abc import Sequence
from typing import Any, Literal

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel, 
    PreTrainedTokenizer
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.utils.utils import replace_last_norm


def hidden_states_from_input_ids(
    input_ids: torch.LongTensor,
    model: PreTrainedModel,
    layer_idx: int,
    require_grad: bool = False
) -> torch.Tensor:
    cond = len(input_ids.shape) == 1
    if cond:
        input_ids = input_ids.unsqueeze(0) # type: ignore
    ctx = torch.enable_grad() if require_grad else torch.no_grad()
    with ctx:
        outputs: CausalLMOutputWithPast = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False
        ) # type: ignore
        hidden_states = outputs.hidden_states[layer_idx] # type: ignore
        hidden_states = (
            hidden_states if not cond else 
            hidden_states[0]
        )
    return hidden_states if require_grad else hidden_states.detach()

 
def all_hidden_states_from_input_ids(
    input_ids: torch.LongTensor,
    model: PreTrainedModel,
    require_grad: bool = False
) -> tuple[torch.Tensor, ...]:
    cond = len(input_ids.shape) == 2
    if cond:
        input_ids = input_ids.unsqueeze(0) # type: ignore
    ctx = torch.enable_grad() if require_grad else torch.no_grad()
    with ctx:
        outputs: CausalLMOutputWithPast = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False
        ) # type: ignore
        hidden_states: tuple[torch.Tensor, ...] = outputs.hidden_states # type: ignore
        hidden_states = (
            hidden_states if not cond else 
            tuple(layer_hidden_states[0] for layer_hidden_states in hidden_states)
        )
    return (
        hidden_states if require_grad else 
        tuple(layer_hidden_states.detach() for layer_hidden_states in hidden_states)
    )
    

def hidden_states_from_prompt(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer_idx: int,
    require_grad: bool = False,
    add_special_tokens: bool = False
) -> torch.Tensor:
    device = model.device # type: ignore
    encoded = tokenizer(
        prompt, 
        return_tensors='pt',
        add_special_tokens=add_special_tokens,
        truncation=True,
        max_length=min(
            getattr(tokenizer, 'model_max_length', 2048), 
            2048
        ) # type: ignore
    ) # type: ignore
    return hidden_states_from_input_ids(
        input_ids=encoded['input_ids'].to(device), # shape (1, seq_len) # type: ignore
        model=model,
        layer_idx=layer_idx,
        require_grad=require_grad
    )


def _decoder_blocks(model: PreTrainedModel) -> Sequence[Any]:
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):  # type: ignore[attr-defined]
        return model.transformer.h  # type: ignore[attr-defined]
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):  # type: ignore[attr-defined]
        return model.model.layers  # type: ignore[attr-defined]
    raise NotImplementedError(
        f'Forwarding from hidden states is not implemented for {model.__class__.__name__}.'
    )


def logits_from_hidden_states(
    hidden_states: torch.Tensor,
    model: PreTrainedModel,
    layer_idx: int,
    input_ids: torch.LongTensor | None = None,
) -> torch.Tensor:
    """
    Continue a decoder-only LM forward pass from a saved hidden-state tensor.

    Hugging Face causal LM hidden-state tuples use index 0 for token embeddings
    and index N for the output after decoder block N - 1. To resume from index K,
    this injects the saved activation at decoder block K's input. For K == N,
    the activation is already at the final hidden-state position and only the
    final norm/lm_head path is applied.
    """
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)
    if hidden_states.dim() != 3:
        raise ValueError(
            f'Expected hidden states with shape (seq, hidden) or (batch, seq, hidden), '
            f'got {tuple(hidden_states.shape)}.'
        )

    blocks = _decoder_blocks(model)
    total_layers = len(blocks)
    if layer_idx < 0:
        layer_idx = total_layers + layer_idx + 1
    if layer_idx < 0 or layer_idx > total_layers:
        raise ValueError(f'layer_idx must resolve to [0, {total_layers}], got {layer_idx}.')

    device = model.device  # type: ignore
    model_dtype = getattr(model, 'dtype', hidden_states.dtype)
    hidden_states = hidden_states.to(device=device, dtype=model_dtype)
    batch_size, seq_len, _ = hidden_states.shape

    if input_ids is not None:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if tuple(input_ids.shape) != (batch_size, seq_len):
            raise ValueError(
                f'input_ids shape {tuple(input_ids.shape)} does not match hidden-state '
                f'batch/sequence shape {(batch_size, seq_len)}.'
            )
        placeholder_ids = input_ids.to(device)
    else:
        placeholder_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    with torch.no_grad():
        if layer_idx == total_layers:
            output_embeddings = model.get_output_embeddings()
            if hasattr(output_embeddings, 'weight'):
                hidden_states = hidden_states.to(
                    device=output_embeddings.weight.device,
                    dtype=output_embeddings.weight.dtype,
                )
            final_norm = None
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):  # type: ignore[attr-defined]
                final_norm = model.transformer.ln_f  # type: ignore[attr-defined]
            elif hasattr(model, 'model') and hasattr(model.model, 'norm'):  # type: ignore[attr-defined]
                final_norm = model.model.norm  # type: ignore[attr-defined]
            if final_norm is not None:
                hidden_states = final_norm(hidden_states)
            return output_embeddings(hidden_states)  # type: ignore[misc]

        if layer_idx == 0:
            outputs: CausalLMOutputWithPast = model(
                inputs_embeds=hidden_states,
                output_hidden_states=False,
                use_cache=False,
            )  # type: ignore
            return outputs.logits  # type: ignore[return-value]

        def replace_hidden(_module, inputs):
            incoming_hidden_states = inputs[0]
            replacement = hidden_states.to(
                device=incoming_hidden_states.device,
                dtype=incoming_hidden_states.dtype,
            )
            return (replacement,) + tuple(inputs[1:])

        handle = blocks[layer_idx].register_forward_pre_hook(replace_hidden)
        try:
            outputs: CausalLMOutputWithPast = model(
                input_ids=placeholder_ids,
                output_hidden_states=False,
                use_cache=False,
            )  # type: ignore
        finally:
            handle.remove()

    return outputs.logits  # type: ignore[return-value]


def hidden_states_from_embeddings(
    embeddings: torch.Tensor,
    model: PreTrainedModel,
    layer_idx: int,
    require_grad: bool = False
) -> torch.Tensor:
    cond = len(embeddings.shape) == 2
    if cond:
        embeddings = embeddings.unsqueeze(0)
    ctx = torch.enable_grad() if require_grad else torch.no_grad()
    with ctx:
        outputs: CausalLMOutputWithPast = model(
            inputs_embeds=embeddings,
            output_hidden_states=True,
            use_cache=False
        ) # type: ignore
        hidden_states = outputs.hidden_states[layer_idx] # type: ignore
        hidden_states = (
            hidden_states if not cond else 
            hidden_states[0]
        )
    return hidden_states if require_grad else hidden_states.detach()


def hidden_states_from_input_ids_iterative(
    input_ids: torch.Tensor,
    model: PreTrainedModel,
    layer_idx: int,
):    
    if len(input_ids.shape) > 1:
        input_ids = input_ids[0]
    with torch.no_grad():
        target_input_embeddings = model.get_input_embeddings().weight[input_ids] # type: ignore
        target_input_embeddings = target_input_embeddings.detach()
        target_embeddings = [
            hidden_states_from_embeddings(
                embeddings=target_input_embeddings[:token_idx, :],
                model=model, 
                layer_idx=layer_idx,
                require_grad=False
            )[-1]
            for token_idx in range(1, target_input_embeddings.size(0) + 1)
        ]
    return torch.stack(target_embeddings)


def continuous_grad_and_discrete_verify(
    model: PreTrainedModel,
    layer_idx: int,
    input_embeddings: tuple[torch.Tensor, torch.LongTensor],
    target_hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.FloatTensor, torch.Tensor]:
    device = model.device # type: ignore

    continuous_embeddings: torch.Tensor = input_embeddings[0].to(device)
    discrete_tokens: torch.LongTensor = input_embeddings[1].to(device) # type: ignore
    target_hidden_states = target_hidden_states.to(device)

    fixed_embs = continuous_embeddings.clone().detach()
    last_emb = fixed_embs[-1:, :].clone().requires_grad_(True)

    inputs_embeds_cont = torch.cat([fixed_embs[:-1, :], last_emb], dim=0)
    cont_hidden_states = hidden_states_from_embeddings(
        embeddings=inputs_embeds_cont,
        model=model,
        layer_idx=layer_idx,
        require_grad=True
    )[-1]

    disc_hidden_states = hidden_states_from_input_ids(
        input_ids=discrete_tokens,
        model=model,
        layer_idx=layer_idx,
        require_grad=False
    )[-1]

    loss_cont = torch.nn.functional.mse_loss(
        cont_hidden_states.float(),
        target_hidden_states.float(),
        reduction='mean'
    )
    loss_disc = torch.nn.functional.mse_loss(
        disc_hidden_states.float(),
        target_hidden_states.float(),
        reduction='mean'
    )
    loss_cont.backward()

    return (
        last_emb.grad.squeeze(0), # type: ignore
        loss_disc, # type: ignore
        disc_hidden_states
    )


def continuous_grad_matrix(
    model: PreTrainedModel,
    layer_idx: int,
    continuous_embeddings: torch.Tensor,
    target_hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.FloatTensor, torch.Tensor]:
    device = model.device # type: ignore

    continuous_embeddings = continuous_embeddings.clone().detach().to(device).requires_grad_(True)
    target_hidden_states = target_hidden_states.to(device)

    cont_hidden_states = hidden_states_from_embeddings(
        embeddings=continuous_embeddings,
        model=model,
        layer_idx=layer_idx,
        require_grad=True
    )

    loss_cont = torch.nn.functional.mse_loss(
        cont_hidden_states.float(),
        target_hidden_states.float(),
        reduction='mean'
    )
    loss_cont.backward()

    return (
        continuous_embeddings.grad, # type: ignore
        loss_cont, # type: ignore
        cont_hidden_states
    )


def setup(
    model_id: str,
    precision: Literal[4, 8, 16, 32],
    layer_idx: int,
    print_stats: bool = True
) -> tuple[PreTrainedModel, PreTrainedTokenizer, str, int]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bnb_4bit_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32
    )

    bnb_8bit_config = BitsAndBytesConfig(
        load_in_8bit=True,          
        llm_int8_threshold=6.0,  
        llm_int8_has_fp16_weight=False,
    )

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)
    model_kwargs: dict[str, Any] = dict(
        pretrained_model_name_or_path=model_id, 
        device_map='auto', 
    )
    if precision == 4:
        model_kwargs['quantization_config'] = bnb_4bit_config
        model_kwargs['torch_dtype'] = torch.float32
    elif precision == 8:
        model_kwargs['quantization_config'] = bnb_8bit_config
        model_kwargs['torch_dtype'] = torch.float32
    elif precision == 16:
        model_kwargs['torch_dtype'] = torch.float16
    else:
        model_kwargs['torch_dtype'] = torch.float32

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    size_in_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) # type: ignore
    size_in_gb = size_in_bytes / ((2 ** 10) ** 3)

    if print_stats:
        print(f'Total parameter memory: {size_in_bytes:,} bytes')
        print(f'                        {size_in_gb:.2f} GB')

    total_layers = model.config.num_hidden_layers
    if layer_idx < 0:
        layer_idx = total_layers + layer_idx + 1

    replace_last_norm(model_id, model)

    model.eval() # type: ignore
    model.requires_grad_(False) # type: ignore
    tokenizer.pad_token = tokenizer.eos_token # type: ignore

    torch.set_grad_enabled(True)

    return model, tokenizer, device, layer_idx
