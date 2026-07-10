from __future__ import annotations

import json
from pathlib import Path

import torch

from src.utils.model import logits_from_hidden_states, setup
from src.utils.parser import HiddenStateGenerationParser
from src.utils.utils import set_seed


def build_parser():
    return HiddenStateGenerationParser()


def _normalize_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
        return hidden_states.squeeze(0)
    return hidden_states


def _load_tensor_payload(input_path: Path) -> tuple[torch.Tensor, dict]:
    if input_path.is_dir():
        manifest_path = input_path / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'Missing manifest.json in {input_path}')

        with manifest_path.open('r', encoding='utf-8') as f:
            manifest = json.load(f)

        artifacts = manifest.get('artifacts', [])
        if not artifacts:
            raise ValueError(f'No tensor artifacts recorded in {manifest_path}')

        input_path = input_path / str(artifacts[0]['file'])

    payload = torch.load(input_path, map_location='cpu')
    if isinstance(payload, dict) and 'hidden_states' in payload:
        return _normalize_hidden_states(payload['hidden_states']), payload
    if torch.is_tensor(payload):
        return _normalize_hidden_states(payload), {'file': str(input_path)}
    raise TypeError(f'Unsupported tensor payload in {input_path}')


def _select_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: int,
) -> torch.LongTensor:
    next_token_logits = logits[:, -1, :]
    if not do_sample:
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    if temperature <= 0:
        raise ValueError('--temperature must be positive when --do-sample is set.')

    next_token_logits = next_token_logits / temperature
    if top_k > 0:
        k = min(top_k, next_token_logits.size(-1))
        values, _indices = torch.topk(next_token_logits, k, dim=-1)
        threshold = values[:, -1].unsqueeze(-1)
        next_token_logits = next_token_logits.masked_fill(next_token_logits < threshold, -float('inf'))

    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _payload_input_ids(payload: dict) -> torch.LongTensor | None:
    input_ids = payload.get('input_ids') if isinstance(payload, dict) else None
    if input_ids is None:
        return None
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    return input_ids.detach().cpu().long()


def run(args) -> int:
    if args.max_new_tokens < 1:
        raise ValueError('--max-new-tokens must be at least 1.')

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    model, tokenizer, _device, layer_idx = setup(
        model_id=args.model_id,
        precision=args.precision,
        layer_idx=args.layer_idx,
        print_stats=True,
    )

    hidden_states, payload = _load_tensor_payload(input_path)
    input_ids = _payload_input_ids(payload)
    logits = logits_from_hidden_states(
        hidden_states=hidden_states,
        model=model,
        layer_idx=layer_idx,
        input_ids=input_ids,
    )
    first_token = _select_next_token(
        logits=logits,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated_ids = first_token.detach().cpu()
    if args.max_new_tokens > 1:
        if input_ids is None:
            raise ValueError(
                'Generating more than one token requires an artifact with input_ids. '
                'Use a capture-hidden-state payload or set --max-new-tokens 1.'
            )
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        continuation_input = torch.cat([input_ids.to(model.device), first_token.to(model.device)], dim=-1)  # type: ignore
        generate_kwargs = {
            'input_ids': continuation_input,
            'max_new_tokens': args.max_new_tokens - 1,
            'do_sample': args.do_sample,
            'pad_token_id': tokenizer.pad_token_id,
        }
        if args.do_sample:
            generate_kwargs['temperature'] = args.temperature
            if args.top_k > 0:
                generate_kwargs['top_k'] = args.top_k
        continuation = model.generate(**generate_kwargs)  # type: ignore
        generated_ids = continuation[:, input_ids.size(-1):].detach().cpu()

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    output_path.write_text(generated_text, encoding='utf-8')

    print(f'Generated text written to {output_path}')
    print(f'Generated token ids: {generated_ids[0].tolist()}')
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
