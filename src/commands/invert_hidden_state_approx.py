from __future__ import annotations

from pathlib import Path

from src.algorithm import CosineRMSESIPIT
from src.commands.invert_hidden_state import _load_tensor_payload
from src.utils.model import setup
from src.utils.parser import ApproximateHiddenStateInversionParser
from src.utils.utils import set_seed


def build_parser():
    return ApproximateHiddenStateInversionParser()


def run(args) -> int:
    if args.method != 'SIPIT':
        raise ValueError('Approximate cosine/RMSE inversion currently supports only SIPIT.')
    if args.skip_target_tokens < 0:
        raise ValueError('skip-target-tokens must be non-negative.')

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
    if hidden_states.dim() == 1:
        hidden_states = hidden_states.unsqueeze(0)
    if args.skip_target_tokens >= hidden_states.size(0):
        raise ValueError(
            'skip-target-tokens must be smaller than the number of target tokens.'
        )
    hidden_states = hidden_states[args.skip_target_tokens:]

    algorithm = CosineRMSESIPIT(
        log_dir=args.log_file_path,
        log_name=args.log_file_name,
        special_start_token_id=args.special_start_token,
        use_scheduler=args.scheduler,
        projection_iters_base=args.projection_iters_base,
        vocab_scale_factor=args.vocab_scale_factor,
        min_cosine_similarity=args.min_cosine_similarity,
        max_rmse=args.max_rmse,
    )

    inversion_time, discovered_ids, timesteps, times = algorithm.find_prompt(
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        target_hidden_states=hidden_states,
        step_size=args.step_size,
    )
    if any(x is None for x in (inversion_time, discovered_ids, timesteps, times)):
        print('Approximate inversion failed!')
        return 1

    prompt_token_ids = discovered_ids
    if args.special_start_token is not None and prompt_token_ids:
        prompt_token_ids = prompt_token_ids[1:]
    prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
    output_path.write_text(prompt_text, encoding='utf-8')

    print(f'Recovered prompt written to {output_path}')
    print(
        'Acceptance criteria: '
        f'cosine >= {args.min_cosine_similarity}, RMSE <= {args.max_rmse}'
    )
    if isinstance(payload, dict) and 'prompt' in payload:
        print(f'Original prompt: {payload["prompt"]}')
    return 0


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
