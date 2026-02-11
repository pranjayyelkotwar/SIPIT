import numpy as np
import torch

from src.algorithm import *
from src.utils.model import setup
from src.utils.parser import SingleInversionParser


def build_parser():
    return SingleInversionParser()

def run(args) -> int:
    model, tokenizer, device, layer_idx = setup(
        model_id=args.model_id,
        precision=args.precision,
        layer_idx=args.layer_idx,
        print_stats=True,
    )

    inversion_method_dict: dict[str, type[InversionAlgorithm]] = {
        'SIPIT': SIPIT,
        'BruteForce': BruteForce,
        'HardPrompts': HardPrompts
    }

    inversion_algorithm = inversion_method_dict[args.method](
        log_dir=args.log_file_path,
        log_name=args.log_file_name,
        special_start_token_id=args.special_start_token,
        use_scheduler=args.scheduler,
        projection_iters_base=args.projection_iters_base,
        vocab_scale_factor=args.vocab_scale_factor
    )

    enc = tokenizer(
        args.prompt,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_ids_list: list[int] = enc["input_ids"] # type: ignore

    match, time_taken, timesteps, times = inversion_algorithm.inversion_attack(
        input_ids=torch.tensor(input_ids_list, dtype=torch.long, device=device),
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        step_size=args.step_size,
        seed=args.seed,
    )

    if any(x is None for x in (match, time_taken, timesteps, times)):
        print("Inversion Failed!")
        return 1

    iters_mean, iters_std = float(np.mean(timesteps)), float(np.std(timesteps)) # type: ignore
    time_mean, time_std = float(np.mean(times)), float(np.std(times)) # type: ignore

    print(f'Exact Inversion   : {"yes" if match else "no"}')
    print(f"Time Taken        : {time_taken:.2f} seconds")
    print(f"Stats Iters/Token : {iters_mean:.2f} ± {iters_std:.2f}")
    print(f"Stats Time/Token  : {time_mean:.2f} ± {time_std:.2f}")
    return 0

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)

if __name__ == "__main__":
    raise SystemExit(main())
