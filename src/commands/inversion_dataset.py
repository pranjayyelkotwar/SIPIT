from pathlib import Path

import pandas as pd

from src.algorithm import *
from src.datasets.dataset import DatasetCollection
from src.utils.model import setup
from src.utils.parser import DatasetInversionParser


def build_parser():
    return DatasetInversionParser()


def run(args) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)

    rank = args.rank
    pct = args.pct

    csv_name = (
        output_path.stem
        if rank < 0 or pct <= 0
        else f'{output_path.stem}-{rank}-{pct:.2f}'
    )
    output_dir = output_path.parent
    output_file = output_dir / f'{csv_name}.csv'
    output_dir.mkdir(parents=True, exist_ok=True)

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
        projection_iters_base=args.projection_iters_base,
        vocab_scale_factor=args.vocab_scale_factor
    )

    datasets = DatasetCollection.load(input_path)

    if rank >= 0 and pct > 0:
        total = len(datasets.datasets[0])
        i1 = int(pct * total * rank)
        i2 = int(pct * total * (rank + 1))
        print(f'Processing indices [{i1}, {i2}]')

    write_header = True
    results: list[dict] = []

    for dataset_idx, dataset in enumerate(datasets):
        for sample_idx, token_ids in enumerate(dataset):
            if rank >= 0 and pct > 0:
                lo = int(pct * len(dataset) * rank)
                hi = int(pct * len(dataset) * (rank + 1))
                if sample_idx < lo or sample_idx >= hi:
                    continue

            print(
                f'Dataset: {dataset_idx + 1}, Sample: {sample_idx + 1}, Length: {len(token_ids)}' # type: ignore
            )
            print(tokenizer.decode(token_ids))
            match, time_taken, timesteps, times = inversion_algorithm.inversion_attack(
                input_ids=token_ids.to(device), # type: ignore
                model=model,
                tokenizer=tokenizer,
                layer_idx=layer_idx,
                step_size=args.step_size,
                seed=args.seed,
            )

            row = {
                'dataset': datasets.dataset_names[dataset_idx],
                'index': sample_idx,
                'layer': args.layer_idx,
                'step_size': args.step_size,
                'token_length': len(token_ids), # type: ignore
                'match': match,
                'inversion_time': time_taken if match else -1,
                'timesteps': '_'.join(str(x) for x in timesteps) if match else '', # type: ignore
                'times': '_'.join(f'{x:.2f}' for x in times) if match else '', # type: ignore
            }
            results.append(row)

        # flush after each dataset
        partial_df = pd.DataFrame(results)
        partial_df.to_csv(
            output_file,
            mode='w' if write_header else 'a',
            header=write_header,
            index=False,
        )
        results = []
        write_header = False

    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
