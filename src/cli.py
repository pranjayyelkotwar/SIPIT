import argparse
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dotenv import load_dotenv
load_dotenv()

import transformers
transformers.logging.set_verbosity_error()

from src.commands import (
    capture_hidden_state,
    create_dataset_collection,
    create_dataset_random,
    generate_from_hidden_state,
    invert_hidden_state,
    invert_hidden_state_approx,
    inversion_dataset,
    inversion_single
)

COMMANDS = {
    'capture-hidden-state': capture_hidden_state,
    'create-dataset-collection': create_dataset_collection,
    'create-dataset-random': create_dataset_random,
    'generate-from-hidden-state': generate_from_hidden_state,
    'invert-hidden-state': invert_hidden_state,
    'invert-hidden-state-approx': invert_hidden_state_approx,
    'invert-dataset': inversion_dataset,
    'invert-single': inversion_single
}

def main(argv=None) -> int:
    root = argparse.ArgumentParser(prog='sipit')
    root.add_argument('--command', choices=COMMANDS.keys())
    ns, rest = root.parse_known_args(argv)

    cmd = COMMANDS[ns.command]

    parser = cmd.build_parser()
    args = parser.parse_args(rest)

    return cmd.run(args)

if __name__ == '__main__':
    raise SystemExit(main())
