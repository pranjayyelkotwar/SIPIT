import argparse
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dotenv import load_dotenv
load_dotenv()

import transformers
transformers.logging.set_verbosity_error()

from src.commands import (
    create_dataset_collection,
    create_dataset_random,
    inversion_dataset,
    inversion_single
)

COMMANDS = {
    'create-dataset-collection': create_dataset_collection,
    'create-dataset-random': create_dataset_random,
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
