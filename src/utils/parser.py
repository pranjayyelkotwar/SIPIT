from argparse import ArgumentParser


class BaseInversionParser(ArgumentParser):
    def __init__(self, description: str):
        super().__init__(description=description)

        self.add_argument(
            '--method',
            type=str,
            default='SIPIT',
            choices=['SIPIT', 'BruteForce', 'HardPrompts'],
            help='Inversion Algorithm. Choices: {`SIPIT`, `BruteForce`, `HardPrompts`}. Default: `SIPIT`.'
        )
        self.add_argument(
            '--log-file-path',
            type=str,
            default='logs',
            help='Directory where log files will be saved (default: logs).'
        )
        self.add_argument(
            '--log-file-name',
            type=str,
            default=None,
            help='Optional prefix for the log file name. If not provided, a log file will not be created.'
        )
        self.add_argument(
            '--seed',
            type=int,
            default=8,
            help='Random seed for reproducibility (default: 8).'
        )
        self.add_argument(
            '--model-id',
            type=str,
            default='openai-community/gpt2',
            help='Hugging Face model identifier to load (default: openai-community/gpt2).'
        )
        self.add_argument(
            '--step-size',
            type=float,
            default=1.0,
            help='Learning rate / step size used during optimization (default: 1.0).'
        )
        self.add_argument(
            '--layer-idx',
            type=int,
            default=-1,
            help='Index of the model layer to invert. Use -1 for the last layer (default: -1).'
        )
        self.add_argument(
            '--precision',
            type=int,
            default=32,
            choices=[4, 8, 16, 32],
            help='Bit precision for model weights. Choices: {4, 8, 16, 32}. Default: 32.'
        )
        self.add_argument(
            '--special-start-token',
            type=int,
            default=None,
            help='Optional tokenizer start token ID to prepend to prompts.'
        )
        self.add_argument(
            '--scheduler',
            action='store_true',
            help='Whether to use a step-size scheduler. Used when applicable.'
        )
        self.add_argument(
            '--projection-iters-base',
            type=int,
            default=50,
            help='Base number of steps between projections of continuous embedding(s). Used when applicable.'
        )
        self.add_argument(
            '--vocab-scale-factor',
            type=int,
            default=25_000,
            help=(
                'Vocabulary scale factor for steps between projections ' +
                'of continuous embedding(s). Used when applicable.'
            )
        )
        self.add_argument(
            '--n-tokens-scale-factor',
            type=int,
            default=4,
            help=(
                'Sequence length scale factor for maximum number ' +
                'of iteration. Used when applicable.'
            )
        )

class SingleInversionParser(BaseInversionParser):
    def __init__(self, description='Arguments for running SIPIT on a single prompt.'):
        super().__init__(description=description)

        self.add_argument(
            '-p', '--prompt',
            type=str,
            required=True,
            help='Input prompt text to be inverted.'
        )

class DatasetInversionParser(BaseInversionParser):
    def __init__(self, description='Arguments for running SIPIT on a dataset.'):
        super().__init__(description=description)

        self.add_argument(
            '-i', '--input',
            type=str,
            required=True,
            help=(
                'Path to the dataset directory. If it does not exist, '
                'a new dataset will be created and saved at this location.'
            )
        )
        self.add_argument(
            '-o', '--output',
            type=str,
            required=True,
            help='Path to the output csv file.'
        )
        self.add_argument(
            '-n', '--max-prompts',
            type=int,
            default=1000,
            help='Maximum number of prompts to process from the dataset (default: 1000).'
        )
        self.add_argument(
            '--step',
            type=int,
            default=20,
            help='Step size for increasing token length between experiments (default: 20).'
        )
        self.add_argument(
            '--total-lengths',
            type=int,
            default=10,
            help='Number of different prompt lengths to evaluate (default: 10).'
        )
        self.add_argument(
            '--rank',
            type=int,
            default=-1,
            help='Process rank for distributed execution. Use -1 for single-process mode (default: -1).'
        )
        self.add_argument(
            '--pct',
            type=float,
            default=0.0,
            help='Fraction of the dataset to process [0.0, 1.0]. Default: 0.0 (use all data).'
        )


class DatasetBaseParser(ArgumentParser):
    def __init__(self, description='Arguments for creating a SIPIT dataset.'):
        super().__init__(description=description)

        self.add_argument(
            '--model-names',
            type=str,
            nargs='+',
            default=[
                "openai-community/gpt2",
                "mistralai/Mistral-7B-v0.1",
                "meta-llama/Meta-Llama-3-8B",
            ],
            help=(
                'List of model identifiers to use. '
                'Provide one or more values separated by spaces.'
            )
        )
        self.add_argument(
            '-o', '--output',
            type=str,
            default='data/',
            help='Path to the dataset directory.'
        )
        self.add_argument(
            '--dataset-name',
            type=str,
            default='SIPIT-Dataset',
            help='Nameof dataset.'
        )
        self.add_argument(
            '--seed',
            type=int,
            default=8,
            help='Random seed for dataset generation (default: 8).'
        )
        self.add_argument(
            '-n-tokens', '--tokens',
            type=int,
            default=20,
            help='Number of tokens per generated prompt (default: 20).'
        )

class DatasetRandomParser(DatasetBaseParser):
    def __init__(self, description='Arguments for creating a SIPIT random dataset.'):
        super().__init__(description=description)

        self.add_argument(
            '-n', '--prompts',
            type=int,
            default=100,
            help='Number of prompts to generate in the dataset (default: 100).'
        )

class DatasetCollectionParser(DatasetBaseParser):
    def __init__(self, description='Arguments for creating a SIPIT collection dataset.'):
        super().__init__(description=description)

        self.add_argument(
            '--dataset-config',
            type=str,
            required=True,
            help='Path to a JSON file containing dataset configurations.'
        )

        self.add_argument(
            '--output-text-column',
            type=str,
            default='text',
            help='Name of the text column in the saved dataset on disk (default: text).'
        )
        self.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrite existing datasets if they already exist.'
        )


class HiddenStateCaptureParser(ArgumentParser):
    def __init__(self, description='Arguments for capturing hidden states from a model.'):
        super().__init__(description=description)

        self.add_argument(
            '--model-id',
            type=str,
            default='openai-community/gpt2',
            help='Hugging Face model identifier to load (default: openai-community/gpt2).'
        )
        self.add_argument(
            '--precision',
            type=int,
            default=32,
            choices=[4, 8, 16, 32],
            help='Bit precision for model weights. Choices: {4, 8, 16, 32}. Default: 32.'
        )
        self.add_argument(
            '--layer-idx',
            type=int,
            default=-1,
            help='Index of the model layer to capture. Use -1 for the last layer (default: -1).'
        )
        self.add_argument(
            '--seed',
            type=int,
            default=8,
            help='Random seed for reproducibility (default: 8).'
        )
        self.add_argument(
            '-o', '--output',
            type=str,
            required=True,
            help='Directory where the captured hidden-state bundle will be written.'
        )

        input_group = self.add_mutually_exclusive_group(required=True)
        input_group.add_argument(
            '-p', '--prompt',
            type=str,
            help='Prompt text to capture hidden states for.'
        )
        input_group.add_argument(
            '-i', '--input',
            type=str,
            help='Path to a saved dataset collection directory.'
        )