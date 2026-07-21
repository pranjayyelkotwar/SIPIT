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
                "meta-llama/Llama-3.1-8B-Instruct",
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


class HiddenStateInversionParser(BaseInversionParser):
    def __init__(self, description='Arguments for inverting a saved hidden-state tensor.'):
        super().__init__(description=description)

        self.add_argument(
            '-i', '--input',
            type=str,
            required=True,
            help='Path to a prompt tensor file or capture bundle directory.'
        )
        self.add_argument(
            '-o', '--output',
            type=str,
            required=True,
            help='Path to the output text file that will contain the recovered prompt.'
        )


class ApproximateHiddenStateInversionParser(HiddenStateInversionParser):
    def __init__(self):
        super().__init__(
            description=(
                'Invert a perturbed hidden-state tensor using cosine similarity '
                'and RMSE acceptance criteria.'
            )
        )
        self.add_argument(
            '--min-cosine-similarity',
            type=float,
            default=0.78,
            help=(
                'Minimum per-token cosine similarity required for acceptance. '
                'Default 0.78 is rounded from the non-BOS LlamaScope SAE rows '
                'in misc/llama31_l22_token_deltas.csv.'
            ),
        )
        self.add_argument(
            '--max-rmse',
            type=float,
            default=0.22,
            help=(
                'Maximum per-token RMSE allowed for acceptance. Default 0.22 '
                'is rounded from the non-BOS LlamaScope SAE rows in the '
                'token-level baseline CSV.'
            ),
        )
        self.add_argument(
            '--skip-target-tokens',
            type=int,
            default=0,
            help=(
                'Number of leading target hidden-state rows to ignore. Use 1 '
                'for the captured Llama prompt bundle whose SAE-reconstructed '
                'BOS row is an outlier (default: 0).'
            ),
        )


class HiddenStateGenerationParser(ArgumentParser):
    def __init__(self, description='Arguments for generating from a saved hidden-state tensor.'):
        super().__init__(description=description)

        self.add_argument(
            '-i', '--input',
            type=str,
            required=True,
            help='Path to a prompt tensor file or capture bundle directory.'
        )
        self.add_argument(
            '-o', '--output',
            type=str,
            required=True,
            help='Path to the output text file that will contain generated text.'
        )
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
            help='Index of the saved activation layer. Use -1 for the last layer (default: -1).'
        )
        self.add_argument(
            '--seed',
            type=int,
            default=8,
            help='Random seed for reproducibility (default: 8).'
        )
        self.add_argument(
            '--max-new-tokens',
            type=int,
            default=64,
            help='Number of tokens to generate. The first token is generated from the activation.'
        )
        self.add_argument(
            '--do-sample',
            action='store_true',
            help='Sample tokens instead of using greedy argmax decoding.'
        )
        self.add_argument(
            '--temperature',
            type=float,
            default=1.0,
            help='Sampling temperature for activation-derived and continuation tokens.'
        )
        self.add_argument(
            '--top-k',
            type=int,
            default=0,
            help='If sampling, keep only the top-k logits. Use 0 to disable top-k filtering.'
        )
