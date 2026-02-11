import gc
from functools import partial
from pathlib import Path
from time import time
from typing import Optional

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils.logger import Logging
from src.utils.model import (
    hidden_states_from_input_ids,
    hidden_states_from_input_ids_iterative
)
from src.utils.utils import set_seed


class InversionAlgorithm:
    """
    Base class for prompt inversion algorithms based on continuous
    embedding optimization and discrete token recovery.

    This class implements the common infrastructure for reconstructing
    input token sequences from internal hidden representations of a
    causal language model. It supports both iterative and direct hidden
    state extraction and provides logging, seeding, and recovery logic.

    Subclasses may override internal routines (e.g. `find_token`) to customize the inversion strategy.
    """
    def __init__(
        self,
        log_dir: str,
        log_name: str | None,
        special_start_token_id: Optional[int] = None,
        logger_calls: dict[str, str] = {
            'after_backprop': (
                '\r{}[{:5d}/{:5d}]: ' +
                'Loss: {:.2e} - Gradient norm: {:.2e} - ' +
                'Token: {:15s} - Emb Norm: {:.2e} - Time: {}'
            )
        },
        iterative_target: bool = True,
        use_scheduler: bool = False,
        **kwargs
    ):
        self.logger = Logging(
            log_path=Path(log_dir), 
            log_name=log_name,
            defined=logger_calls,
            write_to_file=log_name is not None
        )
        self.special_start_token_id = special_start_token_id
        self.target_extraction_fn = (
            hidden_states_from_input_ids_iterative if iterative_target else
            partial(hidden_states_from_input_ids, require_grad=False)
        )
        self.use_scheduler = use_scheduler
    
    def setup_optimzer_scheduler(
        self, 
        embeddings: torch.Tensor, 
        step_size: float,
        total_iters: int | None = None,
        end_factor: float | None = None
    ):
        self.optimizer = SGD([embeddings], lr=step_size)

        self.scheduler = None
        if (
            self.use_scheduler and 
            total_iters is not None and 
            end_factor is not None
        ):
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=total_iters
            )
        return self.optimizer, self.scheduler
    
    def step(self, loss: torch.Tensor):
        self.optimizer.step(lambda : loss)
        if self.scheduler is not None:
            self.scheduler.step()
    
    def is_match(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        rtol: float = 1e-5, 
        atol: float = 1e-5
    ):
        return torch.allclose(
            x, y,
            rtol=rtol,
            atol=atol
        )

    def find_token(self, *args, **kwargs):
        raise NotImplementedError

    def find_prompt(
        self,
        *,
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        layer_idx: int, 
        target_hidden_states: torch.Tensor,
        step_size: float,
        **kwargs
    ):
        """
        Reconstruct a full token sequence from target hidden states.

        Although the token-level optimization strategy differs between
        algorithms (e.g. SIPIT vs. BruteForce, or the non-iterative nature of HardPrompts), 
        the outer prompt reconstruction logic is identical and 
        therefore implemented in this base class.

        The method proceeds left-to-right, optionally prepending a special
        start token, and records per-token optimization statistics.

        Args:
            model: Language model used for inversion.
            tokenizer: Tokenizer associated with the model.
            layer_idx: Index of the layer whose hidden states are targeted.
            target_hidden_states: Target hidden representations to invert.
                Shape: (seq_len, hidden_dim) or (hidden_dim,).
            step_size: Learning rate for continuous embedding optimization.

        Returns:
            A tuple containing:
                - Total inversion time in seconds
                - List of recovered token IDs
                - List of optimization step counts per token
                - List of per-token runtimes in seconds

            Returns (None, None, None, None) if any token inversion fails.
        """
        embedding_matrix = model.get_input_embeddings().weight

        if target_hidden_states.dim() == 1:
            target_hidden_states = target_hidden_states.unsqueeze(0)

        discovered_embeddings: list[torch.Tensor] = []
        discovered_ids: list[int] = []
        timesteps: list[int] = []
        times: list[float] = []

        if self.special_start_token_id is not None:
            discovered_ids.append(self.special_start_token_id)
            discovered_embeddings.append(
                embedding_matrix[self.special_start_token_id] # type: ignore
                .clone()
                .detach()
                .requires_grad_(False)
            )

        start_time = time()
        for token_idx in range(target_hidden_states.size(0)):
            token_start_time = time()

            next_token_id, next_token_embedding, final_timestep = self.find_token(
                token_idx=token_idx, 
                embedding_matrix=embedding_matrix, 
                discovered_embeddings=discovered_embeddings, 
                discovered_ids=discovered_ids, 
                model=model, 
                tokenizer=tokenizer, 
                layer_idx=layer_idx, 
                target_hidden_states=target_hidden_states, 
                step_size=step_size 
            )

            token_end_time = time()

            if (
                next_token_id is None or 
                next_token_embedding is None or 
                final_timestep is None
            ):
                return None, None, None, None

            discovered_embeddings.append(next_token_embedding)
            discovered_ids.append(next_token_id)
            timesteps.append(final_timestep)
            times.append(token_end_time - token_start_time)

            gc.collect()
            torch.cuda.empty_cache()

        end_time = time()

        return end_time - start_time, discovered_ids, timesteps, times

    def inversion_attack(
        self,
        *,
        input_ids: torch.Tensor, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        layer_idx: int,
        step_size: float,
        seed: int = 8, 
        **kwargs
    ):
        """
        Perform a complete inversion attack on a given input sequence.

        This method orchestrates the full inversion pipeline. It first
        extracts target hidden states from the specified model layer,
        then reconstructs the original token sequence using `find_prompt`,
        and finally evaluates whether the recovered sequence matches the input.

        Optionally, a special start token is prepended before extracting
        hidden states if `special_start_token_id` is defined (usefull for Llama-3.1-8B).

        The attack is executed deterministically with respect to the provided random seed.

        Args:
            input_ids: Input token IDs to invert.
                Shape: (1, sequence_length).
            model: Language model under attack.
            tokenizer: Tokenizer associated with the model.
            layer_idx: Index of the layer whose hidden states are targeted.
            step_size: Learning rate for continuous embedding optimization.
            seed: Random seed for reproducibility.

        Returns:
            A tuple containing:
                - match: Whether the reconstructed sequence exactly matches the original input.
                - inversion_time: Total runtime in seconds.
                - timesteps: List of optimization step counts per token.
                - times: List of per-token runtimes in seconds.

            Returns (False, None, None, None) if the inversion fails or diverges.
        """

        set_seed(seed)
        new_input_ids: torch.LongTensor = (
            torch.cat(
                [
                    torch.tensor(
                        [self.special_start_token_id], 
                        dtype=torch.long, 
                        device=input_ids.device
                    ), 
                    input_ids
                ],
                dim=0
            ) if self.special_start_token_id is not None else
            input_ids
        ) # type: ignore

        target_hidden_states = self.target_extraction_fn(new_input_ids, model, layer_idx)
        start_from = new_input_ids.size(0) - input_ids.size(0) # type: ignore

        invertion_time, discovered_ids, timesteps, times = self.find_prompt(
            model=model, 
            tokenizer=tokenizer, 
            layer_idx=layer_idx, 
            target_hidden_states=target_hidden_states[start_from:], 
            step_size=step_size,
            **kwargs
        )
        
        if (
            invertion_time is None or 
            discovered_ids is None or 
            timesteps is None or 
            times is None
        ):
            print('Inversion failed or diverged with the given parameters.')
            return False, None, None, None

        match = all([x == y for x, y in zip(input_ids, discovered_ids[start_from:])])
        return match, invertion_time, timesteps, times
