from time import time
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.algorithm.base import InversionAlgorithm
from src.utils.model import hidden_states_from_input_ids
from src.utils.utils import format_time_minutes, format_token


class BruteForce(InversionAlgorithm):
    """
    BruteForce prompt inversion algorithm based on brute-force token search.

    This implementation reconstructs each token by iterating over the entire
    vocabulary at random and selecting the token whose hidden-state representation
    exactly matches the target.
    """
    def __init__(
        self,
        log_dir: str,
        log_name: str | None,
        special_start_token_id: Optional[int] = None,
        logger_calls: dict[str, str] = {
            'after_backprop': '\r{}[{:5d}/{:5d}]: Token: {:15s} - Time: {}'
        },
        iterative_target: bool = True,
        **kwargs
    ):
        super().__init__(
            log_dir=log_dir, 
            log_name=log_name, 
            special_start_token_id=special_start_token_id, 
            logger_calls=logger_calls, 
            iterative_target=iterative_target,
            **kwargs
        )

    def find_token(
        self,
        *,
        token_idx: int,
        embedding_matrix: torch.Tensor,
        discovered_ids: list[int],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        layer_idx: int, 
        target_hidden_states: torch.Tensor,
        **kwargs
    ):
        """
        Recover the next token via randomized brute-force search over the vocabulary.

        This method samples a random permutation of the vocabulary, evaluates each
        candidate token by computing the hidden states of the sequence
        `discovered_ids + [candidate]`, and returns the first candidate whose hidden
        state matches the target at the current position.

        Args:
            token_idx: Index of the token currently being recovered.
            embedding_matrix: Model input embedding matrix (used to infer vocab size/device).
            discovered_ids: Token IDs recovered so far (prefix).
            model: Language model under attack.
            tokenizer: Tokenizer associated with the model.
            layer_idx: Target layer index for hidden-state matching.
            target_hidden_states: Target hidden representations for the full sequence.

        Returns:
            A tuple (token_id, token_embedding, final_timestep) where:
                - token_id is the recovered discrete token ID,
                - token_embedding is its embedding vector,
                - final_timestep is the number of candidates evaluated until a match.
        """
        # Test tokens in randomized order to avoid bias
        perm = torch.randperm(embedding_matrix.size(0))

        # Logger Info
        initial_desc = f'Token [{token_idx + 1:2d}/{target_hidden_states.size(0):2d}]'
        final_timestep = embedding_matrix.size(0)
        start_time = time()
        
        for idx, token_id in enumerate(perm):
            discrete_tokens: torch.LongTensor = torch.tensor(
                discovered_ids + [token_id],
                dtype=torch.long,
                device=embedding_matrix.device
            ).unsqueeze(0)  # type: ignore

            predicted_hidden_states = hidden_states_from_input_ids(
                input_ids=discrete_tokens, 
                model=model, 
                layer_idx=layer_idx, 
                require_grad=False
            )
            
            curr_token = tokenizer.decode([token_id], skip_special_tokens=True) # type: ignore
            
            self.logger.after_backprop(
                initial_desc, 
                idx + 1, 
                embedding_matrix.size(0),
                format_token(curr_token, length=15), 
                format_time_minutes(time() - start_time)
            )

            if self.is_match(predicted_hidden_states, target_hidden_states[token_idx]):
                final_timestep = idx + 1
                break

        self.logger.new_line()

        correct_token_id = perm[final_timestep - 1]
        correct_embedding = embedding_matrix[correct_token_id].clone().detach()
        return correct_token_id, correct_embedding, final_timestep