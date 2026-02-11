from time import time
from typing import Optional

import torch
from torch.optim import SGD
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.algorithm.base import InversionAlgorithm
from src.utils.model import continuous_grad_and_discrete_verify
from src.utils.utils import format_time_minutes, format_token


class SIPIT(InversionAlgorithm):
    """
    SIPIT inversion algorithm based on continuous embedding optimization
    followed by discrete token projection.

    This implementation alternates between gradient-based optimization in
    embedding space and periodic projection onto the nearest discrete token
    embedding.

    The frequency of projection resets is scaled based on vocabulary size
    to maintain stable optimization behavior across different models.
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
        use_scheduler: bool = True,
        projection_iters_base: int = 50,
        vocab_scale_factor: int = 25_000,
        **kwargs
    ):
        super().__init__(
            log_dir=log_dir, 
            log_name=log_name, 
            special_start_token_id=special_start_token_id, 
            logger_calls=logger_calls, 
            iterative_target=iterative_target,
            use_scheduler=use_scheduler,
            **kwargs
        )
        
        # Control how frequently projections are performed
        self.projection_iters_base = projection_iters_base
        self.vocab_scale_factor = vocab_scale_factor

    def find_token(
        self,
        *,
        token_idx: int,
        embedding_matrix: torch.Tensor,
        discovered_embeddings: list[torch.Tensor], 
        discovered_ids: list[int],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        layer_idx: int, 
        target_hidden_states: torch.Tensor,
        step_size: float,
        **kwargs
    ):
        """
        Recover the next token using continuous optimization and periodic projection.

        This method optimizes a single continuous embedding variable intended to
        represent the next token in the sequence. Each iteration:
          1) Computes a gradient oracle and loss for the current candidate sequence.
          2) Takes an optimizer step on the continuous embedding.
          3) Periodically projects the embedding onto the nearest vocabulary row to
             update the discrete token id and "snap" the continuous state.

        Args:
            token_idx: Index of the token currently being recovered.
            embedding_matrix: Input embedding matrix of shape (V, d).
            discovered_embeddings: Embeddings for previously recovered tokens.
            discovered_ids: Token IDs for previously recovered tokens.
            model: Language model under attack.
            tokenizer: Tokenizer associated with the model.
            layer_idx: Target layer index for hidden-state matching.
            target_hidden_states: Target hidden representations for the full sequence.
                Shape: (T, d).
            step_size: Initial learning rate for optimizing the continuous embedding.

        Returns:
            (token_id, token_embedding, final_timestep) where:
              - token_id: recovered discrete token id (int)
              - token_embedding: corresponding embedding vector (Tensor[d])
              - final_timestep: number of iterations executed for this token

            Returns (None, None, None) if the optimization diverges.
        """
        # Determine projection frequency based on vocabulary size
        reset_extra = embedding_matrix.size(0) // self.vocab_scale_factor
        
        # The former is a bit Llama-specific, the latter performs well on smaller vocab models.
        reset_every = self.projection_iters_base * (1 + reset_extra * int(token_idx == 0))
        # reset_every = self.projection_iters_base * (1 + reset_extra)
        
        # Initialize continuous embedding from random discrete token
        copy_embedding_matrix = embedding_matrix.clone().detach().requires_grad_(False)
        token_id = int(torch.randint(0, embedding_matrix.size(0), (1,)).item())
        continuous_embedding = copy_embedding_matrix[token_id].clone().requires_grad_(True)
        
        # Initialize optimizer and scheduler for the current projection window
        self.setup_optimzer_scheduler(
            embeddings=continuous_embedding, 
            step_size=step_size,
            total_iters=reset_every,
            end_factor=1e-2
        )

        # Logger Info
        initial_desc = f'Token [{token_idx + 1:2d}/{target_hidden_states.size(0):2d}]'
        final_timestep = embedding_matrix.size(0)
        start_time = time()
        
        for idx in range(embedding_matrix.size(0)):        
            continuous_embeddings = torch.stack(
                discovered_embeddings + [continuous_embedding]
            )
            # ).unsqueeze(0) 
            discrete_tokens: torch.LongTensor = torch.tensor(
                discovered_ids + [token_id],
                dtype=torch.long
            )  # type: ignore
            # ).unsqueeze(0)  # type: ignore

            # No need to do `optimizer.zero_grad()` 
            # since this function will not backpropagate to the local `continuous_embeddings` tensor.
            grad_oracle, loss, predicted_hidden_states = continuous_grad_and_discrete_verify(
                model=model,
                layer_idx=layer_idx, 
                input_embeddings=(continuous_embeddings, discrete_tokens),
                target_hidden_states=target_hidden_states[token_idx]
            )

            if torch.isnan(loss) or torch.isnan(grad_oracle).any():
                return None, None, None

            grad_norm = grad_oracle.norm().item()
            curr_token = tokenizer.decode([token_id], skip_special_tokens=True) # type: ignore
            
            emb_norm = continuous_embedding.norm().item()
            self.logger.after_backprop(
                initial_desc, 
                idx + 1, 
                embedding_matrix.size(0),
                loss.item(), 
                grad_norm, 
                format_token(curr_token, length=15), 
                emb_norm,
                format_time_minutes(time() - start_time)
            )

            if self.is_match(predicted_hidden_states, target_hidden_states[token_idx]):
                final_timestep = idx + 1
                break

            if grad_norm > 1.0:
                grad_oracle = grad_oracle / grad_norm

            continuous_embedding.grad = grad_oracle

            self.step(loss)

            copy_embedding_matrix[token_id] = float('inf')
            distances = torch.norm(copy_embedding_matrix - continuous_embedding, dim=1)
            token_id = int(torch.argmin(distances))
            
            if (idx + 1) % reset_every == 0:
                continuous_embedding.data = copy_embedding_matrix[token_id].clone().data

        self.logger.new_line()

        # Do this instead of just returning `copy_embedding_matrix[token_id]`
        # in order to not keep the embedding matrix in memory when we only need one row.
        distances = torch.norm(copy_embedding_matrix - continuous_embedding, dim=1)
        token_id = int(torch.argmin(distances))
        correct_embedding = copy_embedding_matrix[token_id].clone().detach()
        del copy_embedding_matrix, continuous_embedding
        return token_id, correct_embedding, final_timestep