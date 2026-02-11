from time import time
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.algorithm.base import InversionAlgorithm
from src.utils.model import continuous_grad_matrix
from src.utils.utils import format_time_minutes


class HardPrompts(InversionAlgorithm):
    """
    HardPrompts inversion algorithm based on joint continuous optimization
    of all prompt embeddings with periodic discrete projection.

    This method optimizes the full prompt embedding sequence simultaneously
    using gradient descent, periodically projecting the continuous solution
    back onto the nearest discrete token embeddings.

    A learning-rate scheduler and projection heuristics are used to stabilize
    optimization and prevent cycling.
    """
    def __init__(
        self,
        log_dir: str,
        log_name: str | None,
        special_start_token_id: Optional[int] = None,
        logger_calls: dict[str, str] = {
            'after_backprop': (
                '\r[{:7,d}/{:7,d}]: ' +
                'Loss: {:.2e} - Gradient norm: {:.2e} - ' +
                'Emb Norm: {:.2e} - LR: {:.2e} - Time: {}'
            )
        },
        iterative_target: bool = False,
        use_scheduler: bool = True,
        projection_iters_base: int = 50,
        vocab_scale_factor: int = 7_250,
        n_tokens_scale_factor: int = 4,
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
        
        # Scales total iteration budget with respect to sequence length
        self.n_tokens_scale_factor = n_tokens_scale_factor

    def find_prompt(
        self,
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        layer_idx: int, 
        target_hidden_states: torch.Tensor,
        step_size: float,
        end_factor: float = 1e-2,
        **kwargs
    ):
        """
        Recover a full prompt using joint continuous optimization.

        This method optimizes all token embeddings simultaneously to match
        the target hidden states. Optimization is performed in continuous
        embedding space and periodically projected onto the nearest discrete
        tokens.

        A scheduler controls the learning rate between projections, and
        previously visited token sequences are tracked to avoid cycling.

        Args:
            model: Language model used for inversion.
            tokenizer: Associated tokenizer.
            layer_idx: Target layer index.
            target_hidden_states: Target hidden representations.
            step_size: Initial learning rate.

        Returns:
            A tuple containing:
                - total_time: Total runtime in seconds.
                - token_ids: Recovered token sequence.
                - timesteps: List containing final iteration count.
                - times: List containing total runtime.

            Returns (None, None, None, None) if optimization diverges.
        """
        embedding_matrix: torch.Tensor = model.get_input_embeddings().weight # type: ignore

        if target_hidden_states.dim() == 1:
            target_hidden_states = target_hidden_states.unsqueeze(0)
        
        # Maximum optimization budget (scaled by prompt length)
        max_iters = (
            embedding_matrix.size(0) * 
            target_hidden_states.size(0) // 
            self.n_tokens_scale_factor 
        )
        final_timestep = max_iters
        start_time = time()

        # Determine projection frequency based on vocabulary size
        reset_extra = embedding_matrix.size(0) // self.vocab_scale_factor
        reset_every = self.projection_iters_base * (1 + reset_extra) * max_iters // 10_000

        # Initialize continuous embeddings from random discrete tokens
        copy_embedding_matrix = embedding_matrix.clone().detach().requires_grad_(False)
        token_ids: torch.LongTensor = torch.randint(
            0, embedding_matrix.size(0), 
            (target_hidden_states.size(0),),
            dtype=torch.long,
            device=model.device # type: ignore
        ) # type: ignore
        continuous_embeddings: torch.Tensor = copy_embedding_matrix[token_ids].clone().requires_grad_(True)
        
        # Track previously visited token sequences to prevent cycling
        visited_token_ids: set[tuple[int, ...]] = set(tuple(token_ids.tolist()))

        # Initialize optimizer and scheduler for the current projection window
        self.setup_optimzer_scheduler(
            embeddings=continuous_embeddings, 
            step_size=step_size,
            total_iters=reset_every,
            end_factor=end_factor
        )

        for idx in range(max_iters):
            # No need to do `optimizer.zero_grad()` 
            # since this function will not backpropagate to the local `continuous_embeddings` tensor.
            grad_oracle, loss, predicted_hidden_states = continuous_grad_matrix(
                model=model,
                layer_idx=layer_idx, 
                continuous_embeddings=continuous_embeddings,
                target_hidden_states=target_hidden_states
            )

            if torch.isnan(loss) or torch.isnan(grad_oracle).any():
                return None, None, None, None

            grad_norm = grad_oracle.norm().item()
            
            emb_norm = continuous_embeddings.norm(dim=-1).mean().item()
            self.logger.after_backprop(
                idx + 1, 
                max_iters,
                loss.item(), 
                grad_norm, 
                emb_norm,
                self.optimizer.param_groups[0]["lr"],
                format_time_minutes(time() - start_time)
            )

            if self.is_match(predicted_hidden_states, target_hidden_states):
                final_timestep = idx + 1
                break

            if grad_norm > 1.0:
                grad_oracle = grad_oracle / grad_norm

            continuous_embeddings.grad = grad_oracle

            self.step(loss)

            if (idx + 1) % reset_every == 0:
                distances = torch.cdist(
                    continuous_embeddings,
                    copy_embedding_matrix,
                    p=2
                )
                token_ids = distances.argmin(dim=1) # type: ignore
                tuple_token_ids: tuple[int, ...] = tuple(token_ids.tolist())
                mult = 0.8

                # Avoid revisiting identical token sequences
                if tuple_token_ids not in visited_token_ids:
                    continuous_embeddings.data = copy_embedding_matrix[token_ids].clone().data
                    visited_token_ids.add(tuple_token_ids)
                    mult = 0.9
                    self.logger.write('Projected!', end='\n')

                self.logger.write(tokenizer.decode(token_ids, skip_special_tokens=True), end='\n') # type: ignore
                
                # As we get closer to the correct embeddings, reduce the step sizes
                step_size *= mult
                end_factor *= mult
                self.setup_optimzer_scheduler(
                    embeddings=continuous_embeddings,
                    step_size=step_size,
                    total_iters=reset_every,
                    end_factor=1e-2
                )

        self.logger.new_line()

        total_time = time() - start_time
        distances = torch.cdist(
            continuous_embeddings,
            copy_embedding_matrix,
            p=2
        )
        token_ids = distances.argmin(dim=1) # type: ignore

        del copy_embedding_matrix
        return total_time, token_ids.tolist(), [final_timestep], [total_time]