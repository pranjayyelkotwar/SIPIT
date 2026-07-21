import math

import torch
import torch.nn.functional as F

from src.algorithm.SIPIT import SIPIT


class CosineRMSESIPIT(SIPIT):
    """SIPIT with approximate discrete hidden-state acceptance criteria.

    A candidate is accepted only when its cosine similarity is at least the
    configured minimum and its RMSE is at most the configured maximum.
    """

    def __init__(
        self,
        *,
        min_cosine_similarity: float,
        max_rmse: float,
        **kwargs,
    ):
        if not -1.0 <= min_cosine_similarity <= 1.0:
            raise ValueError('min_cosine_similarity must be between -1 and 1.')
        if not math.isfinite(max_rmse) or max_rmse < 0:
            raise ValueError('max_rmse must be a finite, non-negative number.')

        super().__init__(**kwargs)
        self.min_cosine_similarity = min_cosine_similarity
        self.max_rmse = max_rmse

    def similarity_metrics(
        self,
        candidate: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[float, float]:
        candidate = candidate.detach().float().flatten()
        target = target.detach().to(candidate.device).float().flatten()
        cosine = F.cosine_similarity(candidate, target, dim=0).item()
        rmse = torch.sqrt(torch.mean((candidate - target).square())).item()
        return cosine, rmse

    def is_match(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> bool:
        cosine, rmse = self.similarity_metrics(x, y)
        return cosine >= self.min_cosine_similarity and rmse <= self.max_rmse
