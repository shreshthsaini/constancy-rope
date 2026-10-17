"""Explicit numerical policy for a separate SEGA response diagnostic."""
import torch
from .sega_official import OfficialSEGA


class DeterministicAllocationSEGA(OfficialSEGA):
    """Use deterministic feature-to-gain arithmetic, restoring the caller policy.

    This changes the numerical reduction policy and is not silently substituted
    into the released baseline. The caller must record this policy explicitly.
    Intended for a single-threaded sampler process, since PyTorch policy is global.
    """

    @torch.no_grad()
    def update(self, projected_features, sigma):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
            return super().update(projected_features, sigma)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
