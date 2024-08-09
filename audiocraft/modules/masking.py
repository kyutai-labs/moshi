

from abc import ABC, abstractmethod

import torch
from torch import nn


class TemporalMask(ABC, nn.Module):
    """Abstract class for masks that operate on the temporal dimension of a sequence.

    Attributes:
        dim: Dimension of the input.
        init_std: Standard deviation of the mask embedding at initialization.
        conv_layout: If True, the input is assumed to be in the format (B, C, T) instead of (B, T, C).
        return_mask: Whether to return the mask along with the masked input (needs to be disabled to add Temporal mask
            into a Sequential).
        no_masking_rate: Probability of not applying any masking at all.
        device: Device to use for the mask embedding.
        dtype: Data type to use for the mask embedding.
    """

    def __init__(self,
                 dim: int,
                 init_std: float = 1.,
                 conv_layout: bool = False,
                 no_masking_rate: float = 0.,
                 return_mask: bool = True,
                 device=None,
                 dtype=None):
        super().__init__()
        self.conv_layout = conv_layout
        self.no_masking_rate = no_masking_rate
        masked_emb_shape = (1, dim, 1) if conv_layout else (1, 1, dim)
        if init_std == 0.0:
            self.masked_emb = None
        else:
            self.masked_emb = nn.Parameter(
                init_std * torch.randn(masked_emb_shape, requires_grad=True, device=device, dtype=dtype))
        self.return_mask = return_mask

    @abstractmethod
    def _create_mask(self, x):
        """Creates a binary mask with the same shape as x, where ones represent unmasked positions and zeros
        represent positions replaced by a learnable mask embedding.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Mask.
        """

    def forward(self, x):
        B = x.shape[0]
        if not self.training:
            if self.return_mask:
                return x, torch.ones_like(x)
            else:
                return x
        mask = self._create_mask(x)
        if self.no_masking_rate > 0:
            # from time to time we apply no masking at all for an entire sequence.
            mask += (torch.rand(B, 1, 1, device=x.device) <= self.no_masking_rate).float()
        mask.clamp_(0, 1)
        masked_emb = 0. if self.masked_emb is None else self.masked_emb
        self._last_mask = mask
        if self.return_mask:
            return mask * x + (1. - mask) * masked_emb, mask
        else:
            return mask * x + (1. - mask) * masked_emb


class GilbertMask(TemporalMask):
    """Gilbert-Elliott span masking.
    This mask is parametrized by its average masking rate and expected burst length (average length of a masked span).

    Attributes:
        masking_rate (float): Average masking rate.
        mask_length (float): Expected mask length.
        **kwargs: Additional arguments to pass to the TemporalMask class.
    Returns:
        torch.Tensor: Masked input.
     """

    def __init__(self,
                 masking_rate: float,
                 mask_length: float,
                 **kwargs):
        super().__init__(**kwargs)
        self.masking_rate = masking_rate
        self.mask_length = mask_length

    def _create_mask(self, x):
        one_to_zero_prob = self.masking_rate / (self.mask_length * (1. - self.masking_rate))
        zero_to_one_prob = 1. / self.mask_length
        B = x.shape[0]
        T = x.shape[2] if self.conv_layout else x.shape[1]
        curr_state = (torch.rand((B,), device=x.device) > self.masking_rate).float()
        all_states = [curr_state]
        for _ in range(T - 1):
            next_state_from_one = (torch.rand((B,), device=x.device) <= (1 - one_to_zero_prob)).float()
            next_state_from_zero = (torch.rand((B,), device=x.device) <= zero_to_one_prob).float()
            next_state = torch.where(curr_state.bool(), next_state_from_one, next_state_from_zero)
            curr_state = next_state
            all_states.append(curr_state)
        mask = torch.stack(all_states, dim=-1)
        channel_axis = 1 if self.conv_layout else 2
        return torch.unsqueeze(mask, dim=channel_axis)


class ConstantSpanMask(TemporalMask):
    """Constant span masking.
    This mask is parametrized by the per-timestep probability of starting a new mask, and the fixed length of each mask.
    Starting a new mask while the previous one is still active creates overlapped masks, resulting in a longer span.

    Unlike wav2vec2.0, hubert, w2vbert, etc. we do not parametrize the masks in terms of per-timestep probability
    of starting a new mask at the API level, but rather in terms of the average masking rate for consistency with
    GilbertMask.

    Attributes:
        mask_prob (float): Probability of starting a new mask at each timestep,
            equal to 1 - (1 - masking_rate) ** (1. / mask_length).
        mask_length: Constant length of each sampled mask (they can overlap).
        **kwargs: Additional arguments to pass to the TemporalMask class.
    """

    def __init__(self,
                 masking_rate: float,
                 mask_length: float,
                 **kwargs):
        super().__init__(**kwargs)
        self.mask_prob = 1. - (1. - masking_rate) ** (1. / mask_length)
        self.mask_length = int(mask_length)

    def _create_mask(self, x):
        B = x.shape[0]
        T = x.shape[2] if self.conv_layout else x.shape[1]
        mask = torch.ones((B, T), device=x.device)
        mask_to_apply = (torch.rand((B, T), device=x.device) < self.mask_prob).float()
        kernel = torch.ones((1, 1, self.mask_length), device=x.device)
        mask_to_apply = torch.nn.functional.conv1d(torch.unsqueeze(
            mask_to_apply, dim=1), kernel, padding=self.mask_length - 1).view(B, -1)
        mask = (mask_to_apply <= 0).float()
        if self.mask_length - 1 > 0:
            mask = mask[:, :-(self.mask_length - 1)]
        channel_axis = 1 if self.conv_layout else 2
        return torch.unsqueeze(mask, dim=channel_axis)
