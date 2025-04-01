import math
import typing as tp
import torch
from torch import nn

from ..modules.transformer import create_norm_fn


def _delay_sequence(delays: tp.List[int], tensor: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    B, K, T = tensor.shape
    assert len(delays) == K, (len(delays), K)
    outs = []

    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(delay, dims=1)
        if delay > 0:
            line[:, :delay] = padding[:, k]
        outs.append(line)
    return torch.stack(outs, dim=1)


def _undelay_sequence(delays: tp.List[int], tensor: torch.Tensor,
                      fill_value: tp.Union[int, float] = float('NaN')) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    B, K, T, *_ = tensor.shape
    assert len(delays) == K
    mask = torch.ones(B, K, T, dtype=torch.bool, device=tensor.device)
    outs = []
    if all([delay == 0 for delay in delays]):
        return tensor, mask
    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(-delay, dims=1)
        if delay > 0:
            line[:, -delay:] = fill_value
            mask[:, k, -delay:] = 0
        outs.append(line)
    return torch.stack(outs, dim=1), mask


def _get_init_fn(input_dim: int) -> tp.Callable[[torch.Tensor], None]:
    def _init(x: torch.Tensor) -> None:
        std = 1 / math.sqrt(input_dim)
        x_orig = x
        if x.device.type == 'cpu' and x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()

        torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if x_orig is not x:
            x_orig.data[:] = x.to(x_orig)
    return _init


def _init_layer(m: nn.Module,
                zero_bias_init: bool = True):
    if isinstance(m, nn.Linear):
        init_fn = _get_init_fn(m.in_features)
        init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = _get_init_fn(m.embedding_dim)
        init_fn(m.weight)


class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).

    Args:
        norm (bool): if True, uses a layer norm after the embedding.
        zero_idx (int): special value indicating that the output should be exactly 0.
        low_rank (int | None): if provided, uses low rank embedding with a linear layer to reach
            the desired dimension. Quite efficient for reducing the number of weights for very large vocabs.
        lr (float or None): learning rate to use, only valid if the `make_optim_group()` method is used.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 *args, norm: bool = False, zero_idx: int = -1,
                 low_rank: int | None = None, lr: float | None = None, **kwargs):
        super().__init__(num_embeddings, low_rank or embedding_dim, *args, **kwargs)
        self.norm = None
        if norm:
            self.norm = create_norm_fn("layer_norm", self.embedding_dim)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx
        self.lr = lr
        self.low_rank = None
        if low_rank is not None:
            self.low_rank = nn.Linear(low_rank, embedding_dim, bias=False)

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        if self.norm is not None:
            y = self.norm(y)
        y = torch.where(is_zero[..., None], zero, y)
        if self.low_rank is not None:
            y = self.low_rank(y)
        return y

    def make_optim_group(self) -> dict:
        group: dict[str, tp.Any] = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group
