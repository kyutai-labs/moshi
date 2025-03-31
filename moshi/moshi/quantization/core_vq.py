# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import typing as tp

from einops import rearrange, repeat
import torch
from torch import nn
from torch import distributed
import torch.nn.functional as F


class _CodebookForwardResult(tp.NamedTuple):
    quantized: torch.Tensor
    codes: torch.Tensor
    metrics: tp.Dict[str, torch.Tensor]


class _VQForwardResult(tp.NamedTuple):
    quantized: torch.Tensor
    codes: torch.Tensor
    loss: torch.Tensor
    metrics: tp.Dict[str, torch.Tensor]


def _ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def _sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def _compute_entropy(usage: torch.Tensor) -> torch.Tensor:
    # Usage is some unnormalized distribution.
    proba = usage / usage.sum()
    p_log_p = torch.where(
        proba == 0, zero_scalar(usage.device), proba * torch.log(proba)
    )
    return -p_log_p.sum()


def _is_distributed() -> bool:
    # Checks if we need to use distributed routines.
    return distributed.is_initialized() and distributed.get_world_size() > 1


def _average_tensors(tensors: tp.Sequence[torch.Tensor]) -> None:
    if not _is_distributed():
        return
    world_size = distributed.get_world_size()
    handles = []
    for tensor in tensors:
        handle = distributed.all_reduce(
            tensor.data, op=distributed.ReduceOp.SUM, async_op=True)
        handles.append(handle)
    for tensor, handle in zip(tensors, handles):
        handle.wait()
        tensor.data /= world_size


def _run_kmeans(samples: torch.Tensor, num_clusters: int, num_iters: int = 50) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    # Kmeans algorithm used to initialize the codebooks.
    dim = samples.shape[-1]
    means = _sample_vectors(samples, num_clusters)
    bins = None

    for _ in range(num_iters):
        dists = torch.cdist(samples[None], means[None], p=2)[0]
        buckets = dists.argmin(dim=-1)
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins.clamp_(min=1)

        new_means = torch.zeros_like(means)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means /= bins[..., None]
        resampled = _sample_vectors(samples, num_clusters)
        means = torch.where(zero_mask[..., None], resampled, new_means)

    assert bins is not None
    return means, bins


def zero_scalar(device) -> torch.Tensor:
    """Returns a 0. value on the given device without introducing a synchronization point."""
    return torch.zeros([1], device=device)[0]


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.

    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_usage_ratio (float): Defines the threshold for the cluster usage under which a centroid
            is replaced. This is expressed as a fraction of the usage a centroid would get under
            a uniform distribution, so that it doesn't depend on the batch size etc.
        replaced_usage_ratio (float): When replacing a centroid, use this as an initial centroid usage,
            to avoid the centroid getting replaced too quickly.
        check_unused_every (int): Check for unused centroids every `check_unused_every` iterations.
            This is to avoid too many synchronization points.

    Buffers:
        cluster_usage (torch.Tensor): EMA of the cluster usage per batch, e.g. this will
            be dependent on the batch size etc.
        embedding_sum (torch.Tensor): EMA of the sum of the assigned points to each cluster.
            In particular, this can be normalized by `cluster_usage` to obtain the
            actual cluster centroids.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_usage_ratio: float = 0.1,
        replaced_usage_ratio: float = 1.0,
        check_unused_every: int = 5,
    ):
        super().__init__()
        self.decay = decay

        self.dim = dim
        self.codebook_size = codebook_size

        self.epsilon = epsilon
        self.threshold_usage_ratio = threshold_usage_ratio
        self.replaced_usage_ratio = replaced_usage_ratio
        self.check_unused_every = check_unused_every
        self._next_unused_check = check_unused_every
        self._cached_initialized = False

        self._initialized: torch.Tensor
        self.cluster_usage: torch.Tensor
        self.embedding_sum: torch.Tensor
        self._embedding: torch.Tensor
        self.register_buffer("_initialized", torch.tensor([False], dtype=torch.float))
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        embedding = torch.zeros(codebook_size, dim)
        self.register_buffer("embedding_sum", embedding)
        self.register_buffer("_embedding", None, persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        # Mapping old names to new names
        mappings = {
            "inited": "_initialized",
            "cluster_size": "cluster_usage",
            "embed_avg": "embedding_sum",
            "embed_sum": "embedding_sum",
        }
        for old_name, new_name in mappings.items():
            old_name = prefix + old_name
            if old_name in state_dict:
                value = state_dict.pop(old_name)
                if new_name is not None:
                    state_dict[prefix + new_name] = value
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding is None:
            embedding = (
                self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            )
            self.register_buffer("_embedding", embedding, persistent=False)
            return embedding
        return self._embedding

    @property
    def initialized(self) -> bool:
        """Cached version of self._initialized,
        This assumes that once the module is initialized, it will never go back to the uninitialized state."""
        if not self._cached_initialized:
            self._cached_initialized = bool(self._initialized.item())
        return self._cached_initialized

    def _init_embedding(self, data: torch.Tensor) -> None:
        # Initialize the codebook, e.g. using kmeans.
        if self.initialized:
            return

        rank = 0
        if _is_distributed():
            rank = distributed.get_rank()
            # First gathering shapes in case not all GPUs have the same effective batch size.
            # then gathering the actual content.
            if rank == 0:
                other_shapes: tp.List[torch.Size] = [None] * distributed.get_world_size()  # type: ignore
                distributed.gather_object(data.shape, other_shapes)
                other_data: tp.List[torch.Tensor] = [
                    torch.empty(shape, device=data.device, dtype=data.dtype) for shape in other_shapes]
                distributed.gather(data, other_data)
                data = torch.cat(other_data, dim=0)
            else:
                distributed.gather_object(data.shape)
                distributed.gather(data)
        if rank == 0:
            embedding, cluster_usage = _run_kmeans(data, self.codebook_size)
            self.embedding_sum.data.copy_(embedding * cluster_usage[:, None])
            self.cluster_usage.data.copy_(cluster_usage)
            self._initialized.data.fill_(1)
        # Make sure all buffers across workers are in sync after initialization
        self._broadcast_buffers()

    def _broadcast_buffers(self) -> None:
        if _is_distributed():
            for buffer in self.buffers():
                distributed.broadcast(buffer, 0)

    def _replace_expired_codes(self, samples: torch.Tensor, mask: torch.Tensor) -> None:
        # Replaces expired centroids, as indicated by `mask` (a true value indicate the code needs to be replaced).
        # The new codes are sampled from the batch `samples`.
        new_vectors = _sample_vectors(samples, self.codebook_size)
        replace_cluster_usage = (
            self.replaced_usage_ratio * self.cluster_usage.sum() / self.codebook_size
        )
        self.embedding_sum[:] = torch.where(
            mask[:, None], replace_cluster_usage * new_vectors, self.embedding_sum
        )
        self.cluster_usage[:] = torch.where(
            mask, replace_cluster_usage, self.cluster_usage
        )

    def _check_expired_codes(self, batch_samples: torch.Tensor) -> torch.Tensor:
        # Checks whether some centroids are under utilized, and replace them if necessary.
        if not self.initialized:
            return zero_scalar(batch_samples.device)

        self._next_unused_check -= 1
        if self._next_unused_check > 0:
            return zero_scalar(batch_samples.device)
        # we don't check every iteration to avoid having too many sync points.
        self._next_unused_check = self.check_unused_every
        threshold_cluster_usage = self.threshold_usage_ratio * self.cluster_usage.sum() / self.codebook_size
        expired_codes = self.cluster_usage < threshold_cluster_usage

        assert batch_samples.dim() == 2
        self._replace_expired_codes(batch_samples, mask=expired_codes)
        self._broadcast_buffers()

        return expired_codes.float().mean()

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        # Flattens all the dimensions but the last one, e.g. return a vector of shape `[N, D]`.
        x = rearrange(x, "... d -> (...) d")
        return x

    def _reshape_codes(self, codes: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        return codes.view(*shape[:-1])

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        # Projects each vector in `x` over the nearest centroid and return its index.
        # `x` should be `[N, D]` with `N` the number of input vectors and `D` the dimension.
        assert x.dim() == 2
        dists = torch.cdist(x[None], self.embedding[None], p=2)[0]
        codes = dists.argmin(dim=-1)
        return codes

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Given a tensor `x` of shape `[*, D]`, returns a tensor of integer codes of shape `[*]`.
        The codes are defined as the indexes of the centroids nearest to each vector in `x`.
        """
        assert x.dtype.is_floating_point, f"Input should be floats, got {x.dtype}"
        shape = x.shape
        x = self._reshape_input(x)
        codes = self._quantize(x)
        codes = self._reshape_codes(codes, shape)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Given a tensor of codes of shape `[*]`, returns a tensor of shape `[*, D]`,
        corresponding to the centroids associated to each code index.
        """
        assert (
            not codes.dtype.is_floating_point
        ), f"Codes should be integers, got {codes.dtype}"
        quantized = F.embedding(codes, self.embedding)
        return quantized

    def forward(
        self, x: torch.Tensor, initialize: bool = True
    ) -> _CodebookForwardResult:
        shape = x.shape
        x = self._reshape_input(x)

        if self.training and initialize:
            # If initialize is False, we are not allowed to initialize this layer
            # and the rest of the code will operate on a 0 filled codebook.
            # This is due to previous layers having used the batch to run kmeans init
            # and thus, the residuals are mostly 0s.
            self._init_embedding(x.detach())

        flat_codes = self._quantize(x)
        codes = self._reshape_codes(flat_codes, shape)
        quantized = self.decode(codes)
        metrics: tp.Dict[str, torch.Tensor] = {}

        if self.training:
            # We do the expiry of the unused codes at this point as buffers are in sync
            # and all the workers will take the same decision.
            expired = self._check_expired_codes(x)
            metrics['rvq_expired'] = expired
            cluster_usage = torch.zeros_like(self.cluster_usage)
            cluster_usage.scatter_add_(
                0, flat_codes, torch.ones_like(flat_codes, dtype=cluster_usage.dtype))
            _ema_inplace(self.cluster_usage, cluster_usage, self.decay)

            if self.initialized:
                # We report the entropy normalized by that of the uniform distribution,
                # This means the codebooks are optimally used when entropy=1.
                metrics['rvq_entropy'] = _compute_entropy(self.cluster_usage) / math.log(self.codebook_size)

            embedding_sum = torch.zeros_like(self.embedding_sum)
            embedding_sum.scatter_add_(0, repeat(flat_codes, "n -> n d", d=self.dim), x)
            _ema_inplace(self.embedding_sum, embedding_sum, self.decay)
            self.register_buffer('_embedding', None)

        return _CodebookForwardResult(quantized, codes, metrics)


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.

    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_usage_ratio (float): Defines the threshold for the cluster usage under which a centroid
            is replaced. This is expressed as a fraction of the usage a centroid would get under
            a uniform distribution, so that it doesn't depend on the batch size etc.
        replaced_usage_ratio (float): When replacing a centroid, use this as an initial centroid usage,
            to avoid the centroid getting replaced too quickly.
        check_unused_every (int): Check for unused centroids every `check_unused_every` iterations.
            This is to avoid too many synchronization points.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_usage_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim

        requires_projection = codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.epsilon = epsilon
        self._codebook = EuclideanCodebook(
            dim=codebook_dim,
            codebook_size=codebook_size,
            decay=decay,
            epsilon=epsilon,
            threshold_usage_ratio=threshold_usage_ratio,
            **kwargs,
        )
        self.codebook_size = codebook_size

    @property
    def embedding(self):
        return self._codebook.embedding

    @property
    def initialized(self):
        return self._codebook.initialized

    def _rearrange_input(self, x):
        x = rearrange(x, "b d n -> b n d")
        return x

    def _rearrange_output(self, quantized):
        quantized = rearrange(quantized, "b n d -> b d n")
        return quantized

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes `x` into discrete integer codes."""
        x = self._rearrange_input(x)
        x = self.project_in(x)
        codes = self._codebook.encode(x)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts integer codes into quantized vectors."""
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        quantized = self._rearrange_output(quantized)
        return quantized

    def forward(self, x: torch.Tensor, initialize: bool = True) -> _VQForwardResult:
        x = self._rearrange_input(x)
        quantized, codes, metrics = self._codebook(x, initialize=initialize)

        if self.training:
            quantized = x + (quantized - x).detach()
            loss = F.mse_loss(x, quantized.detach())
        else:
            loss = zero_scalar(x.device)

        quantized = self.project_out(quantized)
        quantized = self._rearrange_output(quantized)

        return _VQForwardResult(quantized, codes, loss, metrics)


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers: int, codebook_offset: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )
        self.codebook_offset = codebook_offset

    def forward(
        self, x: torch.Tensor, n_q: tp.Optional[int] = None
    ) -> _VQForwardResult:
        """
        Args:
            x (torch.Tensor): input tensor to quantize, of shape `[B, C, T]`.
            n_q (int or None): if provided, number of codebook levels to use in RVQ.
        """

        quantized_out = zero_scalar(x.device)
        residual = x

        all_losses = []
        all_codes = []
        all_metrics: tp.Dict[str, torch.Tensor] = {}

        n_q = n_q or len(self.layers)
        previous_layer_is_initialized = True

        for i, layer in enumerate(self.layers[:n_q]):  # type: ignore
            if self.training:
                this_layer_is_initialized = layer.initialized
            # We only allow the kmeans initialization if the previous layer is already initialized from the previous
            # iterations, this is to avoid learning the subsequent kmeans on the same batch, which would eventually
            # lead to its exhaustion and running kmeans on 0 values.
            quantized, codes, loss, metrics = layer(
                residual, initialize=previous_layer_is_initialized
            )
            if self.training:
                previous_layer_is_initialized = this_layer_is_initialized  # type: ignore

            quantized = quantized.detach()
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_codes.append(codes)
            all_losses.append(loss)

            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key] += value / n_q
                else:
                    all_metrics[key] = value / n_q
                all_metrics[key + f"_{i + self.codebook_offset}"] = value

        if self.training:
            # Solving subtle bug with STE and RVQ: https://github.com/facebookresearch/encodec/issues/25
            quantized_out = x + (quantized_out - x).detach()
            to_average = []
            for layer in self.layers:
                assert isinstance(layer, VectorQuantization)
                to_average += [layer._codebook.cluster_usage, layer._codebook.embedding_sum]
                _average_tensors(to_average)

        out_losses, out_codes = map(torch.stack, (all_losses, all_codes))
        return _VQForwardResult(quantized_out, out_codes, out_losses, all_metrics)

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        """Encodes `x` into discrete integer codes. If `n_q` is provided, only uses the first `n_q` codebook levels."""
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:  # type: ignore
            assert isinstance(layer, VectorQuantization)
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts the integer codes into quantized vectors."""
        quantized = zero_scalar(codes.device)
        for idx, layer_codes in enumerate(codes):
            layer = self.layers[idx]
            assert isinstance(layer, VectorQuantization)
            quantized = quantized + layer.decode(layer_codes)
        return quantized
