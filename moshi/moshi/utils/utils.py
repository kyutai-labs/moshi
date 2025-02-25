import torch

from .compile import torch_compile_lazy


@torch_compile_lazy
def cross_entropy(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, dtype=torch.float32,
        logits_soft_clip: float | None = None) -> torch.Tensor:
    """Compute cross entropy between multi-codebook targets and model's logits.
    The cross entropy is computed per codebook to provide codebook-level cross entropy.
    Valid timesteps for each of the codebook are pulled from the mask, where invalid
    timesteps are set to 0.

    Args:
        logits (torch.Tensor): Model's logits of shape [B, K, T, card].
        targets (torch.Tensor): Target codes, of shape [B, K, T].
        mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        dtype (type): Data type of the output cross entropy.
        logits_soft_clip (float): Clipping value for the logits to avoid numerical instability.
            Recommended value: 30.0.
    Returns:
        ce (torch.Tensor): Cross entropy [B, K, T] with type dtype.
    """
    output_shape = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    logits = logits.view(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    mask = mask.reshape(-1)

    safe_targets = torch.where(
        mask,
        targets,
        torch.zeros(1, device=targets.device, dtype=targets.dtype),
    )

    # Chunking the conversion to float32 to avoid OOMs.
    ce_chunks = []
    for logits_chunk, targets_chunk in zip(torch.chunk(logits, 4), torch.chunk(safe_targets, 4)):
        logits_chunk = logits_chunk.to(dtype)
        if logits_soft_clip is not None:
            logits_chunk = logits_soft_clip * torch.tanh(logits_chunk / logits_soft_clip)
        log_partition = torch.logsumexp(logits_chunk, dim=-1, keepdim=True)

        # For some reason, the PyTorch cross entropy is super slow with inputs with large cardinality (e.g. 32000)
        # so we reimplement the cross entropy ourselves...
        ce_chunks.append(log_partition - logits_chunk.gather(-1, targets_chunk[..., None]))
    ce = torch.cat(ce_chunks, dim=0)
    ce = ce[..., 0]
    ce = torch.where(mask, ce, torch.zeros(1, device=ce.device, dtype=ce.dtype))
    return ce.view(output_shape)
