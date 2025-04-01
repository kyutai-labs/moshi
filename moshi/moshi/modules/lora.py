import torch
import torch.nn as nn


def replace_all_linear_with_lora(module, rank: int, scaling: float, device=None, dtype=None):
    """ Recursively replace all Linear layers with LoRALinear layers."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if device is None:
                this_device = child.weight.device
            else:
                this_device = device
            if dtype is None:
                this_dtype = child.weight.dtype
            else:
                this_dtype = dtype
            lora = LoRALinear(child.in_features, child.out_features,
                              rank, scaling, device=this_device, dtype=this_dtype)
            lora.frozen_W = child
            setattr(module, name, lora)
        else:
            replace_all_linear_with_lora(child, rank, scaling, device=device, dtype=dtype)


def replace_lora_with_linear(module):
    """Recursively replace all LoRALinear layers with Linear layers."""
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            # Compute merged weights: W' = W + scaling * B @ A
            merged_weight = child.frozen_W.weight.data + \
                child.scaling * (child.lora_B.weight @ child.lora_A.weight)
            # Create a standard Linear layer with the same in/out features
            new_linear = nn.Linear(child.frozen_W.in_features,
                                   child.frozen_W.out_features, bias=False,
                                   device=torch.device('meta'),
                                   dtype=merged_weight.dtype)
            new_linear.weight = nn.Parameter(
                merged_weight, requires_grad=merged_weight.requires_grad)  # Transfer merged weights
            setattr(module, name, new_linear)  # Replace the module
        else:
            replace_lora_with_linear(child)  # Recursively process submodules


class LoRALinear(nn.Module):
    """
    Implementation of:
        - LoRA: https://arxiv.org/abs/2106.09685

    Notes:
        - Freezing is handled at the network level, not the layer level.
        - Scaling factor controls relative importance of LoRA skip
          connection versus original frozen weight. General guidance is
          to keep it to 2.0 and sweep over learning rate when changing
          the rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scaling: float,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert not bias
        self.bias = bias
        self.rank = rank
        self.scaling = scaling

        self.lora_A = nn.Linear(
            self.in_features,
            self.rank,
            bias=self.bias,
            device=device,
            dtype=dtype,
        )
        self.lora_B = nn.Linear(
            self.rank,
            self.out_features,
            bias=self.bias,
            device=device,
            dtype=dtype,
        )

        self.frozen_W = nn.Linear(self.in_features,
                                  self.out_features,
                                  bias=self.bias,
                                  device=device,
                                  dtype=dtype)

        self._register_load_state_dict_pre_hook(LoRALinear._load_hook, with_module=True)

    def merge_weight(self):
        with torch.no_grad():
            down_weight = self.lora_A.weight
            up_weight = self.lora_B.weight

            weight = up_weight.mm(down_weight) * self.scaling

            weight += self.frozen_W.weight
        return weight

    @staticmethod
    def _load_hook(module, state_dict, prefix, *_):
        key_name = prefix + "weight"
        if key_name in state_dict:
            w_ref = state_dict.pop(key_name)
            state_dict[prefix + 'frozen_W.weight'] = w_ref

    def forward(self, x: torch.Tensor):
        lora = self.lora_B(self.lora_A(x))
        return self.frozen_W(x) + lora * self.scaling

    def __repr__(self) -> str:
        return "{}Linear(in_features={}, out_features={}, r={})".format(
            "LoRA", self.in_features, self.out_features, self.rank)
