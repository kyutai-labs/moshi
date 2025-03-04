from typing import NamedTuple

import torch
import torch.nn as nn
from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from functools import partial, reduce

@dataclass
class LoraArgs(Serializable):
    enable: bool = False
    rank: int = 64
    scaling: float = 2.0

    def __post_init__(self) -> None:
        if self.enable:
            assert self.rank > 0
            assert self.scaling > 0.0



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

        self.frozen_W = nn.Linear(self.in_features, self.out_features, bias=self.bias, device=device, dtype=dtype)

        # make sure no LoRA weights are marked as "missing" in load_state_dict
        def ignore_missing_keys(m: nn.Module, incompatible_keys: NamedTuple):
            # empty missing keys in place
            incompatible_keys.missing_keys[:] = []  # type: ignore

        self.register_load_state_dict_post_hook(ignore_missing_keys)

    def merge_weight(self):
        with torch.no_grad():
            down_weight = self.lora_A.weight
            up_weight = self.lora_B.weight

            weight = up_weight.mm(down_weight) * self.scaling

            weight += self.frozen_W.weight
        return weight

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata, 
        strict, 
        missing_keys, 
        unexpected_keys, 
        error_msgs
    ):
        key_name = prefix + "weight"

        # full checkpoint
        if key_name in state_dict:
            w_ref = state_dict[key_name]

            # load frozen weights
            self.frozen_W.load_state_dict({"weight": w_ref}, assign=True)

    def forward(self, x: torch.Tensor):
        lora = self.lora_B(self.lora_A(x))
        return self.frozen_W(x) + lora * self.scaling

    def __repr__(self) -> str:
        return "{}Linear(in_features={}, out_features={}, r={})".format(
            "LoRA", self.in_features, self.out_features, self.rank
        )
        
        

def maybe_lora_layer(
    lora_args: LoraArgs | None
) -> partial[LoRALinear] | type[nn.Linear]:
    MaybeLora: partial[LoRALinear] | type[nn.Linear]
    if lora_args is None or not lora_args.enable:
        return nn.Linear

    rank = lora_args.rank
    scaling = lora_args.scaling

    MaybeLora = partial(
        LoRALinear,
        rank=rank,
        scaling=scaling,
    )

    return MaybeLora