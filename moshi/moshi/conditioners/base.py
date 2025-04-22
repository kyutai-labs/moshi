# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
import logging
import typing as tp

import torch
from torch import nn

from ..modules.transformer import create_sin_embedding


logger = logging.getLogger(__name__)
TextCondition = tp.Optional[str]  # a text condition can be a string or None (if doesn't exist)
ConditionTensors = dict[str, 'ConditionType']


class ConditionType(tp.NamedTuple):
    """Return type for a conditioner: both a condition tensor, and a mask indicating valid positions.
    """
    condition: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class TensorCondition:
    """Looks quite similar to ConditionType, but represents the input to TensorConditioners.
    `tensor` should be [B | 1, T, D], and `mask` should be `[B | 1, T]`.
    """
    tensor: torch.Tensor
    mask: torch.Tensor

    @staticmethod
    def from_tensor(tensor: torch.Tensor):
        B, T, _ = tensor.shape
        mask = torch.ones(B, T, dtype=torch.bool, device=tensor.device)
        return TensorCondition(tensor, mask)

    @staticmethod
    def cat(conditions: tp.Sequence['TensorCondition']) -> 'TensorCondition':
        assert conditions, "Cannot cat empty list."
        ref_tensor = conditions[0].tensor
        B, _, D = ref_tensor.shape
        assert B == 1
        B = len(conditions)
        T = max(condition.tensor.shape[1] for condition in conditions)
        mask = torch.zeros(B, T, dtype=torch.bool, device=ref_tensor.device)
        tensor = torch.zeros(B, T, D, dtype=ref_tensor.dtype, device=ref_tensor.device)
        for b, condition in enumerate(conditions):
            tensor[b, :condition.tensor.shape[1], :] = condition.tensor[0]
            mask[b, :condition.mask.shape[1]] = condition.mask[0]
        return TensorCondition(tensor, mask)


@dataclass
class ConditionAttributes:
    """Standard class for representing the set of potential inputs to the conditioners.
    Typically, `audiocraft.data.audio_dataset.SegmentInfo` will convert
    to this class to make conditioning agnostic to the type of dataset.

    There are two kinds of conditionings: text (or None), or raw torch tensors (with a mask).

    """
    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    tensor: tp.Dict[str, TensorCondition] = field(default_factory=dict)

    @property
    def text_attributes(self) -> tp.Iterable[str]:
        return self.text.keys()

    @property
    def tensor_attributes(self) -> tp.Iterable[str]:
        return self.text.keys()

    @staticmethod
    def condition_types() -> tp.FrozenSet[str]:
        return frozenset(["text", "tensor"])

    def copy(self) -> 'ConditionAttributes':
        return ConditionAttributes(dict(self.text), dict(self.tensor))


Prepared = tp.TypeVar('Prepared')  # represents the prepared condition input type.


class BaseConditioner(nn.Module, tp.Generic[Prepared]):
    """Base model for all conditioner modules.

    Args:
        dim (int): internal dim of the model.
        output_dim (int): Output dim of the conditioner.
        force_linear (bool, optional): Force linear projection even when `dim == output_dim`.
        pad_empty (bool): if True, conditionings of 0 length will be padded to have length 1.
        output_bias (bool): if True, the output projection will have a bias.
        learn_padding (bool): if True, the padding value will be learnt, zero otherwise.
    """

    def __init__(self, dim: int,
                 output_dim: int,
                 device: tp.Union[torch.device, str],
                 force_linear: bool = True,
                 pad_empty: bool = True,
                 output_bias: bool = False,
                 learn_padding: bool = True):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.pad_empty = pad_empty
        self.device = device
        self.output_proj: nn.Module
        if force_linear or dim != output_dim:
            self.output_proj = nn.Linear(dim, output_dim, bias=output_bias, device=device)
            assert not output_bias
        else:
            self.output_proj = nn.Identity()
        self.learnt_padding: tp.Optional[torch.Tensor]
        if learn_padding:
            self.learnt_padding = nn.Parameter(
                torch.randn(1, 1, output_dim, device=device), requires_grad=True)
            self.learnt_padding.data *= 0.2
        else:
            self.learnt_padding = None

    def prepare(self, *args, **kwargs) -> Prepared:
        """Should be any part of the processing that will lead to a synchronization
        point, e.g. BPE tokenization with transfer to the GPU.

        The returned value will be saved and return later when calling forward().
        """
        raise NotImplementedError()

    def _get_condition(self, inputs: Prepared) -> ConditionType:
        """Gets input that should be used as conditioning (e.g, genre, description or a waveform).
        Outputs a ConditionType, after the input data was embedded as a dense vector.

        Returns:
            ConditionType:
                - A tensor of size [B, T, dim] where B is the batch size, T is the length of the
                  output embedding and `dim` is the internal dimension of the embedding.
                - And a mask indicating where the padding tokens, of shape `[B, T]`.
        """
        raise NotImplementedError()

    def forward(self, inputs: Prepared) -> ConditionType:
        cond, mask = self._get_condition(inputs)
        B, T, C = cond.shape
        if T == 0 and self.pad_empty:
            cond = torch.zeros(B, T, C, device=cond.device, dtype=cond.dtype)
            mask = torch.zeros_like(cond[..., 0], dtype=torch.bool)

        cond = self.output_proj(cond)

        maskf = mask.float()[..., None]
        if self.learnt_padding is not None:
            cond = cond * maskf + self.learnt_padding * (1 - maskf)
        else:
            cond = cond * maskf
        return ConditionType(cond, mask)


class _BaseTextConditioner(BaseConditioner[Prepared]):
    pass


class _BaseTensorConditioner(BaseConditioner[Prepared]):
    pass


def dropout_tensor(condition: TensorCondition) -> TensorCondition:
    """Utility function for nullifying a WavCondition object.
    """
    return TensorCondition(
        tensor=torch.zeros_like(condition.tensor),
        mask=torch.zeros_like(condition.mask))


def dropout_condition_(sample: ConditionAttributes, condition_type: str, condition: str) -> None:
    """Utility function for nullifying an attribute inside a ConditionAttributes object.
    Works in-place.
    """
    valid_conditions = ConditionAttributes.condition_types()
    if condition_type not in valid_conditions:
        raise ValueError(
            "dropout_condition got an unexpected condition type!"
            f" expected one of {valid_conditions} but got '{condition_type}'")

    if condition not in getattr(sample, condition_type):
        raise ValueError(
            "dropout_condition received an unexpected condition!"
            f" expected tensor={sample.tensor.keys()} and text={sample.text.keys()}"
            f" but got '{condition}' of type '{condition_type}'!"
        )

    if condition_type == 'tensor':
        tensor_condition = sample.tensor[condition]
        sample.tensor[condition] = dropout_tensor(tensor_condition)
    elif condition_type == 'text':
        sample.text[condition] = None
    else:
        assert False


def dropout_all_conditions(attributes: tp.Sequence[ConditionAttributes]) -> list[ConditionAttributes]:
    """
    Args:
        attributes (list[ConditionAttributes]): All conditions attributes.
    Returns:
        list[ConditionAttributes]: Same with all conditions dropped.
    """
    attributes = [attribute.copy() for attribute in attributes]
    for condition_type in ConditionAttributes.condition_types():
        for attribute in attributes:
            for condition in getattr(attribute, condition_type):
                dropout_condition_(attribute, condition_type, condition)
    return attributes


class ConditionProvider(nn.Module):
    """Prepare and provide conditions given all the supported conditioners.

    Args:
        conditioners (dict): Dictionary of conditioners.
        device (torch.device or str, optional): Device for conditioners and output condition types.
    """

    def __init__(self, conditioners: tp.Dict[str, BaseConditioner], device: tp.Union[torch.device, str] = "cpu"):
        super().__init__()
        self.device = device
        self.conditioners = nn.ModuleDict(conditioners).to(device)

    @property
    def text_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, _BaseTextConditioner)]

    @property
    def tensor_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, _BaseTensorConditioner)]

    def _collate_text(self, samples: tp.Sequence[ConditionAttributes]) -> tp.Dict[str, tp.List[tp.Optional[str]]]:
        """Given a list of ConditionAttributes objects, compile a dictionary where the keys
        are the attributes and the values are the aggregated input per attribute.
        For example:
        Input:
        [
            ConditionAttributes(text={"genre": "Rock", "description": "A rock song with a guitar solo"}, wav=...),
            ConditionAttributes(text={"genre": "Hip-hop", "description": "A hip-hop verse"}, wav=...),
        ]
        Output:
        {
            "genre": ["Rock", "Hip-hop"],
            "description": ["A rock song with a guitar solo", "A hip-hop verse"]
        }

        Args:
            samples (list of ConditionAttributes): List of ConditionAttributes samples.
        Returns:
            dict[str, list[str, optional]]: A dictionary mapping an attribute name to text batch.
        """
        out: tp.Dict[str, tp.List[tp.Optional[str]]] = defaultdict(list)
        texts = [x.text for x in samples]
        for text in texts:
            for condition in self.text_conditions:
                out[condition].append(text[condition])
        return out

    def _collate_tensors(self, samples: tp.Sequence[ConditionAttributes]) -> tp.Dict[str, TensorCondition]:
        """For each tensor attribute, collate the tensor from individual batch items.

        Args:
            samples (list of ConditionAttributes): List of ConditionAttributes samples.
        Returns:
            dict[str, TensorCondition]: A dictionary mapping an attribute name to tensor.
        """
        per_attribute = defaultdict(list)
        out: tp.Dict[str, TensorCondition] = {}
        for sample in samples:
            for attribute in self.tensor_conditions:
                per_attribute[attribute].append(sample.tensor[attribute])

        # stack all tensors to a single tensor
        for attribute in self.tensor_conditions:
            out[attribute] = TensorCondition.cat(per_attribute[attribute])

        return out

    def prepare(self, inputs: tp.Sequence[ConditionAttributes]) -> tp.Dict[str, tp.Any]:
        """Match attributes/tensors with existing conditioners in self, and call `prepare` for each one.
        This should be called before starting any real GPU work to avoid synchronization points.
        This will return a dict matching conditioner names to their arbitrary prepared representations.

        Args:
            inputs (list[ConditionAttributes]): List of ConditionAttributes objects containing
                text and tensors conditions.
        """
        assert all([isinstance(x, ConditionAttributes) for x in inputs]), (
            "Got unexpected types input for conditioner! should be tp.List[ConditionAttributes]",
            f" but types were {set([type(x) for x in inputs])}"
        )

        output = {}
        text = self._collate_text(inputs)
        tensors = self._collate_tensors(inputs)

        assert set(text.keys() | tensors.keys()).issubset(set(self.conditioners.keys())), (
            f"Got an unexpected attribute! Expected {self.conditioners.keys()}, ",
            f"got {text.keys(), tensors.keys()}"
        )

        missing_inputs = set(self.conditioners.keys()) - (set(text.keys()) | set(tensors.keys()))
        if missing_inputs:
            raise RuntimeError(f"Some conditioners did not receive an input: {missing_inputs}")
        for attribute, batch in chain(text.items(), tensors.items()):
            conditioner = self.conditioners[attribute]
            assert isinstance(conditioner, BaseConditioner)
            output[attribute] = conditioner.prepare(batch)
        return output

    def forward(self, prepared: tp.Dict[str, tp.Any]) -> tp.Dict[str, ConditionType]:
        """Compute pairs of `(embedding, mask)` using the configured conditioners and the prepared representations.
        The output is for example:
        {
            "genre": (torch.Tensor([B, 1, D_genre]), torch.Tensor([B, 1])),
            "description": (torch.Tensor([B, T_desc, D_desc]), torch.Tensor([B, T_desc])),
            ...
        }

        Args:
            prepared (dict): Dict of prepared representations as returned by `prepare()`.
        """
        output = {}
        for name, inputs in prepared.items():
            condition, mask = self.conditioners[name](inputs)
            output[name] = ConditionType(condition, mask)
        return output


class ConditionFuser(nn.Module):
    """Condition fuser handles the logic to combine the different conditions
    to the actual model input.

    Args:
        fuse2cond (tp.Dict[str, str]): A dictionary that says how to fuse
            each condition. For example:
            {
                "prepend": ["description"],
                "sum": ["genre", "bpm"],
                "cross": ["description"],
            }
        cross_attention_pos_emb (bool, optional): Use positional embeddings in cross attention.
        cross_attention_pos_emb_scale (int): Scale for positional embeddings in cross attention if used.
    """
    FUSING_METHODS = ["sum", "prepend", "cross"]

    def __init__(self, fuse2cond: tp.Dict[str, tp.List[str]], cross_attention_pos_emb: bool = False,
                 cross_attention_pos_emb_scale: float = 1.0):
        super().__init__()
        assert all(
            [k in self.FUSING_METHODS for k in fuse2cond.keys()]
        ), f"Got invalid fuse method, allowed methods: {self.FUSING_METHODS}"
        self.cross_attention_pos_emb = cross_attention_pos_emb
        self.cross_attention_pos_emb_scale = cross_attention_pos_emb_scale
        self.fuse2cond: tp.Dict[str, tp.List[str]] = fuse2cond
        self.cond2fuse: tp.Dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = fuse_method
                if fuse_method not in ['cross', 'sum']:
                    raise RuntimeError("only `sum` and `cross` conditionings are supported "
                                       f"for now, got {fuse_method}.")

    @property
    def has_conditions(self) -> bool:
        return bool(self.cond2fuse)

    @property
    def has_prepend(self) -> bool:
        """Is there a conditioning that needs to be prepending to the Transformer sequence."""
        return bool(self.fuse2cond['prepend'])

    def get_cross(self, conditions: ConditionTensors) -> torch.Tensor | None:
        """Return the tensor to be provided for the cross attention."""
        cross = None
        for name in self.fuse2cond['cross']:
            cond, _ = conditions[name]
            if cross is None:
                cross = cond
            else:
                cross = torch.cat([cross, cond], dim=1)

        if self.cross_attention_pos_emb and cross is not None:
            positions = torch.arange(
                cross.shape[1],
                device=cross.device
            ).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, cross.shape[-1]).to(cross.dtype)
            cross = cross + self.cross_attention_pos_emb_scale * pos_emb
        return cross

    def get_sum(self, conditions: ConditionTensors) -> torch.Tensor | None:
        """Return the tensor to be provided as an extra sum offset shared for each step."""
        sum = None
        for name in self.fuse2cond['sum']:
            cond, _ = conditions[name]
            assert cond.shape[1] == 1, cond.shape
            if sum is None:
                sum = cond
            else:
                sum = sum + cond
        return sum

    def get_prepend(self, conditions: ConditionTensors) -> torch.Tensor | None:
        """Return the tensor to be prepended to the transformer."""
        prepend = None
        for name in self.fuse2cond['prepend']:
            cond, _ = conditions[name]
            if prepend is None:
                prepend = cond
            else:
                prepend = torch.cat([cond, prepend], dim=1)
        if prepend is not None:
            sum = self.get_sum(conditions)
            if sum is not None:
                prepend = prepend + sum
        return prepend
