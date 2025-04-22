# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from .base import _BaseTensorConditioner, TensorCondition, ConditionType


class TensorConditioner(_BaseTensorConditioner[TensorCondition]):
    """Does basically nothing.
    """

    def prepare(self, tensor: TensorCondition) -> TensorCondition:
        device = next(iter(self.parameters())).device
        return TensorCondition(tensor.tensor.to(device=device), tensor.mask.to(device=device))

    def _get_condition(self, inputs: TensorCondition) -> ConditionType:
        return ConditionType(inputs.tensor, inputs.mask)
