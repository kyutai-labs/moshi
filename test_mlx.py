# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx

import msh_mlx

lm_config = msh_mlx.models.config_v0_1()

model = msh_mlx.models.Lm(lm_config)
print(model)
print(lm_config)
model.set_dtype()
