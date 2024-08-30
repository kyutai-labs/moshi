# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import time
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

import msh_mlx

STEPS = 1000

mx.random.seed(299792458)

lm_config = msh_mlx.models.config_v0_1()

model = msh_mlx.models.Lm(lm_config)
print(model)
print(lm_config)
model.set_dtype(mx.bfloat16)
print("model created")
nn.quantize(model, bits=8)
print("model quantized")

cache = None
start_time = 0
for i in range(STEPS + 1):
    if i == 1:
        start_time = time.time()
    token_ids = mx.array([[42]])
    logits, cache = model(token_ids, cache)
    print(i, logits)
token_per_second = STEPS / (time.time() - start_time)
print(f"steps: {STEPS}, token per sec: {token_per_second}")
