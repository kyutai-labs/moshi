# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

class CompressionModel(nn.Module):
    def __init__(self):
        super().__init__()

class EncodecModel(nn.Module):
    def __init__(self):
        super().__init__()
