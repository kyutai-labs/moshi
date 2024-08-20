# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Models for EnCodec, AudioGen, MusicGen, as well as the generic LMModel.
"""

# flake8: noqa
from .encodec import (
    CompressionModel,
    EncodecModel,
)
from .lm import LMModel, LMGen
from .moshi import get_encodec, get_lm
