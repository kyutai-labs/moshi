# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
moshi is the inference codebase for Kyutai audio generation models.

The code has been adapted from Audiocraft, see LICENSE.audiocraft
  Copyright (c) Meta Platforms, Inc. and affiliates.
"""

# flake8: noqa
from . import conditioners
from . import models
from . import modules
from . import quantization
from . import utils

__version__ = "0.2.10"
