# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import time
from typing import List, Optional, Tuple
import sentencepiece

import mlx.core as mx
import mlx.nn as nn

import msh_mlx

STEPS = 1000
MODEL_FILE = "/Users/kyutai/tmp/mimi_0abbed5f@100.safetensors"
TOKENIZER_FILE = "/Users/kyutai/tmp/tokenizer_spm_32k_3.model"

text_tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_FILE)
mx.random.seed(299792458)

lm_config = msh_mlx.models.config_v0_1()

model = msh_mlx.models.Lm(lm_config)
print(model)
print(lm_config)
model.set_dtype(mx.bfloat16)
print("model created")
# nn.quantize(model, bits=8)
print("model quantized, loading weights...")
model.load_weights(MODEL_FILE, strict=False)
print("weights loaded")

cache = None
start_time = 0
last_token = mx.array([[32000]])
sampler = msh_mlx.utils.Sampler()
for i in range(STEPS + 1):
    if i == 1:
        start_time = time.time()
    logits, cache = model(last_token, cache)
    last_token, _ = sampler(logits[:, 0])
    text_token = last_token[0].item()
    if text_token not in (0, 3):
        _text = text_tokenizer.id_to_piece(text_token)
        _text = _text.replace("▁", " ")
        print(_text, end='', flush=True)

    last_token = last_token[None]
token_per_second = STEPS / (time.time() - start_time)
print(f"steps: {STEPS}, token per sec: {token_per_second}")
