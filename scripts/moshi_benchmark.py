# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import time

import numpy as np
import sphn
import torch
from torch.profiler import profile, ProfilerActivity

from moshi.models import loaders, LMGen


parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                    help="HF repo to look into, defaults Moshiko. "
                         "Use this to select a different pre-trained model.")
parser.add_argument("--lora-weight", type=str, help="Path to a local checkpoint file for LoRA.", default=None)
parser.add_argument("--config-path", type=str, help="Path to a local config file.", default=None)
parser.add_argument("--steps", default=100, type=int)
parser.add_argument("--no_fuse_lora", action="store_false", dest="fuse_lora", default=True,
                    help="Do not fuse LoRA layers intot Linear layers.")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all(42424242)

print("retrieving checkpoint")
checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
    args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer,
    lora_weights=args.lora_weight, config_path=args.config_path)
print("loading mimi")
mimi = checkpoint_info.get_mimi(device=args.device)
print("mimi loaded")

text_tokenizer = checkpoint_info.get_text_tokenizer()

print("loading moshi")
lm = checkpoint_info.get_moshi(device=args.device, dtype=torch.bfloat16, fuse_lora=args.fuse_lora)
lm_gen = LMGen(lm)
print("moshi loaded")

def cb(step, total):
    print(f"{step:06d} / {total:06d}", end="\r")


def streaming_test(bs):
    main_audio = []
    main_text = []

    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    def run_step():
        start_time = time.time()
        # Chunk should contain the pcm data from the user, single channel with a sample rate of 24000.
        chunk = torch.zeros((bs, 1, frame_size), dtype=torch.float, device=args.device)
        codes = mimi.encode(chunk)
        assert codes.shape[-1] == 1
        be = time.time()
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        tokens = lm_gen.step(codes[:, :, :1])
        if tokens is None:
            print("Skipping")
            return
        evb = torch.cuda.Event(enable_timing=True)
        evb.record()
        dt_step = time.time() - be
        text_tokens = tokens[:, 0, 0]
        audio_tokens = tokens[:, 1:, :]
        main_pcm = mimi.decode(audio_tokens)
        # main_pcm is the audio to be played back to the user, here we just append it and store it in
        # a file once the loop is finished.
        main_audio.append(main_pcm[0])
        evb.synchronize()
        dg = ev.elapsed_time(evb)
        torch.cuda.synchronize()
        dt = time.time() - start_time
        print(
            f"step time: {1000 * dt:.2f}ms, lm step: {1000 * dt_step:.2f}, gpu step {dg:.2f}"
        )
        text_token = text_tokens[0].item()
        if text_token not in (0, 3):
            _text = text_tokenizer.id_to_piece(text_token)
            _text = _text.replace("▁", " ")
            main_text.append(_text)

    for step in range(args.steps):
        run_step()
    print()
    if args.profile:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True
        ) as prof:
            for step in range(5):
                run_step()
        print()
        prof.export_chrome_trace("trace.json")
    main_audio_th = torch.cat(main_audio, dim=-1)
    print(main_audio_th.shape)
    print("generated text:")
    print("".join(main_text))
    sphn.write_wav(
        "gen_main.wav",
        main_audio_th[0].cpu().numpy().astype(np.float32),
        mimi.sample_rate,
    )


print("streaming test")
bs = 1
with torch.no_grad():
    with mimi.streaming(bs), lm_gen.streaming(bs):
        streaming_test(bs)
