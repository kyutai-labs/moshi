# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import time

from huggingface_hub import hf_hub_download
import numpy as np
import sphn
import torch
from torch.profiler import profile, ProfilerActivity

from moshi.models import loaders


parser = argparse.ArgumentParser()
parser.add_argument("--mimi-weight", type=str)
parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.device_count() else "cpu"
)
parser.add_argument("--profile", action="store_true")
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


print("loading mimi")
if args.mimi_weight is None:
    args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
mimi = loaders.get_mimi(args.mimi_weight, args.device)
print("mimi loaded")


def mimi_streaming_test(mimi, max_duration_sec=10.0):
    pcm_chunk_size = int(mimi.sample_rate / mimi.frame_rate)
    # wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
    sample_pcm, sample_sr = sphn.read("bria.mp3")
    sample_rate = mimi.sample_rate
    print("loaded pcm", sample_pcm.shape, sample_sr)
    sample_pcm = sphn.resample(
        sample_pcm, src_sample_rate=sample_sr, dst_sample_rate=sample_rate
    )
    sample_pcm = torch.tensor(sample_pcm, device=args.device)
    max_duration_len = int(sample_rate * max_duration_sec)
    if sample_pcm.shape[-1] > max_duration_len:
        sample_pcm = sample_pcm[..., :max_duration_len]
    print("resampled pcm", sample_pcm.shape, sample_sr)
    sample_pcm = sample_pcm[None].to(device=args.device)

    print("streaming encoding...")
    start_time = time.time()
    all_codes = []

    def run_loop():
        for start_idx in range(0, sample_pcm.shape[-1], pcm_chunk_size):
            end_idx = min(sample_pcm.shape[-1], start_idx + pcm_chunk_size)
            chunk = sample_pcm[..., start_idx:end_idx]
            codes = mimi.encode(chunk)
            if codes.shape[-1]:
                print(start_idx, codes.shape, end="\r")
                all_codes.append(codes)

    if args.profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run_loop()
        prof.export_chrome_trace("trace.json")
    else:
        run_loop()
    all_codes_th = torch.cat(all_codes, dim=-1)
    print(f"codes {all_codes_th.shape} generated in {time.time() - start_time:.2f}s")
    print("streaming decoding...")
    all_pcms = []
    with mimi.streaming(1):
        for i in range(all_codes_th.shape[-1]):
            codes = all_codes_th[..., i : i + 1]
            pcm = mimi.decode(codes)
            print(i, pcm.shape, end="\r")
            all_pcms.append(pcm)
    all_pcms = torch.cat(all_pcms, dim=-1)
    print("pcm", all_pcms.shape, all_pcms.dtype)
    sphn.write_wav("streaming_out.wav", all_pcms[0, 0].cpu().numpy(), sample_rate)
    pcm = mimi.decode(all_codes_th)
    print("pcm", pcm.shape, pcm.dtype)
    sphn.write_wav("roundtrip_out.wav", pcm[0, 0].cpu().numpy(), sample_rate)


with torch.no_grad():
    mimi_streaming_test(mimi)
