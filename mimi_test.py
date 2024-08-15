import argparse
import msh
import time
import torch
import torchaudio
import torchaudio.functional as F
from torch.profiler import profile, ProfilerActivity
import numpy as np
import random

SAMPLE_RATE = msh.models.moshi.SAMPLE_RATE
DEVICE = "cuda:0"
ENABLE_PROFILING = False

parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str)
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
ec = msh.models.moshi.get_encodec(args.weights, DEVICE)
print("mimi loaded")


def encodec_streaming_test(ec, pcm_chunk_size=1920, max_duration_sec=10.0):
    # wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
    sample_pcm, sample_sr = torchaudio.load("bria.mp3")
    print("loaded pcm", sample_pcm.shape, sample_sr)
    sample_pcm = F.resample(sample_pcm, orig_freq=sample_sr, new_freq=SAMPLE_RATE)
    max_duration_len = int(SAMPLE_RATE * max_duration_sec)
    if sample_pcm.shape[-1] > max_duration_len:
        sample_pcm = sample_pcm[..., :max_duration_len]
    print("resampled pcm", sample_pcm.shape, sample_sr)
    sample_pcm = sample_pcm[None].to(device=DEVICE)

    print("streaming encoding...")
    start_time = time.time()
    all_codes = []

    def run_loop():
        for start_idx in range(0, sample_pcm.shape[-1], pcm_chunk_size):
            end_idx = min(sample_pcm.shape[-1], start_idx + pcm_chunk_size)
            chunk = sample_pcm[..., start_idx:end_idx]
            codes, _scale = ec.encode(chunk)
            if codes.shape[-1]:
                print(start_idx, codes.shape, end="\r")
                all_codes.append(codes)

    if ENABLE_PROFILING:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run_loop()
        prof.export_chrome_trace("trace.json")
    else:
        run_loop()
    all_codes = torch.cat(all_codes, dim=-1)
    print(f"codes {all_codes.shape} generated in {time.time() - start_time:.2f}s")
    print("streaming decoding...")
    all_pcms = []
    with ec.streaming():
        for i in range(all_codes.shape[-1]):
            codes = all_codes[..., i : i + 1]
            pcm = ec.decode(codes, scale=None)
            print(i, pcm.shape, end="\r")
            all_pcms.append(pcm)
    all_pcms = torch.cat(all_pcms, dim=-1)
    print("pcm", all_pcms.shape)
    torchaudio.save("streaming_out.wav", all_pcms[0].cpu(), SAMPLE_RATE)
    pcm = ec.decode(all_codes, scale=None)
    print("pcm", pcm.shape)
    torchaudio.save("roundtrip_out.wav", pcm[0].cpu(), SAMPLE_RATE)


encodec_streaming_test(ec)
