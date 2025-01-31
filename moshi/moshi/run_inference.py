# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import torch
import sphn


from .client_utils import make_log
from .models import loaders, MimiModel, LMModel, LMGen


def log(level: str, msg: str):
    print(make_log(level, msg))


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


@dataclass
class InferenceState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen

    def __init__(self, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, batch_size: int, device: str | torch.device):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm_gen = LMGen(lm)
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.mimi.streaming_forever(batch_size)
        self.lm_gen.streaming_forever(batch_size)

    def run(self, in_pcms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_pcms = []
        out_text_tokens = []
        log("info", "starting the inference loop")
        start_time = time.time()
        ntokens = 0
        for chunk in in_pcms.split(1920, dim=2):
            if chunk.shape[-1] != 1920:
                break
            codes = self.mimi.encode(chunk)
            tokens = self.lm_gen.step(codes)
            if tokens is None:
                continue
            assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
            out_pcm = self.mimi.decode(tokens[:, 1:])
            out_text_tokens.append(tokens[:, 0])
            out_pcms.append(out_pcm)
            ntokens += 1
        dt = time.time() - start_time
        log("info", f"processed {ntokens} steps in {dt:.0f}s, {1000 * dt / ntokens:.2f}ms/step")
        out_pcms = torch.cat(out_pcms, dim=2)
        out_text_tokens = torch.cat(out_text_tokens, dim=1)
        return out_pcms, out_text_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults Moshiko. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size to be used for inference.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--lm-config", type=str, help="The LM config as a json file.")
    parser.add_argument("infile", type=str, help="Input audio file.")
    parser.add_argument("outfile", type=str, help="Output audio file in wav format.")

    args = parser.parse_args()
    seed_all(42424242)

    lm_kwargs = None
    num_codebooks = 8
    if args.lm_config is not None:
        log("info", f"loading config from {args.lm_config}")
        with open(args.lm_config, "r") as fobj:
            lm_kwargs = json.load(fobj)
            num_codebooks = lm_kwargs.get("dep_q", num_codebooks)

    log("info", "loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device, num_codebooks=num_codebooks)
    log("info", "mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    log("info", "loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, args.device, lm_kwargs=lm_kwargs)
    log("info", "moshi loaded")

    state = InferenceState(mimi, text_tokenizer, lm, args.batch_size, args.device)
    in_pcms, _ = sphn.read(args.infile, sample_rate=24000)
    in_pcms = torch.from_numpy(in_pcms).to(device=args.device)
    in_pcms = in_pcms[None, 0:1].expand(args.batch_size, -1, -1)
    out_pcms, out_text_tokens = state.run(in_pcms)
    log("info", f"out-pcm: {out_pcms.shape}, out-text: {out_text_tokens.shape}")
    if args.batch_size == 1:
        sphn.write_wav(args.outfile, out_pcms[0, 0].cpu().numpy(), sample_rate=24000)
    else:
        outfile = Path(args.outfile)
        for index in range(args.batch_size):
            outfile_ = outfile.with_name(f"{outfile.stem}-{index}{outfile.suffix}")
            log("info", f"writing {outfile_}")
            sphn.write_wav(str(outfile_), out_pcms[index, 0].cpu().numpy(), sample_rate=24000)


if __name__ == "__main__":
    with torch.no_grad():
        main()
