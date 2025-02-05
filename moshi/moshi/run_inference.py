# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import time

import numpy as np
import sentencepiece
import torch
import sphn


from .client_utils import log
from .conditioners import ConditionAttributes, ClassifierFreeGuidanceDropout
from .models import loaders, MimiModel, LMModel, LMGen


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
                 lm: LMModel, batch_size: int, cfg_coef: float, device: str | torch.device):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        condition_tensors = None
        if lm.condition_provider is not None:
            conditions = [ConditionAttributes(text={"description": "very_good"}, wav={})] * batch_size
            if cfg_coef != 1.:
                # Extending the conditions with the negatives for the CFG.
                conditions += ClassifierFreeGuidanceDropout(1.)(conditions)
            prepared = lm.condition_provider.prepare(conditions)
            condition_tensors = lm.condition_provider(prepared)
        self.lm_gen = LMGen(lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors)
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
        for i, chunk in enumerate(in_pcms.split(1920, dim=2)):
            if chunk.shape[-1] != 1920:
                break
            codes = self.mimi.encode(chunk)
            if i == 0:
                # Ensure that the first slice of codes is properly seen by the transformer
                # as otherwise the first slice is replaced by the initial tokens.
                tokens = self.lm_gen.step(codes)
                assert tokens is None
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
    parser.add_argument("--config", "--lm-config", dest="config", type=str, help="The config as a json file.")
    parser.add_argument("--cfg-coef", type=float, default=1., help="CFG coefficient.")
    parser.add_argument("infile", type=str, help="Input audio file.")
    parser.add_argument("outfile", type=str, help="Output audio file in wav format.")

    args = parser.parse_args()
    seed_all(42424242)

    log("info", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer, args.config)
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    log("info", "loading moshi")
    lm = checkpoint_info.get_moshi(device=args.device)
    log("info", "moshi loaded")

    log("info", f"loading input file {args.infile}")
    in_pcms, _ = sphn.read(args.infile, sample_rate=mimi.sample_rate)
    in_pcms = torch.from_numpy(in_pcms).to(device=args.device)
    in_pcms = in_pcms[None, 0:1].expand(args.batch_size, -1, -1)

    state = InferenceState(mimi, text_tokenizer, lm, args.batch_size, args.cfg_coef, args.device)
    out_pcms, out_text_tokens = state.run(in_pcms)
    log("info", f"out-pcm: {out_pcms.shape}, out-text: {out_text_tokens.shape}")

    if args.batch_size == 1:
        sphn.write_wav(args.outfile, out_pcms[0, 0].cpu().numpy(), sample_rate=mimi.sample_rate)
    else:
        outfile = Path(args.outfile)
        for index in range(args.batch_size):
            outfile_ = outfile.with_name(f"{outfile.stem}-{index}{outfile.suffix}")
            log("info", f"writing {outfile_}")
            sphn.write_wav(str(outfile_), out_pcms[index, 0].cpu().numpy(), sample_rate=24000)


if __name__ == "__main__":
    with torch.no_grad():
        main()
