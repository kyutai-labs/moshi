# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import time

import numpy as np
import sentencepiece
import torch
import sphn


from .client_utils import log, AnyPrinter, Printer, RawPrinter
from .conditioners import ConditionAttributes, ConditionTensors
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


def get_condition_tensors(
    model_type: str, lm: LMModel, batch_size: int, cfg_coef: float
) -> ConditionTensors:
    condition_tensors = {}
    if lm.condition_provider is not None and lm.condition_provider.conditioners:
        conditions: list[ConditionAttributes] | None = None
        if model_type == "hibiki":
            conditions = [
                ConditionAttributes(text={"description": "very_good"}, tensor={})
                for _ in range(batch_size)
            ]
            if cfg_coef != 1.0:
                # Extending the conditions with the negatives for the CFG.
                conditions += [
                    ConditionAttributes(text={"description": "very_bad"}, tensor={})
                    for _ in range(batch_size)
                ]
        else:
            raise RuntimeError(
                f"Model expects conditioning but model type {model_type} is not supported."
            )
        assert conditions is not None
        prepared = lm.condition_provider.prepare(conditions)
        condition_tensors = lm.condition_provider(prepared)
    return condition_tensors


@dataclass
class InferenceState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen

    def __init__(
        self,
        checkpoint_info: loaders.CheckpointInfo,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        batch_size: int,
        cfg_coef: float,
        device: str | torch.device,
        **kwargs,
    ):
        self.checkpoint_info = checkpoint_info
        model_type = checkpoint_info.model_type
        self.model_type = model_type
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        condition_tensors = get_condition_tensors(model_type, lm, batch_size, cfg_coef)
        self.lm_gen = LMGen(
            lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **kwargs
        )
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.batch_size = batch_size
        self.mimi.streaming_forever(batch_size)
        self.lm_gen.streaming_forever(batch_size)
        self.printer: AnyPrinter
        if sys.stdout.isatty():
            self.printer = Printer()
        else:
            self.printer = RawPrinter()

    def run(self, in_pcms: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Returns a list of tupel `(text_tokens, audio_tokens)`"""
        out_pcms_per_item: list[list[torch.Tensor]] = [
            [] for _ in range(self.batch_size)
        ]
        out_text_tokens_per_item: list[list[torch.Tensor]] = [
            [] for _ in range(self.batch_size)
        ]
        # For the Hibiki translation model, we feed a special token for the end of the input stream,
        # which corresponds to `2048` on all the codebooks of the audio stream, and wait
        # for the EOS on the output text stream to be emitted, as indication that the model is done.
        eos_reached: list[bool] = [False] * self.batch_size
        need_eos_input: bool = True
        self.printer.log(
            "info",
            "starting inference, "
            f"sampling: {self.lm_gen.use_sampling}, "
            f"audio temp: {self.lm_gen.temp}, "
            f"text temp: {self.lm_gen.temp_text}",
        )
        device = self.lm_gen.lm_model.device
        start_time = time.time()
        ntokens = 0
        first_frame = True
        if self.model_type == "stt":
            stt_config = self.checkpoint_info.stt_config
            pad_right = stt_config.get("audio_delay_seconds", 0.0)
            pad_left = stt_config.get("audio_silence_prefix_seconds", 0.0)
            pad_left = int(pad_left * 24000)
            pad_right = int((pad_right + 1.0) * 24000)
            in_pcms = torch.nn.functional.pad(in_pcms, (pad_left, pad_right), mode="constant")
        # We keep only fully frames.
        chunks = deque(
            [
                chunk
                for chunk in in_pcms.split(self.frame_size, dim=2)
                if chunk.shape[-1] == self.frame_size
            ]
        )

        self.printer.print_header()
        while not all(eos_reached):
            if chunks:
                chunk = chunks.popleft()
                codes = self.mimi.encode(chunk)
            else:
                if self.model_type == "hibiki":
                    if need_eos_input:
                        # First frame after the end of the file, we feed a code full of 2048
                        # to indicate the end of stream.
                        need_eos_input = False
                        eos_value = self.mimi.cardinality
                        codes = torch.full(
                            (self.batch_size, self.mimi.num_codebooks, 1),
                            eos_value,
                            device=device,
                            dtype=torch.long,
                        )
                    else:
                        silence = torch.zeros(
                            (self.batch_size, self.mimi.channels, self.frame_size),
                            device=device,
                        )
                        codes = self.mimi.encode(silence)
                else:
                    # For other models, we stop as soon as we are reaching the end of the audio.
                    break
            if first_frame:
                # Ensure that the first slice of codes is properly seen by the transformer
                # as otherwise the first slice is replaced by the initial tokens.
                tokens = self.lm_gen.step(codes)
                if max(self.lm_gen.lm_model.delays) > 0:
                    assert tokens is None
                first_frame = False
            tokens = self.lm_gen.step(codes)
            if tokens is None:
                continue
            assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
            if self.lm_gen.lm_model.dep_q > 0:
                out_pcm = self.mimi.decode(tokens[:, 1:]).cpu()
                for b, (one_text, one_pcm) in enumerate(
                    zip(tokens[:, 0].cpu(), out_pcm)
                ):
                    if eos_reached[b]:
                        continue
                    elif one_text.item() == self.text_tokenizer.eos_id():
                        if need_eos_input:
                            # We sampled the EOS before the end of the file! Not possible.
                            self.printer.log("warning", "EOS sampled too early.")
                        else:
                            eos_reached[b] = True

                    out_text_tokens_per_item[b].append(one_text)
                    out_pcms_per_item[b].append(one_pcm)
                    if b == 0:
                        if one_text.item() not in [0, 3]:
                            text = self.text_tokenizer.id_to_piece(one_text.item())  # pyright: ignore
                            text = text.replace("▁", " ")
                            self.printer.print_token(text)
            else:
                one_text = tokens[0, 0].cpu()
                if one_text.item() not in [0, 3]:
                    text = self.text_tokenizer.id_to_piece(one_text.item())  # pyright: ignore
                    text = text.replace("▁", " ")
                    self.printer.print_token(text)
            ntokens += 1
        dt = time.time() - start_time
        self.printer.log(
            "info",
            f"processed {ntokens} steps in {dt:.0f}s, {1000 * dt / ntokens:.2f}ms/step",
        )
        if self.lm_gen.lm_model.dep_q > 0:
            out = [
                (torch.cat(one_texts, dim=0), torch.cat(one_pcms, dim=1))
                for one_texts, one_pcms in zip(
                    out_text_tokens_per_item, out_pcms_per_item
                )
            ]
            return out
        else:
            return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument(
        "--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi."
    )
    parser.add_argument(
        "--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi."
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo to look into, defaults Moshiko. "
        "Use this to select a different pre-trained model.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size to be used for inference."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run, defaults to 'cuda'.",
    )
    parser.add_argument(
        "--half",
        action="store_const",
        const=torch.float16,
        default=torch.bfloat16,
        dest="dtype",
        help="Run inference with float16, not bfloat16, better for old GPUs.",
    )
    parser.add_argument(
        "--config",
        "--lm-config",
        dest="config",
        type=str,
        help="The config as a json file.",
    )
    parser.add_argument("--cfg-coef", type=float, default=1.0, help="CFG coefficient.")
    parser.add_argument("infile", type=str, help="Input audio file.")
    parser.add_argument(
        "outfile",
        type=str,
        help="Output audio file in wav format.",
        nargs="?",
        default="",
    )

    args = parser.parse_args()
    seed_all(4242)

    log("info", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer, args.config
    )
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    log("info", "loading moshi")
    lm = checkpoint_info.get_moshi(device=args.device, dtype=args.dtype)
    log("info", "moshi loaded")
    if lm.dep_q == 0:
        args.batch_size = 1

    log("info", f"loading input file {args.infile}")
    in_pcms, _ = sphn.read(args.infile, sample_rate=mimi.sample_rate)
    in_pcms = torch.from_numpy(in_pcms).to(device=args.device)
    in_pcms = in_pcms[None, 0:1].expand(args.batch_size, -1, -1)

    state = InferenceState(
        checkpoint_info,
        mimi,
        text_tokenizer,
        lm,
        args.batch_size,
        args.cfg_coef,
        args.device,
        **checkpoint_info.lm_gen_config,
    )
    out_items = state.run(in_pcms)

    if args.outfile:
        outfile = Path(args.outfile)
        for index, (_, out_pcm) in enumerate(out_items):
            if len(out_items) > 1:
                outfile_ = outfile.with_name(f"{outfile.stem}-{index}{outfile.suffix}")
            else:
                outfile_ = outfile
            duration = out_pcm.shape[1] / mimi.sample_rate
            log("info", f"writing {outfile_} with duration {duration:.1f} sec.")
            sphn.write_wav(
                str(outfile_), out_pcm[0].numpy(), sample_rate=mimi.sample_rate
            )


if __name__ == "__main__":
    with torch.no_grad():
        main()
