# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Utility script to synthesize speech from a JSONL file containing the text and voice conditioning.
This uses Delayed Streams Modeling for TTS, as implemented in `moshi/models/tts.py`."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import sentencepiece
import sphn

from .client_utils import make_log
from . import models
from .utils.loaders import hf_get
from .models.tts import TTSModel, DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO


def log(level: str, msg: str):
    print(make_log(level, msg))


@dataclass
class TTSRequest:
    """Format for a single TTS request in the provided JSONL file.

    Args:
        turns: list of strings, one per turn, starting with the MAIN speaker.
            If you want to generate single turn, just put a single entry.
        voices: list of voice names, starting with MAIN. Put a single voice for single
            speaker generation. A corresponding file should exist in the voice repository.
        id: id that will be used for the output file. A wav file will be created, along
            with a `.safetensors` and `.json` file containing debug information.
        prefix: path to an audio file to use as prefix.
    """
    turns: list[str]
    voices: list[str]
    id: str


def main():
    parser = argparse.ArgumentParser(prog='moshi-tts', description='Run Moshi')
    parser.add_argument("--out-folder", type=Path, help="Output folder for TTSed files.",
                        default=Path('tts-outputs'))

    parser.add_argument("--hf-repo", type=str, default=DEFAULT_DSM_TTS_REPO,
                        help="HF repo in which to look for the pretrained models.")
    parser.add_argument("--voice-repo", default=DEFAULT_DSM_TTS_VOICE_REPO,
                        help="HF repo in which to look for pre-computed voice embeddings.")

    # The following flags are only to use a local checkpoint.
    parser.add_argument("--config", "--lm-config", dest="config", type=str, help="The config as a json file.")
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--mimi-weights", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--moshi-weights", type=str, help="Path to a local checkpoint file for Moshi.")

    # The following flags will customize generation.
    parser.add_argument("--nq", type=int, default=32, help="Number of codebooks to generate.")
    parser.add_argument("--temp", type=float, default=0.6, help="Temperature for text and audio.")
    parser.add_argument("--cfg-coef", type=float, default=2., help="CFG coefficient.")

    parser.add_argument("--quantize", type=int, help="The quantization to be applied, e.g. 8 for 8 bits.")
    parser.add_argument("--max-padding", type=int, default=8, help="Max padding in a row, in steps.")
    parser.add_argument("--initial-padding", type=int, default=2, help="Initial padding, in steps.")
    parser.add_argument("--final-padding", type=int, default=4, help="Amount of padding after the last word, in steps.")
    parser.add_argument("--padding-bonus", type=float, default=0.,
                        help="Bonus for the padding logits, should be between -2 and 2, "
                             "will change the speed of speech, with positive values being slower.")
    parser.add_argument("--padding-between", type=int, default=1,
                        help="Forces a minimal amount of fixed padding between words.")

    parser.add_argument("--only-wav", action='store_true',
                        help='Only save the audio. Otherwise, a .safetensors file with raw tokens is saved, '
                             'along with a .json file with various informations on the generation.')
    parser.add_argument("jsonl", type=Path, help="JSONL file containing the stuff to TTS.")

    args = parser.parse_args()
    assert args.jsonl.exists(), f"Not found: {args.jsonl}"
    args.out_folder.mkdir(parents=True, exist_ok=True)

    mx.random.seed(299792458)

    log("info", "retrieving checkpoints")

    raw_config = args.config
    if raw_config is None:
        raw_config = hf_get("config.json", args.hf_repo)

    log("info", f"loading config from {args.config}")
    with open(hf_get(raw_config), "r") as fobj:
        raw_config = json.load(fobj)

    mimi_weights = args.mimi_weights
    if mimi_weights is None:
        mimi_weights = hf_get(raw_config["mimi_name"], args.hf_repo)
    mimi_weights = hf_get(mimi_weights)

    moshi_weights = args.moshi_weights
    if moshi_weights is None:
        moshi_name = raw_config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_get(moshi_name, args.hf_repo)
    moshi_weights = hf_get(moshi_weights)

    tokenizer = args.tokenizer
    if tokenizer is None:
        tokenizer = hf_get(raw_config["tokenizer_name"], args.hf_repo)
    tokenizer = hf_get(tokenizer)

    lm_config = models.LmConfig.from_config_dict(raw_config)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)

    log("info", f"loading model weights from {moshi_weights}")
    model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

    if args.quantize is not None:
        log("info", f"quantizing model to {args.quantize} bits")
        nn.quantize(model.depformer, bits=args.quantize)
        for layer in model.transformer.layers:
            nn.quantize(layer.self_attn, bits=args.quantize)
            nn.quantize(layer.gating, bits=args.quantize)

    log("info", f"loading the text tokenizer from {tokenizer}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer))  # type: ignore

    log("info", f"loading the audio tokenizer {mimi_weights}")
    generated_codebooks = lm_config.generated_codebooks
    audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
    audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

    cfg_coef_conditioning = None
    tts_model = TTSModel(
        model,
        audio_tokenizer,
        text_tokenizer,
        voice_repo=args.voice_repo,
        n_q=args.nq,
        temp=args.temp,
        cfg_coef=args.cfg_coef,
        max_padding=args.max_padding,
        initial_padding=args.initial_padding,
        final_padding=args.final_padding,
        padding_bonus=args.padding_bonus,
        raw_config=raw_config,
    )
    if tts_model.valid_cfg_conditionings:
        # Model was trained with CFG distillation.
        cfg_coef_conditioning = tts_model.cfg_coef
        tts_model.cfg_coef = 1.
        cfg_is_no_text = False
        cfg_is_no_prefix = False
    else:
        cfg_is_no_text = True
        cfg_is_no_prefix = True
    mimi = tts_model.mimi

    def _flush():
        all_entries = []
        all_attributes = []
        prefixes = None
        if not tts_model.multi_speaker:
            prefixes = []
        begin = time.time()

        for request in batch:
            entries = tts_model.prepare_script(request.turns, padding_between=args.padding_between)
            all_entries.append(entries)
            if tts_model.multi_speaker:
                voices = [tts_model.get_voice_path(voice) for voice in request.voices]
            else:
                voices = []
            all_attributes.append(tts_model.make_condition_attributes(voices, cfg_coef_conditioning))
            if prefixes is not None:
                assert len(request.voices) == 1, "For this model, only exactly one voice is supported."
                prefix_path = hf_get(request.voices[0], args.voice_repo, check_local_file_exists=True)
                prefixes.append(tts_model.get_prefix(prefix_path))

        log("info", f"Starting batch of size {len(batch)}")
        result = tts_model.generate(
            all_entries, all_attributes, prefixes=prefixes,
            cfg_is_no_prefix=cfg_is_no_prefix, cfg_is_no_text=cfg_is_no_text)
        frames = mx.concat(result.frames, axis=-1)
        total_duration = frames.shape[0] * frames.shape[-1] / mimi.frame_rate
        time_taken = time.time() - begin
        total_speed = total_duration / time_taken
        log("info", f"[LM] Batch of size {len(batch)} took {time_taken:.2f}s, "
            f"total speed {total_speed:.2f}x")

        wav_frames = []
        for frame in result.frames:
            # We are processing frames one by one, although we could group them to improve speed.
            _pcm = tts_model.mimi.decode_step(frame)
            wav_frames.append(_pcm)
        wavs = mx.concat(wav_frames, axis=-1)
        effective_duration = 0.
        for idx, request in enumerate(batch):
            end_step = result.end_steps[idx]
            if end_step is None:
                log("warning", f"end step is None, generation failed for {request.id}")
                wav_length = wavs.shape[-1]
            else:
                wav_length = int((mimi.sample_rate * (end_step + tts_model.final_padding) / mimi.frame_rate))
            effective_duration += wav_length / mimi.sample_rate
            wav = wavs[idx, :, :wav_length]
            start_step = 0
            if prefixes is not None:
                start_step = prefixes[idx].shape[-1]
                start = int(mimi.sample_rate * start_step / mimi.frame_rate)
                wav = wav[:, start:]
            filename = args.out_folder / f"{request.id}.wav"
            debug_tensors = {'frames': frames[idx]}
            sphn.write_wav(filename, np.array(mx.clip(wav, -1, 1)), mimi.sample_rate)
            if not args.only_wav:
                mx.save_safetensors(str(filename.with_suffix('.safetensors')), debug_tensors)
                debug_info = {
                    'hf_repo': args.hf_repo,
                    'voice_repo': args.voice_repo,
                    'model_id': raw_config['model_id'],
                    'cfg_coef': tts_model.cfg_coef,
                    'temp': tts_model.temp,
                    'max_padding': tts_model.machine.max_padding,
                    'initial_padding': tts_model.machine.initial_padding,
                    'final_padding': tts_model.final_padding,
                    'padding_between': args.padding_between,
                    'padding_bonus': tts_model.padding_bonus,
                    'transcript': result.all_transcripts[idx],
                    'consumption_times': result.all_consumption_times[idx],
                    'turns': request.turns,
                    'voices': request.voices,
                    'logged_text_tokens': result.logged_text_tokens[idx],
                    'end_step': end_step,
                    'start_step': start_step,
                }
                with open(filename.with_suffix('.json'), 'w') as f:
                    json.dump(debug_info, f)
            log("info", f"saved {filename.absolute()}")
        time_taken = time.time() - begin
        total_speed = total_duration / time_taken
        effective_speed = effective_duration / time_taken
        # Total speed is the speed we get assuming all the item in the batch are the same length.
        # However, some items might have finished earlier, in which case the computation for those
        # was wasted. Effective speed accounts for that, and gives the speed up accounting for
        # the actual amount of usable audio generated.
        log("info", f"[TOT] Batch of size {len(batch)} took {time_taken:.2f}s, "
            f"total speed {total_speed:.2f}x, "
            f"effective speed {effective_speed:.2f}x")
        batch.clear()

    batch: list[TTSRequest] = []
    with args.jsonl.open('r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            batch.append(TTSRequest(**data))

            # We currently only support a batch size of 1 in the mlx implementation.
            _flush()
    if batch:
        _flush()


if __name__ == "__main__":
    main()
