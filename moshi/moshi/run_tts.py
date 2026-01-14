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

from safetensors.torch import save_file
import sphn
import torch

from .models.tts import TTSModel, DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO
from .models.loaders import CheckpointInfo


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
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")

    # The following flags will customize generation.
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size to be used for inference.")
    parser.add_argument("--nq", type=int, default=32, help="Number of codebooks to generate.")
    parser.add_argument("--temp", type=float, default=0.6, help="Temperature for text and audio.")
    parser.add_argument("--cfg-coef", type=float, default=2., help="CFG coefficient.")

    parser.add_argument("--max-padding", type=int, default=8, help="Max padding in a row, in steps.")
    parser.add_argument("--initial-padding", type=int, default=2, help="Initial padding, in steps.")
    parser.add_argument("--final-padding", type=int, default=4, help="Amount of padding after the last word, in steps.")
    parser.add_argument("--padding-bonus", type=float, default=0.,
                        help="Bonus for the padding logits, should be between -2 and 2, "
                             "will change the speed of speech, with positive values being slower.")
    parser.add_argument("--padding-between", type=int, default=1,
                        help="Forces a minimal amount of fixed padding between words.")

    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--half", action="store_const", const=torch.float16, default=torch.bfloat16,
                        dest="dtype", help="Run inference with float16, not bfloat16, better for old GPUs.")

    parser.add_argument("--only-wav", action='store_true',
                        help='Only save the audio. Otherwise, a .safetensors file with raw tokens is saved, '
                             'along with a .json file with various informations on the generation.')
    parser.add_argument("jsonl", type=Path, help="JSONL file containing the stuff to TTS.")

    args = parser.parse_args()
    assert args.jsonl.exists(), f"Not found: {args.jsonl}"
    args.out_folder.mkdir(parents=True, exist_ok=True)

    print("retrieving checkpoint")
    checkpoint_info = CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer, args.config)

    cfg_coef_conditioning = None
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info, voice_repo=args.voice_repo, n_q=args.nq, temp=args.temp, cfg_coef=args.cfg_coef,
        max_padding=args.max_padding, initial_padding=args.initial_padding, final_padding=args.final_padding,
        padding_bonus=args.padding_bonus, device=args.device, dtype=args.dtype)
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
                prefix_path = tts_model.get_voice_path(request.voices[0])
                prefixes.append(tts_model.get_prefix(prefix_path))

        print(f"Starting batch of size {len(batch)}")
        result = tts_model.generate(
            all_entries, all_attributes, prefixes=prefixes,
            cfg_is_no_prefix=cfg_is_no_prefix, cfg_is_no_text=cfg_is_no_text)
        frames = torch.cat(result.frames, dim=-1).cpu()
        total_duration = frames.shape[0] * frames.shape[-1] / mimi.frame_rate
        time_taken = time.time() - begin
        total_speed = total_duration / time_taken
        print(f"[LM] Batch of size {len(batch)} took {time_taken:.2f}s, "
              f"total speed {total_speed:.2f}x")

        wav_frames = []
        with torch.no_grad(), tts_model.mimi.streaming(len(all_entries)):
            for frame in result.frames[tts_model.delay_steps:]:
                # We are processing frames one by one, although we could group them to improve speed.
                wav_frames.append(tts_model.mimi.decode(frame[:, 1:]))
        wavs = torch.cat(wav_frames, dim=-1)
        effective_duration = 0.
        for idx, request in enumerate(batch):
            end_step = result.end_steps[idx]
            if end_step is None:
                print(f"Warning: end step is None, generation failed for {request.id}")
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
            debug_tensors = {
                'frames': frames[idx].short(),
            }
            sphn.write_wav(filename, wav.clamp(-1, 1).cpu().numpy(), mimi.sample_rate)
            if not args.only_wav:
                save_file(debug_tensors, filename.with_suffix('.safetensors'))
                debug_info = {
                    'hf_repo': args.hf_repo,
                    'voice_repo': args.voice_repo,
                    'model_id': checkpoint_info.model_id,
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
            print("Saved", filename)
        time_taken = time.time() - begin
        total_speed = total_duration / time_taken
        effective_speed = effective_duration / time_taken
        # Total speed is the speed we get assuming all the item in the batch are the same length.
        # However, some items might have finished earlier, in which case the computation for those
        # was wasted. Effective speed accounts for that, and gives the speed up accounting for
        # the actual amount of usable audio generated.
        print(f"[TOT] Batch of size {len(batch)} took {time_taken:.2f}s, "
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

            if len(batch) >= args.batch_size:
                _flush()
    if batch:
        _flush()


if __name__ == "__main__":
    main()
