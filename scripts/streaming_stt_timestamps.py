"""An example script that illustrates how one can get per-word timestamps from
Kyutai STT models.

Usage:
```
uv run scripts/streaming_stt_timestamps.py \
    --hf-repo kyutai/stt-2.6b-en \
    --file bria.mp3
```

"""

import itertools
import dataclasses
import julius
import sphn
import argparse
import math

import torch
import moshi.models
import tqdm


@dataclasses.dataclass
class TimestampedText:
    text: str
    timestamp: tuple[float, float]

    def __str__(self):
        return f"{self.text} ({self.timestamp[0]:.2f}:{self.timestamp[1]:.2f})"


def tokens_to_timestamped_text(
    text_tokens,
    tokenizer,
    frame_rate,
    end_of_padding_id,
    padding_token_id,
    offset_seconds,
) -> list[TimestampedText]:
    text_tokens = text_tokens.cpu().view(-1)

    # Normally `end_of_padding` tokens indicate word boundaries.
    # Everything between them should be a single word;
    # the time offset of the those tokens correspond to word start and
    # end timestamps (minus silence prefix and audio delay).
    #
    # However, in rare cases some complexities could arise. Firstly,
    # for words that are said quickly but are represented with
    # multiple tokens, the boundary might be omitted. Secondly,
    # for the very last word the end boundary might not happen.
    # Below is a code snippet that handles those situations a bit
    # more carefully.

    sequence_timestamps = []

    def _tstmp(start_position, end_position):
        return (
            max(0, start_position / frame_rate - offset_seconds),
            max(0, end_position / frame_rate - offset_seconds),
        )

    def _decode(t):
        t = t[t > padding_token_id]
        return tokenizer.decode(t.numpy().tolist())

    def _decode_segment(start, end):
        nonlocal text_tokens
        nonlocal sequence_timestamps

        text = _decode(text_tokens[start:end])
        words_inside_segment = text.split()

        if len(words_inside_segment) == 0:
            return
        if len(words_inside_segment) == 1:
            # Single word within the boundaries, the general case
            sequence_timestamps.append(
                TimestampedText(text=text, timestamp=_tstmp(start, end))
            )
        else:
            # We're in a rare situation where multiple words are so close they are not separated by `end_of_padding`.
            # We tokenize words one-by-one; each word is assigned with as many frames as much tokens it has.
            for adjacent_word in words_inside_segment[:-1]:
                n_tokens = len(tokenizer.encode(adjacent_word))
                sequence_timestamps.append(
                    TimestampedText(
                        text=adjacent_word, timestamp=_tstmp(start, start + n_tokens)
                    )
                )
                start += n_tokens

            # The last word takes everything until the boundary
            adjacent_word = words_inside_segment[-1]
            sequence_timestamps.append(
                TimestampedText(text=adjacent_word, timestamp=_tstmp(start, end))
            )

    (segment_boundaries,) = torch.where(text_tokens == end_of_padding_id)

    if not segment_boundaries.numel():
        return []

    for i in range(len(segment_boundaries) - 1):
        segment_start = int(segment_boundaries[i]) + 1
        segment_end = int(segment_boundaries[i + 1])

        _decode_segment(segment_start, segment_end)

    last_segment_start = segment_boundaries[-1] + 1

    boundary_token = torch.tensor([tokenizer.eos_id()])
    (end_of_last_segment,) = torch.where(
        torch.isin(text_tokens[last_segment_start:], boundary_token)
    )

    if not end_of_last_segment.numel():
        # upper-bound either end of the audio or 1 second duration, whicher is smaller
        last_segment_end = min(text_tokens.shape[-1], last_segment_start + frame_rate)
    else:
        last_segment_end = last_segment_start + end_of_last_segment[0]
    _decode_segment(last_segment_start, last_segment_end)

    return sequence_timestamps


def main(args):
    info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        moshi_weights=args.moshi_weight,
        mimi_weights=args.mimi_weight,
        tokenizer=args.tokenizer,
        config_path=args.config_path,
    )

    mimi = info.get_mimi(device=args.device)
    tokenizer = info.get_text_tokenizer()
    lm = info.get_moshi(
        device=args.device,
        dtype=torch.bfloat16,
    )
    lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)

    audio_silence_prefix_seconds = info.stt_config.get(
        "audio_silence_prefix_seconds", 1.0
    )
    audio_delay_seconds = info.stt_config.get("audio_delay_seconds", 5.0)
    padding_token_id = info.raw_config.get("text_padding_token_id", 3)

    audio, input_sample_rate = sphn.read(args.file)
    audio = torch.from_numpy(audio).to(args.device)
    audio = julius.resample_frac(audio, input_sample_rate, mimi.sample_rate)
    if audio.shape[-1] % mimi.frame_size != 0:
        to_pad = mimi.frame_size - audio.shape[-1] % mimi.frame_size
        audio = torch.nn.functional.pad(audio, (0, to_pad))

    text_tokens_accum = []

    n_prefix_chunks = math.ceil(audio_silence_prefix_seconds * mimi.frame_rate)
    n_suffix_chunks = math.ceil(audio_delay_seconds * mimi.frame_rate)
    silence_chunk = torch.zeros(
        (1, 1, mimi.frame_size), dtype=torch.float32, device=args.device
    )

    chunks = itertools.chain(
        itertools.repeat(silence_chunk, n_prefix_chunks),
        torch.split(audio[:, None], mimi.frame_size, dim=-1),
        itertools.repeat(silence_chunk, n_suffix_chunks),
    )

    with mimi.streaming(1), lm_gen.streaming(1):
        for audio_chunk in tqdm.tqdm(chunks):
            audio_tokens = mimi.encode(audio_chunk)
            text_tokens = lm_gen.step(audio_tokens)
            if text_tokens is not None:
                text_tokens_accum.append(text_tokens)

    utterance_tokens = torch.concat(text_tokens_accum, dim=-1)
    timed_text = tokens_to_timestamped_text(
        utterance_tokens,
        tokenizer,
        mimi.frame_rate,
        end_of_padding_id=0,
        padding_token_id=padding_token_id,
        offset_seconds=int(n_prefix_chunks / mimi.frame_rate) + audio_delay_seconds,
    )

    decoded = " ".join([str(t) for t in timed_text])
    print(decoded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example streaming STT w/ timestamps.")
    parser.add_argument(
        "--file",
        required=True,
        help="File to transcribe.",
    )

    parser.add_argument(
        "--hf-repo", type=str, help="HF repo to load the STT model from. "
    )
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument(
        "--moshi-weight", type=str, help="Path to a local checkpoint file."
    )
    parser.add_argument(
        "--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi."
    )
    parser.add_argument(
        "--config-path", type=str, help="Path to a local config file.", default=None
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run, defaults to 'cuda'.",
    )
    args = parser.parse_args()

    main(args)
