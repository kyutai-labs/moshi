# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

try:
    import IPython.display as ipd  # type: ignore
except ImportError:
    # Note in a notebook...
    pass


import torch
from ..data.speech_dataset import Alignment
from ..solvers.musicgen import MusicGenSolver


def display_audio(samples: torch.Tensor, sample_rate: int):
    """Renders an audio player for the given audio samples.

    Args:
        samples (torch.Tensor): a Tensor of decoded audio samples
            with shapes [B, C, T] or [C, T]
        sample_rate (int): sample rate audio should be displayed with.
    """
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for audio in samples:
        ipd.display(ipd.Audio(audio, rate=sample_rate))


def display_interleaved(solver: MusicGenSolver, k: int, tokens: torch.Tensor):
    assert solver.moshi_text_lm is not None
    offset: int = 0
    T = tokens.shape[-1]
    zero = solver.model.zero_token_id
    pad = solver.model.text_padding_token_id
    end_pad = solver.model.end_of_text_padding_id
    tokens = tokens[k].data.cpu()
    text = tokens[0].tolist()
    first_audio = tokens[1].tolist()
    EOS = solver.moshi_text_lm.tokenizer.eos_id()
    BOS = solver.moshi_text_lm.tokenizer.bos_id()

    is_main = False
    tasks = []
    for offset in range(T):
        if text[offset] == BOS:
            is_main = True
        if text[offset] == zero:
            if first_audio[offset] == zero:
                task = 'e'
            else:
                task = 'a'
        elif first_audio[offset] == zero:
            task = 't'
        else:
            task = 'b'
        if is_main:
            task = task.upper()
        tasks.append(task)
        if text[offset] == EOS:
            is_main = False
    print(''.join(tasks))

    offset = 0
    while offset < T:
        task = tasks[offset]
        for end in range(offset, T + 1):
            if end == T:
                break
            if tasks[end] != task:
                break
        has_audio = task.lower() in ['a', 'b']
        has_text = task.lower() in ['b', 't']
        is_main = task.upper() == task
        chunk = tokens[:, offset: end]
        if has_text:
            text_tokens = chunk[0][(chunk[0] != pad) & (chunk[0] != end_pad)].tolist()
            if text_tokens:
                decoded_text = solver.moshi_text_lm.tokenizer.decode(text_tokens)
            else:
                decoded_text = ''
            if is_main is True:
                extra = '(main)'
            else:
                extra = '(other)'
            print(f"Text{extra}@{offset}:", decoded_text)
        elif has_audio:
            print(f"Audio@{offset}")
        else:
            print(f"Empty@{offset}")
        if has_audio:
            assert chunk[None, 1:].min() >= 0
            assert chunk[None, 1:].max() < solver.compression_model.cardinality
            with torch.no_grad():
                w = solver.compression_model.decode(chunk[None, 1:].to(solver.device), None)
            display_audio(w, 24000)
        offset = end


def display_alignment(aligment: list[Alignment],
                      new_line_on_change: bool = False, timestamps: bool = True) -> None:
    """Display the alignment from audio_ds.read_meta_and_transcribe.

    Args:
        alignment (list[Alignment]): the alignment to display.
        new_line_on_change (bool): whether to add a new line when the speaker changes.
        timestamps (bool): whether to display timestamps (only every second max)."""
    nxt = 1
    lst = None
    for word, ts, spk in aligment:
        if lst != spk:
            b = '<M>' if spk == 'SPEAKER_MAIN' else '<O>'
            if new_line_on_change:
                b = '\n' + b
            print(' ' + b, end='')
            lst = spk
        if ts[0] > nxt and timestamps:
            print(f' [{ts[0]:.1f}]', end='')
            nxt = nxt + 1
        print(word, end='')
    print()
