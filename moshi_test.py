import argparse
import msh
import sentencepiece
import torch
import torchaudio
import numpy as np
import random

SAMPLE_RATE = msh.models.moshi.SAMPLE_RATE
DEVICE = "cuda:0"
ENABLE_PROFILING = False

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--moshi-weights", type=str)
parser.add_argument("--mimi-weights", type=str)
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
ec = msh.models.moshi.get_encodec(args.mimi_weights, DEVICE)
print("mimi loaded")
text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

print("loading moshi")
lm = msh.models.moshi.get_lm(args.moshi_weights, DEVICE)
print("lm loaded")


def cb(step, total):
    print(f"{step:06d} / {total:06d}", end="\r")


def streaming_test():
    lm.reset_streaming()
    max_gen_len = 256
    with torch.no_grad():
        lm_gen = msh.models.LMGen(lm, check=True, max_gen_len=max_gen_len)
        tokens = [lm_gen.ungenerated] * 17
        main_audio = []
        other_audio = []
        for _step in range(max_gen_len):
            tokens = lm_gen.step(tokens)
            main_audio.append(tokens[1:9])
            other_audio.append(tokens[9:])
            text_token = tokens[0]
            if text_token not in (0, 3):
                _text = text_tokenizer.id_to_piece(text_token)
                _text = _text.replace("▁", " ")
                print(_text, end="", flush=True)
        print()
        main_audio = torch.tensor(main_audio).to(device=DEVICE)
        other_audio = torch.tensor(other_audio).to(device=DEVICE)
        print(main_audio.shape, other_audio.shape)
        all_codes = torch.stack([main_audio, other_audio], dim=0).transpose(1, 2)
        print(all_codes.shape)
        print(all_codes)
        # Discard the two first slices.
        pcm = ec.decode(all_codes[:, :, 2:])
        print("pcm", pcm.shape)
        torchaudio.save("gen_main.wav", pcm[0].cpu(), SAMPLE_RATE)
        torchaudio.save("gen_other.wav", pcm[1].cpu(), SAMPLE_RATE)


print("streaming test")
streaming_test()
