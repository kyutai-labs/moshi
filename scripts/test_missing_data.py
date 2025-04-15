"""Testing exec_mask feature of the streaming module, where each batch entry
can advance at its own pace, while retaining full compat with CUDA Graph."""
import sys
import sphn
import torch

from moshi.models import loaders

device = 'cuda'
wnp, sr = sphn.read(sys.argv[1],
                    start_sec=0, duration_sec=8, sample_rate=24000)
wav = torch.from_numpy(wnp)
ci = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
mimi = ci.get_mimi(device=device)
mimi.eval()

frame = int(mimi.sample_rate / mimi.frame_rate)

total = wav.shape[-1] // frame
B = 4
wav = wav[..., :total * frame][None]
remaining = [total for _ in range(B)]

print("Ref computation")
with torch.no_grad():
    ref_codes = mimi.encode(wav[:1].to(device=device))
    ref_audio = mimi.decode(ref_codes[:1].to(device=device))

out_audio = torch.zeros_like(ref_audio.expand(B, -1, -1))
out_codes = torch.zeros_like(ref_codes.expand(B, -1, -1))

print("Going streaming")
with mimi.streaming(B), torch.no_grad():
    while any(remaining):
        inputs = []
        exec_mask = torch.rand(B) < 0.5
        offsets = []
        for b, this_remaining in enumerate(remaining):
            offset = min(total - this_remaining, total - 1)
            offsets.append(offset)
            inputs.append(wav[0, :, offset * frame: offset * frame + frame])
        input_ = torch.stack(inputs)
        mimi.set_exec_mask(exec_mask)
        codes = mimi.encode(input_.to(device=device))
        assert codes.shape[-1] == 1, codes.shape
        w = mimi.decode(codes)
        assert w.shape[-1] == frame, w.shape
        print(remaining)
        for b, active in enumerate(exec_mask.tolist()):
            if not active or remaining[b] == 0:
                continue
            remaining[b] = max(0, remaining[b] - 1)
            offset = offsets[b]
            out_codes[b, :, offset: offset + 1] = codes[b]
            out_audio[b, :, offset * frame: offset * frame + frame] = w[b]

print(ref_codes[0, :, -1])
print(out_codes[0, :, -1])
for b in range(B):
    print((out_codes[b] != ref_codes[:1]).any(dim=0).nonzero()[:1])
d = (out_codes[..., :1] == ref_codes[..., :1]).float().mean(dim=(1, 2))
print(d)
d = (out_codes[..., :2] == ref_codes[..., :2]).float().mean(dim=(1, 2))
print(d)
d = (out_codes[..., :10] == ref_codes[..., :10]).float().mean(dim=(1, 2))
print(d)
d = (out_codes == ref_codes).float().mean(dim=(1, 2))
print(d)
d = (out_audio - ref_audio).norm(dim=-1, p=2) / ref_audio.norm(dim=-1, p=2)
print(d.sum(dim=1))
