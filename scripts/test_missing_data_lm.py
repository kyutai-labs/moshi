"""Testing exec_mask feature of the streaming module, where each batch entry
can advance at its own pace, while retaining full compat with CUDA Graph."""
import sys
import sphn
import torch

from moshi.models import loaders
from moshi.models.lm import LMGen
from moshi.conditioners import ConditionAttributes

device = 'cuda'
wnp, sr = sphn.read(sys.argv[1],
                    start_sec=0, duration_sec=8, sample_rate=24000)
wav = torch.from_numpy(wnp)
ci = loaders.CheckpointInfo.from_hf_repo("kyutai/hibiki-2b-pytorch-bf16")
mimi = ci.get_mimi(device=device)
mimi.eval()

lm = ci.get_moshi(device=device)

B = 4

with torch.no_grad():
    codes = mimi.encode(wav[:1].to(device=device)[None])

T = codes.shape[-1]
offsets = [0 for _ in range(B)]

out_codes: list[list[torch.Tensor]] = [[] for _ in range(B)]
assert lm.condition_provider is not None
conditions = [ConditionAttributes(text={"description": "very_good"}, tensor={})] * B
prepared = lm.condition_provider.prepare(conditions)
condition_tensors = lm.condition_provider(prepared)
lm_gen = LMGen(lm, temp=0., temp_text=0., support_out_of_sync=True, condition_tensors=condition_tensors)
print("Going streaming")
with torch.no_grad(), lm_gen.streaming(B):
    while any(o < T for o in offsets):
        inputs = []
        exec_mask = torch.rand(B) < 0.5
        exec_mask[0] = True
        for offset in offsets:
            inputs.append(codes[:, :, min(offset, T - 1)])
        input_ = torch.cat(inputs)[..., None]
        lm_gen.set_exec_mask(exec_mask)
        pred = lm_gen.step(input_.to(device=device))
        assert pred is not None
        assert pred.shape[-1] == 1, pred.shape
        for b, active in enumerate(exec_mask.tolist()):
            if not active or offsets[b] >= T:
                continue
            if offsets[b] >= 2:
                assert (pred[b] >= 0).all()
            offsets[b] += 1
            out_codes[b].append(pred[b])

alls = []
for frames in out_codes:
    out = torch.cat(frames, -1)
    alls.append(out)

ys = torch.stack(alls)
r = ys[:1]
o = ys[1:]

ma = (r == o).float().mean(dim=(0, 1))
print(ma)
print(r[0, :2, :5])
print(o[0, :2, :5])
print(o[1, :2, :5])
print(o[2, :2, :5])
