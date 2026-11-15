import pytest
import torch
from torch import nn

from moshi.modules.lora import (
    LoRALinear,
    replace_all_linear_with_lora,
    replace_lora_with_linear,
)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("zero_update", [False, True])
def test_recursive_merge_preserves_outputs_and_bias(bias, dtype, zero_update):
    torch.manual_seed(123)
    model = nn.Sequential(
        nn.Linear(3, 4, bias=bias, dtype=dtype), nn.Sequential(nn.Linear(4, 2, bias=bias, dtype=dtype))
    )
    replace_all_linear_with_lora(model, rank=2, scaling=0.75)
    saved_biases = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if zero_update:
                with torch.no_grad():
                    module.lora_B.weight.zero_()
            if module.frozen_W.bias is not None:
                module.frozen_W.bias.requires_grad_(not zero_update)
                saved_biases[name] = module.frozen_W.bias
    inputs = torch.randn(5, 3, dtype=dtype)
    before = model(inputs).detach()
    replace_lora_with_linear(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    assert torch.allclose(model(inputs), before, atol=tolerance, rtol=0)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert module.weight.dtype == dtype
            assert module.weight.device == inputs.device
            if bias:
                assert torch.equal(module.bias, saved_biases[name])
                assert module.bias.data_ptr() != saved_biases[name].data_ptr()
                assert module.bias.requires_grad == saved_biases[name].requires_grad
            else:
                assert module.bias is None


def test_zero_delta_keeps_nonzero_bias_exactly():
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(torch.diag(torch.tensor([1.0, 2.0])))
        linear.bias.copy_(torch.tensor([3.0, 7.0]))
    model = nn.Sequential(linear)
    replace_all_linear_with_lora(model, 1, 1)
    with torch.no_grad():
        model[0].lora_B.weight.zero_()
    replace_lora_with_linear(model)
    assert torch.equal(model(torch.ones(1, 2)), torch.tensor([[4.0, 9.0]]))
