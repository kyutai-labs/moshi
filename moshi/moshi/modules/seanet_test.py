import functools
import torch
import torch.nn as nn
import pytest

from .seanet import SEANetResnetBlock, SEANetDecoder


torch.backends.cudnn.enabled = False  # Disable cuDNN for deterministic behavior and for numerical stability


SEANET_RESNET_DATA = [
    # batch_size, dim, res_layer_index, seq_len, kernel_size
    pytest.param(
        3, 4, 1, 10, 6,
        id='small resnet test 1',
    ),
    pytest.param(
        4, 5, 2, 10, 7,
        id='small resnet test 2',
    ),
    pytest.param(
        5, 6, 4, 10, 2,
        id='small resnet test 3',
    ),
    pytest.param(
        1, 512, 2, 256, 7,
        id='large resnet test 1',
    ),
]
NUM_TIMESTEPS_DATA = [
    pytest.param(
        1,
        id='length 1',
    ),
    pytest.param(
        2,
        id='length 2',
    ),
    pytest.param(
        10,
        id='length 10',
    ),
    pytest.param(
        100,
        id='length 100',
    ),
]

SEANET_KWARGS_DATA = [
    pytest.param(
        {
            "channels": 1,
            "dimension": 8,
            "causal": True,
            "n_filters": 2,
            "n_residual_layers": 1,
            "activation": "ELU",
            "compress": 2,
            "dilation_base": 2,
            "disable_norm_outer_blocks": 0,
            "kernel_size": 7,
            "residual_kernel_size": 3,
            "last_kernel_size": 3,
            # We train using weight_norm but then the weights are pre-processed for inference so
            # that we can use a normal convolution.
            "norm": "none",
            "pad_mode": "constant",
            "ratios": [5],
            "true_skip": True,
        },
        id='Tiny SEANet',
    ),

    pytest.param(
        {
            "channels": 1,
            "dimension": 512,
            "causal": True,
            "n_filters": 64,
            "n_residual_layers": 1,
            "activation": "ELU",
            "compress": 2,
            "dilation_base": 2,
            "disable_norm_outer_blocks": 0,
            "kernel_size": 7,
            "residual_kernel_size": 3,
            "last_kernel_size": 3,
            # We train using weight_norm but then the weights are pre-processed for inference so
            # that we can use a normal convolution.
            "norm": "none",
            "pad_mode": "constant",
            "ratios": [8, 6, 5, 4],
            "true_skip": True,
        },
        id='Large SEANet',
    ),
]


def _init_weights(module, generator=None):
    for name, param in module.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param, generator=generator)
        elif "bias" in name:
            nn.init.constant_(param, 0.0)
        else:
            nn.init.xavier_uniform_(param, generator=generator)


@pytest.mark.parametrize("batch_size, dim, res_layer_index, seq_len, kernel_size", SEANET_RESNET_DATA)
def test_resnet(batch_size, dim, res_layer_index, seq_len, kernel_size):
    """Test that SEANetResnetBlock() calls are causal. Having new inputs does not change the previous output."""
    assert seq_len > kernel_size

    dilation_base = 2
    layer = SEANetResnetBlock(dim=dim, dilations=[dilation_base**res_layer_index, 1], pad_mode="constant", causal=True)

    generator = torch.Generator()
    generator = generator.manual_seed(41)
    layer.apply(functools.partial(_init_weights, generator=generator))

    shape = (batch_size, dim, seq_len,)
    input_hidden_states = torch.rand(shape)

    expected_output = layer(input_hidden_states)

    for end_index in range(kernel_size, seq_len + 1):
        actual_output = layer(input_hidden_states[..., :end_index])
        torch.testing.assert_close(actual_output, expected_output[..., :actual_output.shape[-1]],
                                   msg=lambda original_msg: f"Failed at end_index={end_index}: \n{original_msg}")


@pytest.mark.parametrize("batch_size, dim, res_layer_index, seq_len, kernel_size", SEANET_RESNET_DATA)
def test_resnet_streaming(batch_size, dim, res_layer_index, seq_len, kernel_size):
    """Test that SEANetResnetBlock() streaming works as expected."""
    assert seq_len > kernel_size

    dilation_base = 2
    layer = SEANetResnetBlock(dim=dim, dilations=[dilation_base**res_layer_index, 1], pad_mode="constant", causal=True)

    generator = torch.Generator()
    generator = generator.manual_seed(41)
    layer.apply(functools.partial(_init_weights, generator=generator))

    shape = (batch_size, dim, seq_len,)
    input_hidden_states = torch.rand(shape)

    expected_output = layer(input_hidden_states)

    start_index = 0
    actual_outputs = []
    with layer.streaming(batch_size=batch_size):
        for end_index in range(kernel_size, seq_len + 1):
            actual_output = layer(input_hidden_states[..., start_index:end_index])
            start_index = end_index
            actual_outputs.append(actual_output)
    actual_outputs = torch.cat(actual_outputs, dim=-1)

    torch.testing.assert_close(actual_outputs, expected_output)


@pytest.mark.parametrize("num_timesteps", NUM_TIMESTEPS_DATA)
@pytest.mark.parametrize("seanet_kwargs", SEANET_KWARGS_DATA)
def test_nonstreaming_causal_decode(num_timesteps, seanet_kwargs):
    """Test that the SEANetDecoder does not depend on future inputs."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    decoder = SEANetDecoder(**seanet_kwargs).to(device=device)

    generator = torch.Generator(device=device)
    generator = generator.manual_seed(41)
    decoder.apply(functools.partial(_init_weights, generator=generator))

    rand_generator = torch.Generator(device=device)
    rand_generator.manual_seed(2147483647)
    with torch.no_grad():
        # [B, K = 8, T]
        codes = torch.randn(1, seanet_kwargs['dimension'], num_timesteps, generator=rand_generator, device=device)
        expected_decoded = decoder(codes)

        num_timesteps = codes.shape[-1]
        for t in range(num_timesteps):
            current_codes = codes[..., :t + 1]
            actual_decoded = decoder(current_codes)
            torch.testing.assert_close(expected_decoded[..., :actual_decoded.shape[-1]], actual_decoded,
                                       msg=lambda original_msg: f"Failed at t={t}: \n{original_msg}")
