import functools
import torch
import torch.nn as nn
import pytest

from .conv import StreamingConv1d, StreamingConvTranspose1d


torch.backends.cudnn.enabled = False  # Disable cuDNN for deterministic behavior and for numerical stability


CONV1D_DATA = [
    # batch_size, in_channels, out_channels, seq_len, kernel_size
    pytest.param(
        3, 4, 5, 10, 6,
        id='small conv1d test 1',
    ),
    pytest.param(
        4, 5, 6, 10, 7,
        id='small conv1d test 2',
    ),
    pytest.param(
        5, 6, 7, 10, 2,
        id='small conv1d test 3',
    ),
    pytest.param(
        1, 512, 512, 256, 7,
        id='large conv1d test 1',
    ),
]

CONV1D_TRANSPOSE_DATA = [
    # batch_size, in_channels, out_channels, seq_len, kernel_size, stride
    pytest.param(
        3, 4, 5, 10, 6, 1,
        id='small conv1d transpose test 1',
    ),
    pytest.param(
        4, 5, 6, 10, 7, 2,
        id='small conv1d transpose test 2',
    ),
    pytest.param(
        5, 6, 7, 10, 4, 3,
        id='small conv1d transpose test 3',
    ),
    pytest.param(
        1, 512, 512, 256, 7, 2,
        id='large conv1d transpose test 1',
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


@pytest.mark.parametrize("batch_size, in_channels, out_channels, seq_len, kernel_size", CONV1D_DATA)
def test_conv1d(batch_size, in_channels, out_channels, seq_len, kernel_size):
    """Test that StreamingConv1d() calls are causal. Having new inputs does not change the previous output."""
    assert seq_len > kernel_size

    layer = StreamingConv1d(in_channels, out_channels, kernel_size, causal=True, norm="none", pad_mode="constant")

    generator = torch.Generator()
    generator = generator.manual_seed(41)
    layer.apply(functools.partial(_init_weights, generator=generator))

    shape = (batch_size, in_channels, seq_len,)
    input_hidden_states = torch.rand(shape)

    expected_output = layer(input_hidden_states)

    for end_index in range(kernel_size, seq_len + 1):
        actual_output = layer(input_hidden_states[..., :end_index])
        torch.testing.assert_close(actual_output, expected_output[..., :actual_output.shape[-1]],
                                   msg=lambda original_msg: f"Failed at end_index={end_index}: \n{original_msg}")


@pytest.mark.parametrize("batch_size, in_channels, out_channels, seq_len, kernel_size", CONV1D_DATA)
def test_conv1d_streaming(batch_size, in_channels, out_channels, seq_len, kernel_size):
    """Test that StreamingConv1d() streaming works as expected."""
    assert seq_len > kernel_size

    layer = StreamingConv1d(in_channels, out_channels, kernel_size, causal=True, norm="none", pad_mode="constant")

    generator = torch.Generator()
    generator = generator.manual_seed(41)
    layer.apply(functools.partial(_init_weights, generator=generator))

    shape = (batch_size, in_channels, seq_len,)
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


@pytest.mark.parametrize("batch_size, in_channels, out_channels, seq_len, kernel_size, stride", CONV1D_TRANSPOSE_DATA)
def test_conv1d_transpose(batch_size, in_channels, out_channels, seq_len, kernel_size, stride):
    """Test that StreamingConvTranspose1d() calls are causal. Having new inputs does not change the previous output."""
    assert seq_len > kernel_size

    layer = StreamingConvTranspose1d(in_channels, out_channels, kernel_size, stride, causal=True, norm="none")

    generator = torch.Generator()
    generator = generator.manual_seed(41)
    layer.apply(functools.partial(_init_weights, generator=generator))

    shape = (batch_size, in_channels, seq_len,)
    input_hidden_states = torch.rand(shape)
    expected_output = layer(input_hidden_states)

    for end_index in range(kernel_size, seq_len + 1):
        actual_output = layer(input_hidden_states[..., :end_index])
        torch.testing.assert_close(actual_output, expected_output[..., :actual_output.shape[-1]],
                                   msg=lambda original_msg: f"Failed at end_index={end_index}: \n{original_msg}")


@pytest.mark.parametrize("batch_size, in_channels, out_channels, seq_len, kernel_size, stride", CONV1D_TRANSPOSE_DATA)
def test_conv1d_transpose_streaming(batch_size, in_channels, out_channels, seq_len, kernel_size, stride):
    """Test that StreamingConvTranspose1d() streaming works as expected."""
    assert seq_len > kernel_size

    layer = StreamingConvTranspose1d(in_channels, out_channels, kernel_size, stride, causal=True, norm="none")

    generator = torch.Generator()
    generator = generator.manual_seed(41)
    layer.apply(functools.partial(_init_weights, generator=generator))

    shape = (batch_size, in_channels, seq_len,)
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
