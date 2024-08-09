import msh

SAMPLE_RATE = 24000
FRAME_RATE = 12.5

seanet_kwargs = {
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
    "lstm": 0,
    "norm": "weight_norm",
    "pad_mode": "Constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": seanet_kwargs["dimension"],
    "output_dimension": seanet_kwargs["dimension"],
}
transformer_kwargs = {
    "d_model": seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "cross_attention": False,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "conv_layout": True,
    "input_dimension": seanet_kwargs["dimension"],
    "output_dimensions": [seanet_kwargs["dimension"]],
}


encoder = msh.modules.SEANetEncoder(**seanet_kwargs)
decoder = msh.modules.SEANetDecoder(**seanet_kwargs)
encoder_transformer = msh.modules.transformer.ProjectedTransformer(**transformer_kwargs)
decoder_transformer = msh.modules.transformer.ProjectedTransformer(**transformer_kwargs)
quantizer = msh.quantization.SplitResidualVectorQuantizer()
model = msh.models.EncodecModel(
    encoder,
    decoder,
    quantizer,
    channels=1,
    sample_rate=SAMPLE_RATE,
    frame_rate=FRAME_RATE,
    encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
    renormalize=True,
    encoder_transformer=encoder_transformer,
    decoder_transformer=decoder_transformer,
)
