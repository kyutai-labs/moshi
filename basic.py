import msh
import sentencepiece
import torch
import torchaudio
import safetensors

SAMPLE_RATE = 24000
FRAME_RATE = 12.5
DEVICE = "cuda:0"

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
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
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
    "conv_layout": True,
    "cross_attention": False,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": seanet_kwargs["dimension"],
    "output_dimensions": [seanet_kwargs["dimension"]],
}

lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "card": quantizer_kwargs["bins"],
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "cross_attention": False,
    "gating": "silu",
    "norm": "rms_norm",
    "positional_embedding": "rope",
    "depformer": bool,
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_cross_attention": False,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
}


def get_encodec():
    encoder = msh.modules.SEANetEncoder(**seanet_kwargs)
    decoder = msh.modules.SEANetDecoder(**seanet_kwargs)
    encoder_transformer = msh.modules.transformer.ProjectedTransformer(
        **transformer_kwargs
    )
    decoder_transformer = msh.modules.transformer.ProjectedTransformer(
        **transformer_kwargs
    )
    quantizer = msh.quantization.SplitResidualVectorQuantizer(
        **quantizer_kwargs,
    )
    model = msh.models.EncodecModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        renormalize=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=DEVICE)
    safetensors.torch.load_model(
        model,
        "/home/laurent/tmp/tokenizer-de0e421d-checkpoint40.safetensors",
    )
    model.eval()
    model.set_num_codebooks(8)
    model = msh.models.MultistreamCompressionModel(model, num_sources=2)
    return model


def get_lm():
    model = msh.models.LMModel(
        **lm_kwargs,
        condition_provider=msh.conditioners.ConditionProvider([]),
        fuser=msh.conditioners.ConditionFuser(
            {
                "sum": [],
                "prepend": [],
                "cross": [],
            }
        ),
    ).to(device=DEVICE)
    safetensors.torch.load_model(
        model,
        "/home/laurent/tmp/mimi_0abbed5f@100.safetensors",
    )
    model.eval()
    return model


text_tokenizer = sentencepiece.SentencePieceProcessor(
    "/home/laurent/tmp/tokenizer_spm_32k_3.model"
)

ec = get_encodec()
print("encodec loaded")
lm = get_lm()
print("lm loaded")


def cb(step, total):
    print(f"{step:06d} / {total:06d}", end="\r")


batch_size = 8
max_gen_len_s = 10
with torch.no_grad():
    res = lm.generate(
        prompt=None,
        num_samples=batch_size,
        callback=cb,
        text_or_audio="both",
        max_gen_len=int(12.5 * max_gen_len_s),
        top_k=250,
        temp=0.8,
        strip=0,
    )
outputs = []
for single_res in res:
    print(single_res.shape)
    outputs.append(ec.decode_sources(single_res[None, 1:]))
for idx, output in enumerate(outputs):
    output = output[0, :, 0].cpu()
    print(idx, output.shape)
    torchaudio.save(f"output_{idx}.wav", output, 24000)
