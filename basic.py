import msh
import sentencepiece
import time
import torch
import torchaudio
import torchaudio.functional as F
import safetensors
from torch.profiler import profile, ProfilerActivity

SAMPLE_RATE = 24000
FRAME_RATE = 12.5
DEVICE = "cuda:0"
ENABLE_PROFILING = False
STREAMING_LM_GEN = True

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
    "gating": "silu",
    "norm": "real_rms_norm_f32",
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
        renormalize=False,
        causal=True,
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
        delays=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        **lm_kwargs,
    ).to(device=DEVICE)
    model.eval()
    model.to(torch.bfloat16)
    safetensors.torch.load_model(
        model,
        "/home/laurent/tmp/mimi_0abbed5f@100.safetensors",
    )
    # pkg = torch.load(
    #     "/lustre/scwpod02/client/kyutai/neilz/mimi_exp/xps/1049a9ac/checkpoint_120.th",
    #     "cpu",
    # )
    # model.load_state_dict(pkg["fsdp_best_state"]["model"])
    model.autocast = msh.utils.autocast.TorchAutocast(
        enabled=True, dtype=torch.bfloat16, device_type="cuda"
    )
    return model


text_tokenizer = sentencepiece.SentencePieceProcessor(
    "/home/laurent/tmp/tokenizer_spm_32k_3.model"
)

print("loading encodec")
ec = get_encodec()
print("encodec loaded")


def encodec_streaming_test(ec, pcm_chunk_size=1920, max_duration_sec=10.0):
    # wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
    sample_pcm, sample_sr = torchaudio.load("bria.mp3")
    print("loaded pcm", sample_pcm.shape, sample_sr)
    sample_pcm = F.resample(sample_pcm, orig_freq=sample_sr, new_freq=SAMPLE_RATE)
    max_duration_len = int(SAMPLE_RATE * max_duration_sec)
    if sample_pcm.shape[-1] > max_duration_len:
        sample_pcm = sample_pcm[..., :max_duration_len]
    print("resampled pcm", sample_pcm.shape, sample_sr)
    sample_pcm = sample_pcm[None].to(device=DEVICE)

    print("streaming encoding...")
    start_time = time.time()
    all_codes = []

    def run_loop():
        for start_idx in range(0, sample_pcm.shape[-1], pcm_chunk_size):
            end_idx = min(sample_pcm.shape[-1], start_idx + pcm_chunk_size)
            chunk = sample_pcm[..., start_idx:end_idx]
            codes, _scale = ec.model.encode(chunk)
            if codes.shape[-1]:
                print(start_idx, codes.shape, end="\r")
                all_codes.append(codes)

    if ENABLE_PROFILING:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run_loop()
        prof.export_chrome_trace("trace.json")
    else:
        run_loop()
    all_codes = torch.cat(all_codes, dim=-1)
    # print(all_codes)
    # all_codes, _scale = ec.model.encode(sample_pcm)
    # print(all_codes)
    print(f"codes {all_codes.shape} generated in {time.time() - start_time:.2f}s")
    print("streaming decoding...")
    all_pcms = []
    with ec.model.streaming():
        for i in range(all_codes.shape[-1]):
            codes = all_codes[..., i : i + 1]
            pcm = ec.model.decode(codes, scale=None)
            print(i, pcm.shape, end="\r")
            all_pcms.append(pcm)
    all_pcms = torch.cat(all_pcms, dim=-1)
    print("pcm", all_pcms.shape)
    torchaudio.save("streaming_out.wav", all_pcms[0].cpu(), SAMPLE_RATE)
    pcm = ec.model.decode(all_codes, scale=None)
    print("pcm", pcm.shape)
    torchaudio.save("roundtrip_out.wav", pcm[0].cpu(), SAMPLE_RATE)


encodec_streaming_test(ec)


print("lm loading")
lm = get_lm()
print("lm loaded")


def cb(step, total):
    print(f"{step:06d} / {total:06d}", end="\r")


if STREAMING_LM_GEN:
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
        pcm = ec.model.decode(all_codes[:, :, 2:], scale=None)
        print("pcm", pcm.shape)
        torchaudio.save("gen_main.wav", pcm[0].cpu(), SAMPLE_RATE)
        torchaudio.save("gen_other.wav", pcm[1].cpu(), SAMPLE_RATE)
else:
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
        )
    outputs = []
    for single_res in res:
        print(single_res.shape)
        outputs.append(ec.decode_sources(single_res[None, 1:]))
    for idx, output in enumerate(outputs):
        output = output[0, :, 0].cpu()
        print(idx, output.shape)
        torchaudio.save(f"output_{idx}.wav", output, SAMPLE_RATE)
