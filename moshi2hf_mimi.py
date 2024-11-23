from safetensors.torch import load_file, save_file
import torch
import re
from einops import rearrange

# Load the weight files
root_moshi_mimi_path = '/home/wuzhiyue/huggingface_ckpt/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors'
root_hf_mimi_path = '/home/wuzhiyue/huggingface_ckpt/mimi/model.safetensors'

hf_dict = load_file(root_hf_mimi_path)
# state_dict2 = load_file(root_moshi_mimi_path)

# Load the second weight file (File B)
moshi_dict = load_file(root_moshi_mimi_path)

mapped_weights = {}

for key in moshi_dict.keys():
    value = moshi_dict[key]
    # if key.startswith('quantizer.rvq_rest.vq'):
        # if eval(key.split('.')[4]) >= 7:
            # continue
    new_key = key

    # Mapping rules
    if key.startswith('decoder.model'):
        new_key = key.replace('decoder.model', 'decoder.layers')
    elif key.startswith('encoder.model'):
        new_key = key.replace('encoder.model', 'encoder.layers')
    elif key.startswith('decoder_transformer.transformer'):
        new_key = key.replace('decoder_transformer.transformer', 'decoder_transformer')
    elif key.startswith('encoder_transformer.transformer'):
        new_key = key.replace('encoder_transformer.transformer', 'encoder_transformer')
    elif key.startswith('quantizer.rvq_first'):
        new_key = key.replace('quantizer.rvq_first', 'quantizer.semantic_residual_vector_quantizer')
    elif key.startswith('quantizer.rvq_rest'):
        new_key = key.replace('quantizer.rvq_rest', 'quantizer.acoustic_residual_vector_quantizer')
    elif key.startswith('downsample.conv.conv.conv'):
        new_key = key.replace('downsample.conv.conv.conv', 'downsample.conv')
        mapped_weights[new_key] = value
        continue  # Skip to next iteration since we've handled this key
    elif key.startswith('upsample.convtr.convtr.convtr'):
        new_key = key.replace('upsample.convtr.convtr.convtr', 'upsample.conv')
        mapped_weights[new_key] = value
        continue  # Skip to next iteration since we've handled this key

    # Handle extra 'conv' in convolutional layers
    if 'conv.conv.bias' in new_key:
        new_key = new_key.replace('conv.conv.bias', 'conv.bias')
    if 'conv.conv.weight' in new_key:
        new_key = new_key.replace('conv.conv.weight', 'conv.weight')
    if 'convtr.convtr.bias' in new_key:
        new_key = new_key.replace('convtr.convtr.bias', 'conv.bias')
    if 'convtr.convtr.weight' in new_key:
        new_key = new_key.replace('convtr.convtr.weight', 'conv.weight')

    # Map transformer layer norms
    if 'norm1.bias' in new_key:
        new_key = new_key.replace('norm1.bias', 'input_layernorm.bias')
    if 'norm1.weight' in new_key:
        new_key = new_key.replace('norm1.weight', 'input_layernorm.weight')
    if 'norm2.bias' in new_key:
        new_key = new_key.replace('norm2.bias', 'post_attention_layernorm.bias')
    if 'norm2.weight' in new_key:
        new_key = new_key.replace('norm2.weight', 'post_attention_layernorm.weight')

    # Map MLP layers
    if 'linear1.weight' in new_key:
        new_key = new_key.replace('linear1.weight', 'mlp.fc1.weight')
    if 'linear2.weight' in new_key:
        new_key = new_key.replace('linear2.weight', 'mlp.fc2.weight')

    # Map layer scales
    if 'layer_scale_1.scale' in new_key:
        new_key = new_key.replace('layer_scale_1.scale', 'self_attn_layer_scale.scale')
    if 'layer_scale_2.scale' in new_key:
        new_key = new_key.replace('layer_scale_2.scale', 'mlp_layer_scale.scale')

    # Map codebook parameters in quantizers
    if 'vq.layers' in new_key:
        new_key = new_key.replace('vq.layers', 'layers')
    if '._codebook._initialized' in new_key:
        new_key = new_key.replace('._codebook._initialized', '.codebook.initialized')
    if '._codebook.cluster_usage' in new_key:
        new_key = new_key.replace('._codebook.cluster_usage', '.codebook.cluster_usage')
    if '._codebook.embedding_sum' in new_key:
        new_key = new_key.replace('._codebook.embedding_sum', '.codebook.embed_sum')

    # Map self-attention output projection
    if 'self_attn.out_proj.weight' in new_key:
        new_key = new_key.replace('self_attn.out_proj.weight', 'self_attn.o_proj.weight')

    # Handle self-attention input projections (split into q, k, v)
    if 'self_attn.in_proj_weight' in new_key:
        # Skip adding this key now; we'll handle it separately
        continue

    mapped_weights[new_key] = value

# Handle 'self_attn.in_proj_weight' by splitting into q_proj, k_proj, v_proj
for key in moshi_dict.keys():
    if 'self_attn.in_proj_weight' in key:
        value = moshi_dict[key]
        embed_dim = value.shape[1]
        (q_proj_weight, 
        k_proj_weight, 
        v_proj_weight)= rearrange(value, '(p d) e -> p d e', p = 3)
        # q_proj_weight = value[:embed_dim, :]
        # k_proj_weight = value[embed_dim:2*embed_dim, :]
        # v_proj_weight = value[2*embed_dim:, :]

        base_key = key.replace('self_attn.in_proj_weight', 'self_attn')
        if key.startswith('encoder_transformer.transformer.layers'):
            base_key = base_key.replace('encoder_transformer.transformer.layers', 'encoder_transformer.layers')
        elif key.startswith('decoder_transformer.transformer.layers'):
            base_key = base_key.replace('decoder_transformer.transformer.layers', 'decoder_transformer.layers')

        q_key = base_key + '.q_proj.weight'
        k_key = base_key + '.k_proj.weight'
        v_key = base_key + '.v_proj.weight'

        mapped_weights[q_key] = q_proj_weight
        mapped_weights[k_key] = k_proj_weight
        mapped_weights[v_key] = v_proj_weight

# Save the new state dict
save_file(mapped_weights, '/home/wuzhiyue/huggingface_ckpt/mimi/model.safetensors', metadata = {'format': 'pt'})

ckpt = {'state_dict': mapped_weights}
torch.save(ckpt, '/home/wuzhiyue/ckpt/Mimi/pretrain_hf_mimi.ckpt')

# # 比较两个字典的key中的value是否一致
# count = 0
# for key in mapped_weights.keys():
#     count += 1
#     if key in hf_dict:
#         if not torch.equal(mapped_weights[key], hf_dict[key]):
#             abs_max_diff =(mapped_weights[key] - hf_dict[key]).abs().max()
#             print(f'key {key} is not same, abs_max_diff: {abs_max_diff}')
#         # print(f'key {key} {torch.equal(mapped_weights[key], state_dict1[key])}, count: {count}')
#     else:
#         print(f'key {key} is not in state_dict1')


# import os
# from datasets import load_dataset, Audio
# from huggingface_hub import hf_hub_download
# from transformers import MimiModel, AutoFeatureExtractor
# if os.environ.get("MOSHI_DIR", None) is not None:
#     import sys

#     sys.path.insert(0, os.environ["MOSHI_DIR"])


# from moshi.moshi.models import loaders
# import torch
# import numpy as np

# import librosa
# from scipy.io import wavfile
# from lhotse import CutSet, load_manifest


# def wav_save(path, sr, wav):
#     wav = (wav * 32767) / max(0.01, np.max(np.abs(wav)))
#     wavfile.write(path, sr, wav.astype(np.int16))


# # load a demonstration datasets
# librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# # load the model + feature extractor (for pre-processing the audio)
# model = MimiModel.from_pretrained("/home/wuzhiyue/huggingface_ckpt/mimi")
# feature_extractor = AutoFeatureExtractor.from_pretrained("/home/wuzhiyue/huggingface_ckpt/mimi")

# # cast the audio data to the correct sampling rate for the model
# librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
# audio_sample0 = librispeech_dummy[15]["audio"]["array"]
# audio_sample1 = librispeech_dummy[16]["audio"]["array"]

# # audio_samples, _ = librosa.load('/home/wumenglin/workspace/devcontainer/temp/data/fe_03_00001.wav', sr=24000, duration=25, mono=False)

# # audio_sample0 = audio_samples[0]
# # audio_sample1 = audio_samples[1]

# samples = [audio_sample0, audio_sample1]

# # samples = torch.nn.utils.rnn.pad_sequence(
# #     [torch.from_numpy(sample) for sample in samples],
# #     batch_first=True,
# #     padding_value=0,
# # )
# # audio_samples = samples

# # pre-process the inputs

# inputs = feature_extractor(raw_audio=samples, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

# # explicitly encode then decode the audio inputs
# with torch.no_grad():
#     encoder_outputs_hf_mimi = model.encode(inputs["input_values"])
#     audio_values_hf_mimi = model.decode(encoder_outputs_hf_mimi.audio_codes)[0]

#     # or the equivalent with a forward pass
#     audio_values_moshi_direct = model(inputs["input_values"]).audio_values

# mimi_weight = '/home/wuzhiyue/huggingface_ckpt/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors'

# # mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)

# mimi = loaders.get_mimi(mimi_weight, 'cpu')

# #inputs["input_values"] = torch.from_numpy(audio_samples).unsqueeze(1)

# with torch.no_grad():
#     encoder_outputs_moshi_mimi = mimi.encode(inputs["input_values"])
#     audio_values_moshi_mimi = mimi.decode(encoder_outputs_moshi_mimi)

# wav_save('mimi_audio.wav', 24000, audio_values_moshi_mimi[0, 0].detach().cpu().numpy())
# wav_save('moshi_audio.wav', 24000, audio_values_hf_mimi[0, 0].detach().cpu().numpy())


# manifest_file = "/home/wumenglin/workspace/devcontainer/repo/RSTnet/data/fisher/Fisher_LDC2004S13-x.jsonl.gz"

# data = load_manifest(manifest_file)
# cut_sets = CutSet.from_cuts([data[0], data[1]])

# feature0 = cut_sets[0].load_features()
# feature0 = feature0.reshape(-1, 2, 32)
# feature0_c8 = feature0[..., :8] # T B C
# feature0_c8 = torch.from_numpy(feature0_c8).long()
# feature0_c8 = feature0_c8.permute(1, 2, 0)

# with torch.no_grad():
#     audio_values_h5py = mimi.decode(feature0_c8)

# wav_save('audio_values_h5py.wav', 24000, audio_values_h5py[0, 0].detach().cpu().numpy())