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

for key, value in hf_dict.items():

    new_key = key

    # Mapping rules
    if key.startswith('decoder.layers'):
        new_key = key.replace('decoder.layers', 'decoder.model')
    elif key.startswith('encoder.layers'):
        new_key = key.replace('encoder.layers', 'encoder.model')
    elif key.startswith('decoder_transformer'):
        new_key = key.replace('decoder_transformer', 'decoder_transformer.transformer')
    elif key.startswith('encoder_transformer'):
        new_key = key.replace('encoder_transformer', 'encoder_transformer.transformer')
    elif key.startswith('quantizer.semantic_residual_vector_quantizer'):
        new_key = key.replace('quantizer.semantic_residual_vector_quantizer', 'quantizer.rvq_first')
    elif key.startswith('quantizer.acoustic_residual_vector_quantizer'):
        new_key = key.replace('quantizer.acoustic_residual_vector_quantizer', 'quantizer.rvq_rest')
    elif key.startswith('downsample.conv'):
        new_key = key.replace('downsample.conv', 'downsample.conv.conv.conv')
        mapped_weights[new_key] = value
        continue  # Skip to next iteration since we've handled this key
    elif key.startswith('upsample.conv'):
        new_key = key.replace('upsample.conv', 'upsample.convtr.convtr.convtr')
        mapped_weights[new_key] = value
        continue  # Skip to next iteration since we've handled this key

    # Handle extra 'conv' in convolutional layers
    if 'conv.bias' in new_key and 'decoder' in new_key:
        flag = False
        for str_ in ['decoder.model.2', 'decoder.model.5', 'decoder.model.8', 'decoder.model.11']:
            if str_ in new_key:
                new_key = new_key.replace('conv.bias', 'convtr.convtr.bias')
                flag = True
        if not flag:
            new_key = new_key.replace('conv.bias', 'conv.conv.bias')
    if 'conv.weight' in new_key and 'decoder' in new_key:
        flag = False
        for str_ in ['decoder.model.2', 'decoder.model.5', 'decoder.model.8', 'decoder.model.11']:
            if str_ in new_key:
                new_key = new_key.replace('conv.weight', 'convtr.convtr.weight')
                flag = True
        if not flag:
            new_key = new_key.replace('conv.weight', 'conv.conv.weight')
            
    if 'conv.bias' in new_key and 'encoder' in new_key:
        new_key = new_key.replace('conv.bias', 'conv.conv.bias')
    if 'conv.weight' in new_key and 'encoder' in new_key:
        new_key = new_key.replace('conv.weight', 'conv.conv.weight')
    # if 'convtr.bias' in new_key:
    #     new_key = new_key.replace('convtr.bias', 'convtr.convtr.bias')
    # if 'convtr.weight' in new_key:
    #     new_key = new_key.replace('convtr.weight', 'convtr.convtr.weight')

    # Map transformer layer norms
    if 'input_layernorm.bias' in new_key:
        new_key = new_key.replace('input_layernorm.bias', 'norm1.bias')
    if 'input_layernorm.weight' in new_key:
        new_key = new_key.replace('input_layernorm.weight', 'norm1.weight')
    if 'post_attention_layernorm.bias' in new_key:
        new_key = new_key.replace('post_attention_layernorm.bias', 'norm2.bias')
    if 'post_attention_layernorm.weight' in new_key:
        new_key = new_key.replace('post_attention_layernorm.weight', 'norm2.weight')

    # Map MLP layers
    if 'mlp.fc1.weight' in new_key:
        new_key = new_key.replace('mlp.fc1.weight', 'linear1.weight')
    if 'mlp.fc2.weight' in new_key:
        new_key = new_key.replace('mlp.fc2.weight', 'linear2.weight')

    # Map layer scales
    if 'self_attn_layer_scale.scale' in new_key:
        new_key = new_key.replace('self_attn_layer_scale.scale', 'layer_scale_1.scale')
    if 'mlp_layer_scale.scale' in new_key:
        new_key = new_key.replace('mlp_layer_scale.scale', 'layer_scale_2.scale')

    # Map codebook parameters in quantizers
    if 'layers' in new_key and "transformer" not in new_key:
        new_key = new_key.replace('layers', 'vq.layers')
    if '.codebook.initialized' in new_key:
        new_key = new_key.replace('.codebook.initialized', '._codebook._initialized')
    if '.codebook.cluster_usage' in new_key:
        new_key = new_key.replace('.codebook.cluster_usage', '._codebook.cluster_usage')
    if '.codebook.embed_sum' in new_key:
        new_key = new_key.replace('.codebook.embed_sum', '._codebook.embedding_sum')

    # Map self-attention output projection
    if 'self_attn.o_proj.weight' in new_key:
        new_key = new_key.replace('self_attn.o_proj.weight', 'self_attn.out_proj.weight')

    # Handle self-attention input projections (split into q, k, v)
    elif 'self_attn' in new_key and 'self_attn.o_proj.weight' not in new_key:
        # Skip adding this key now; we'll handle it separately
        continue

    mapped_weights[new_key] = value

# Handle 'self_attn.in_proj_weight' by splitting into q_proj, k_proj, v_proj
for key, value in hf_dict.items():
    if ('self_attn' in key) and ('self_attn.o_proj.weight' not in new_key) and ('q_proj.weight' in key):
        embed_dim = value.shape[1]

        k_proj_weight = hf_dict[key.replace('q_proj.weight', 'k_proj.weight')]
        v_proj_weight = hf_dict[key.replace('q_proj.weight', 'v_proj.weight')]
        moshi_value = torch.cat([value, k_proj_weight, v_proj_weight], dim=0)
        
        if key.startswith('encoder_transformer'):
            base_key = key.replace('encoder_transformer.layers', 'encoder_transformer.transformer.layers')
        elif key.startswith('decoder_transformer'):
            base_key = key.replace('decoder_transformer.layers', 'decoder_transformer.transformer.layers')
            
        
        mapped_weights[base_key.replace('q_proj.weight', 'in_proj_weight')] = moshi_value


# Save the new state dict
save_file(mapped_weights, '/data0/questar/users/wuzhiyue/tmp/hf_mimi_to_moshi/model.safetensors', metadata = {'format': 'pt'})

ckpt = {'state_dict': mapped_weights}
torch.save(ckpt, '/home/wuzhiyue/ckpt/Mimi/pretrain_hf_mimi.ckpt')