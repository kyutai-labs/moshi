# Testing
In order to test the audio tokenizer, you can run the following command.

```bash
wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
PYTHONPATH=. python scripts/mimi_test.py --weights tokenizer-e351c8d8-checkpoint125.safetensors
```

In order to test moshi, run the following.
```bash
PYTHONPATH=. python scripts/moshi_test.py \
    --mimi-weights tokenizer-e351c8d8-checkpoint125.safetensors \
    --tokenizer tokenizer_spm_32k_3.model \
    --moshi-weights moshiko_pt_301e30bf@120.safetensors
```
