# FAQ

Here is the answer to a number of frequently asked questions.

### Will you release training code?

Some finetuning code can be found in the [kyutai-labs/moshi-finetune repo](https://github.com/kyutai-labs/moshi-finetune).

### Will you release the dataset?

We will not release the pre-training dataset.

### Is Moshi multilingual?

At the moment no. Moshi only speaks English. It has some basic support for translating some sentences
or words to other languages, but you shouldn't expect to use it fully in any other language than English.

### Can I change Moshi's voice / personality?

This would require fine tuning, which is not currently supported.

### Can Moshi run on a M1, or smaller GPUs?

Sadly we do not think this is currently possible. Quantizing beyond 4 bits lead to dramatic
decrease in quality, see [PR #58](https://github.com/kyutai-labs/moshi/pull/58).
While we keep those limitations in mind for future versions, there is no immediate solution.

### Can we run quantized Moshi with PyTorch?

At the moment no, we might look into adding this feature when we get the time. At the moment
it is however possible to use the Rust backend, which should run in int8 with CUDA.

### Moshi stopped talking after 5 min.

This is expected on the MLX and Rust implementation.
We only use a fixed buffer, and we do not discard past entries.
The PyTorch version should work for unlimited times, although this is mostly untested and we
expect the quality to degrade after a bit (we have no attention sink or other mechanism to improve the streaming
beyond the finite context used at training).

### The server seems to be running but nothing happens on connect.

For diagnosis, look at your browser console if there is any error being
reported.

If you see issues that look like the following:
```
Uncaught (in promise) TypeError: Cannot read properties of undefined (reading 'addModule')
```
this is likely caused by the http server being remote and audio being disabled
for http in such a case.

To get around this, tunnel the 8998 port from the remote server to the localhost
via ssh and access [localhost:8998](http://localhost:8998) via http normally
after that.

### How to get the key.pem and cert.pem files required for serving over https?
```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```

### Can I run on a 12GB / 8 GB GPU ?
For a 12GB GPU, this is possible following instructions in [issue #54](https://github.com/kyutai-labs/moshi/issues/54).
For 8GB GPU, this is not possible at the moment.
