# FAQ

Here is the answer to a number of frequently asked questions.

### Will you release training code?

We will release some training / fine-tuning code, but we do not have any timeline yet. Please be patient.

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
