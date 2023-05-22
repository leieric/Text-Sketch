# Textual Transform Coding
Based on the [ControlNet](https://github.com/lllyasviel/ControlNet) repo. Also uses a version of [BLIP](https://github.com/salesforce/BLIP) captioning, and the code from [Hard Prompts Made Easy](https://github.com/YuxinWenRick/hard-prompts-made-easy/tree/main), which is the PGD-based prompt inversion method.

I have several scripts so far. Each one will loop through a dataset (either [CLIC](http://compression.cc/tasks/) or [Kodak](https://r0k.us/graphics/kodak/), which are standard benchmarks) and then output the results in the recon_examples folder:

* eval_llmc_pi_compress_kodak.py: uses prompt inversion to transmit a prompt and generate reconstructions
* eval_llmc_pi+hed_compress_kodak.py: uses prompt inversion + sketch to transmit a compressed sketch and prompt (currently finishing up).

The other eval_llmc files are intermediate test versions I used.

