# Text + Sketch

The scripts will loop through a dataset, and then output the results (reconstructed images, sketches, and captions) into the `recon_examples/` folder. 

* eval_llmc_pi_compress_kodak.py: uses prompt inversion to transmit a prompt and generate reconstructions (i.e., PIC)
* eval_llmc_pi+hed_compress_kodak.py: uses prompt inversion + sketch to transmit a compressed sketch and prompt (i.e., PICS)

## Dataloaders
The dataloading assumes pytorch ImageFolder layouts. For CLIC2020, we use the train/valid/test splits from the [tensorflow builder](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/clic/clic_dataset_builder.py).

Make sure you use CUDA_VISIBLE_DEVICES=i before calling python in the command line to run on gpu i.
A working environment can be found by making a conda environment, and then entering `pip install -r requirements.txt`.

