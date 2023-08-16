# Text + Sketch Compression via Text-to-Image Models

Implementation for [Text + Sketch: Image Compression at Ultra Low Rates](https://arxiv.org/abs/2307.01944). 

The following scripts will loop through a dataset, and then output the results (reconstructed images, sketches, and captions) into the `recon_examples/` folder. 

* eval_PIC.py: uses prompt inversion to transmit a prompt and generate reconstructions.
* eval_PICS.py: uses prompt inversion + sketch to transmit a compressed sketch and prompt.

For example, `eval_PICS.py --data_root DATA_ROOT` will run PICS, where the images are contained in the DATA_ROOT folder. 

## Dataloaders
The dataloading assumes pytorch ImageFolder layouts inside DATA_ROOT. See `dataloaders.py` for more details. For CLIC2020, we use the train/valid/test splits from the [tensorflow builder](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/clic/clic_dataset_builder.py).

## Dependencies
* pytorch
* compressai
* diffusers

## Notes
* Since ControlNet was trained on uncompressed HED maps (the sketch), and not the decompressed ones, if the rate is set too low for the sketch, this can cause poor reconstructions for many image types.
* The performance of PICS is dependent on the ControlNet model used (here we use SD), but any improved ControlNet model released in the future can be easily integrated into the text + sketch setup
* Fine-tuning the models are currently in-progress

## Citation

    @inproceedings{lei2023text+sketch,
      title={Text+ Sketch: Image Compression at Ultra Low Rates},
      author={Lei, Eric and Uslu, Yi\u{g}it Berkay and Hassani, Hamed and Bidokhti, Shirin Saeedi},
      booktitle={ICML 2023 Workshop on Neural Compression: From Information Theory to Applications},
      year={2023}
    }

