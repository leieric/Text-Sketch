# Text + Sketch Compression via Text-to-Image Models

Implementation for [Text + Sketch: Image Compression at Ultra Low Rates](https://arxiv.org/abs/2307.01944). 

The following scripts will loop through a dataset, and then output the results (reconstructed images, sketches, and captions) into the `recon_examples/` folder. 

* eval_PIC.py: uses prompt inversion to transmit a prompt and generate reconstructions.
* eval_PICS.py: uses prompt inversion + sketch to transmit a compressed sketch and prompt.

For example, `python eval_PICS.py --data_root DATA_ROOT` will run PICS, where the images are contained in the DATA_ROOT folder. See `scripts/PICS.sh` for example usage. Prior to running this script, you will need to either (a) train the NTC sketch model or (b) download the pre-trained ones into the `models_ntc/` folder. Instructions for both can be found below.

The `annotator` directory is taken from the [ControlNet repo](https://github.com/lllyasviel/ControlNet.git), and the `prompt_inversion` directory is based off of the [Hard Prompts Made Easy repo](https://github.com/YuxinWenRick/hard-prompts-made-easy/tree/main).

## Dataloaders
The dataloading assumes pytorch ImageFolder layouts inside DATA_ROOT. See `dataloaders.py` for more details. 

## Sketch NTC Models
A training script is provided in `train_compressai.py`, which is slightly modified from CompressAI's example training script. See `scripts/train_sketch.sh` example usage. To generate sketch training data, apply one of the filters in `annotator/` to training images, and structure folder to fit the [CompressAI ImageFolder](https://interdigitalinc.github.io/CompressAI/datasets.html#imagefolder).

Pre-trained NTC models for HED sketches, as well as HED sketches generated from CLIC2021 used to train it, can be found [here](https://upenn.box.com/s/m3hjxjouw9moe7xd3xt2jsx7ifywbiha). To download them onto a remote server, run 
- `wget https://upenn.box.com/shared/static/g1fzf9ctn0qvdn9exjpp8mkqh7aja4gm -O trained_ntc_models.zip`
- `wget https://upenn.box.com/shared/static/b90504o4k4onkicm8aal8fxkhltp2rnb -O HED_training_data.zip`
  
## Dependencies
* pytorch
* compressai
* diffusers
* pytorch-lightning
* opencv-python
* einops
* ftfy
* sentence-transformers
* accelerate
* xformers
* basicsr

## Notes
* Since ControlNet was trained on uncompressed HED maps (the sketch), and not the decompressed ones, if the rate is set too low for the sketch, this can cause poor reconstructions for many image types.
* In general, the Text + Sketch is better at reconstructing landscape photos compared to photos of objects. The performance is highly dependent on the pre-trained ControlNet model used (here we use SD), but any improved ControlNet model released in the future can be easily integrated into the Text + Sketch setup
* Fine-tuning the models are currently in-progress

## Citation

    @inproceedings{lei2023text+sketch,
      title={Text+ Sketch: Image Compression at Ultra Low Rates},
      author={Lei, Eric and Uslu, Yi\u{g}it Berkay and Hassani, Hamed and Bidokhti, Shirin Saeedi},
      booktitle={ICML 2023 Workshop on Neural Compression: From Information Theory to Applications},
      year={2023}
    }

