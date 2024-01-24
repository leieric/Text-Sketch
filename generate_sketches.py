'''This script takes in a data set of images
and generates corresponding sketches'''

# import libraries
from annotator.hed import HEDdetector
from annotator.uniformer import UniformerDetector
from annotator.util import HWC3, resize_image
from PIL import Image
import dataloaders
from argparse import ArgumentParser
import tqdm
import numpy as np


def main():
    # parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--sketch_type', default='segmentation', type=str)
    parser.add_argument('--dataset', default='CLIC2021', type=str)
    parser.add_argument('--data_root', default='/home/eric/data/', type=str)
    parser.add_argument('--save_dir', default='/home/eric/data/CLIC/2021/', type=str)

    args = parser.parse_args()

    # load data
    dm = dataloaders.get_dataloader(args)

    # iterate through each data split and generate/save sketches
    for i, x in tqdm.tqdm(enumerate(dm.train_dset)):
        x = x[0]
        x_im = (255*x.permute(1,2,0)).numpy().astype(np.uint8)
        im = resize_image(HWC3(x_im), 512)
        apply_seg = UniformerDetector()
        _, seg_map = apply_seg(im)
        sketch = np.squeeze(seg_map)
        sketch.save(f'{args.data_root}/CLIC/2021/segmentation/test/segmentation_{i}.png')
