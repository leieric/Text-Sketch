'''This script takes in a data set of images
and generates corresponding sketches. 

Assume data_root and save_dir both match structure of CompressAI ImageFolder:
https://interdigitalinc.github.io/CompressAI/datasets.html#imagefolder.

For the save directory:  
    - /save_dir/sketch_type/ 
        - train/
            - sketch_type_0.png
            - sketch_type_1.png
        - test/
            - sketch_type_0.png
            - sketch_type_1.png
        - valid/
            - sketch_type_0.png
            - sketch_type_1.png

For more info on input data_root structure, see dataloaders.py.

Note: For segmentation maps, pixels assume an integer value in the range
[0, 149] based on semantic class membership. We store the maps
as greyscale images [0, 255] to simply the loading and training process,
so many of the png files will look like fully black images, but they
actually store the correct pixel values.'''

# import libraries
from annotator.hed import HEDdetector
from annotator.uniformer import UniformerDetector
from annotator.util import HWC3, resize_image
from PIL import Image
import dataloaders
from argparse import ArgumentParser
import tqdm
import numpy as np
import sys
import torch
import os
from torchvision.transforms.functional import to_pil_image, pil_to_tensor


def main():
    # parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--sketch_type', default='segmentation', type=str, 
                        help='Supported sketch types: segmentation, hed.')
    parser.add_argument('--dataset', default='CLIC2021', type=str)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--data_root', default='/home/eric/data/', type=str)
    parser.add_argument('--save_dir', default='/home/eric/data/CLIC/2021/', type=str)
    args = parser.parse_args()

    # set save path
    # example: '/home/eric/data/CLIC/2021/segmentation/train/'

    # load data
    # dm = dataloaders.get_dataloader(args)

    # # data split
    # if args.split == 'train':
    #     data = dm.train_dset
    # elif args.split == 'test':
    #     data = dm.test_dset
    # elif args.split == 'val' or args.split == 'valid':
    #     data = dm.val_dset
    #     args.split = 'valid'
    # else:
    #     sys.exit('Not a valid split.')

    # sketch generator function
    if args.sketch_type == 'segmentation':
        apply = UniformerDetector()
        mode = 'L'
    elif args.sketch_type == 'hed':
        apply = HEDdetector()
        mode = 'L'
    else:
        sys.exit("Not a valid sketch type. Choose 'segmentation' or 'hed'.")

    # iterate through data and generate/save sketches
    data_dir = os.fsencode(args.data_root)
    for i, file in enumerate(os.listdir(data_dir)):
        filename = os.fsdecode(file)
        if not filename.endswith(".png"):
            continue
        print(f"Image: {filename}")
       
        x = Image.open(os.path.join(args.data_root, filename))
        x = pil_to_tensor(x)
        x_img = (x.permute(1,2,0)).numpy().astype(np.uint8)
        img = resize_image(HWC3(x_img), 512)
        sketch = apply(img)
        print(f"\tClasses: {np.unique(sketch)}\n")
        print(type(sketch))
        torch.save(torch.from_numpy(sketch), os.path.join(args.save_dir, f'{filename[:-4]}_{args.sketch_type}.pt'))
        # sketch_img = Image.fromarray(sketch, mode=mode)
        # sketch_img.save(os.path.join(args.save_dir, f'{filename[:-4]}_{args.sketch_type}.png'))
    return


if __name__ == "__main__":
    main()