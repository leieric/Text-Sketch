'''Script to generate training data for PICS
'''

# import libraries
from PIL import Image
import dataloaders
import argparse
from argparse import ArgumentParser, Namespace
import tqdm
import numpy as np
import sys
import os
import csv

from annotator.hed import HEDdetector
from annotator.uniformer import UniformerDetector

import prompt_inversion.optim_utils as prompt_inv
import prompt_inversion.open_clip as open_clip 

from torchvision import transforms


def generate_sketch(x: Image.Image, sketch_type: str) -> Image.Image:
    '''
    Generates sketch from input image

    Arguments:
        x: input image (PIL.Image.Image)
        sketch_type: type of sketch (str)
    
    Returns:
        sketch: sketch of input image (PIL.Image.Image)
    '''
    if sketch_type == 'hed':
        apply = HEDdetector()
        mode = 'L'
    elif sketch_type =='seg':
        apply = UniformerDetector()
        mode = 'L'
    else:
        raise ValueError("Not a valid sketch type. Choose 'hed' or 'seg'.")
    
    # convert image to numpy array
    x = np.array(x).astype(np.uint8)
    # generate sketch
    sketch = apply(x)
    # convert sketch back to PIL Image
    sketch = Image.fromarray(sketch, mode=mode)
    return sketch

def main():
    # parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--dataset', 
                        type=str,
                        required=True,
                        help='Name of dataset'
    )
    parser.add_argument('--sketch_type',
                        type=str, 
                        required=True, 
                        help="Supported sketch types: 'seg' or 'hed'."
    )
    parser.add_argument('--split',
                        type=str,
                        required=True,
                        help='train/val/test'
    )
    parser.add_argument('--data_root',
                        type=str,
                        required=True,
                        help='Path to directory containing input images'
    )
    parser.add_argument('--save_dir',
                        type=str,
                        required=True,
                        help='Path to directory where PICS training data is saved'
    )
    parser.add_argument('--overwrite',
                        action=argparse.BooleanOptionalAction,
                        help='Overwrite existing metadata file for chosen dataset.')
    args = parser.parse_args()

    args.resolution = 512

    # create subdirs if they do not exist
    save_path = os.path.join(args.save_dir, args.dataset, args.sketch_type, args.split)
    image_path = os.path.join(save_path, 'image')
    condition_path = os.path.join(save_path, 'conditioning_image')
    meta_path = os.path.join(save_path, 'metadata.csv')
    
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(condition_path, exist_ok=True)

    if args.overwrite:
        while True:
            ow_flag = input("Overwrite flag is set. If you continue, the metadata file currently saved at"
                            f"'{meta_path}' will be overwritten. Do you wish to continue? (y/n) ")
            if ow_flag == 'y':
                with open(meta_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["image", "conditioning_image", "text"])
                    file.close()
                break
            elif ow_flag == 'n':
                print(f"Appending new metadata to {meta_path}...\n")
                break
            else:
                print("Valid options are 'y' and 'n'.")

    # define input image transforms
    input_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        ]
    )
    args.transforms = input_transforms
    args.batch_size = 1

    # load CLIP model for prompt inverison
    args_clip = Namespace()
    args_clip.__dict__.update(prompt_inv.read_json("./prompt_inversion/sample_config.json"))
    clip, _, clip_preprocess = open_clip.create_model_and_transforms(args_clip.clip_model, 
                                                                     pretrained=args_clip.clip_pretrain, 
                                                                     device='cuda:0')
    
    # iterate through data and generate/save sketches
    for i, x in enumerate(os.listdir(args.data_root)):
        print(f"\nProcessing image {i+1}/{len(os.listdir(args.data_root))}\n")
        print(x)
        x_path = os.path.join(args.data_root, x)
        # load input image
        image = Image.open(x_path)
        # save image to PICS train dataset
        save_image_path = os.path.join(image_path, x)
        image.save(save_image_path)
        # generate sketch from input image and save to PICS train dataset
        sketch = generate_sketch(image, 'hed')
        save_condition_path = os.path.join(condition_path, x)
        sketch.save(save_condition_path)
        # generate textual caption of input image
        caption = prompt_inv.optimize_prompt(model=clip, 
                                                preprocess=clip_preprocess, 
                                                args=args_clip, 
                                                device='cuda:0', 
                                                target_images=[image])
        image.close()

        # save metadata
        with open(meta_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([x, x, caption])
            file.close()

    return


if __name__ == "__main__":
    main()