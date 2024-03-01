'''This script runs the full pipeline
    for Prompt-Inversion Compression w/ Segmentation Map Sketches'''

# import libraries
import torch
import os
import numpy as np
import math
import pathlib
from PIL import Image
import yaml
import sys, zlib
import tqdm

from argparse import ArgumentParser, Namespace
import dataloaders

from annotator.uniformer import UniformerDetector
from annotator.util import HWC3, resize_image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

import prompt_inversion.optim_utils as prompt_inv
import prompt_inversion.open_clip as open_clip 

from ntc_segmentation_model import Cheng2020AttentionSeg
from ntc_segmentation_recon import segmap_gray2rgb


# global variables to enhance input text prompts in stable diffusion model
prompt_pos = 'high quality'
prompt_neg = 'disfigured, deformed, low quality, lowres, b&w, blurry, Photoshop, video game, bad art'


def get_loss(loss_type: str):
    '''
    Helper function to define loss in Reverse Channel Coding scheme

    Arguments:
        loss_type: type of loss to use (str)

    Returns:
        loss_function: loss function used to determine best index in RCC (function)
    '''
    if loss_type == 'clip':
        # create Namespace() for clip model args
        args_clip = Namespace()
        # populate args_clip with entries from config file
        args_clip.__dict__.update(prompt_inv.read_json("prompt_inversion/sample_config.json"))
        # instantiate pretrained clip model 
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(args_clip.clip_model, 
                                                                               pretrained=args_clip.clip_pretrain, 
                                                                               device='cuda:0')
        # return loss function that computes cosine similarity between input images
        return lambda x, xhat: 1 - prompt_inv.clip_cosine(x, xhat, clip_model, clip_preprocess, 'cuda:0')
    else:
        sys.exit('Not a valid loss')

def encode_rcc(model, clip, preprocess, ntc_sketch, im, caption=None, N=5):
    '''
    Function to encode image using prompt inversion and generate HED sketch on the side

    Arguments: 
        model: ControlNet model
        clip: CLIP model
        preprocess: CLIP model preprocess
        ntc_sketch: NTC model
        im: input image to compress
        caption: caption of image if previously generated
        N: number of candidate images to generate

    Returns:
        caption: text string containing caption (str)
        sketch: Segmentation sketch of original image
        sketch_recon: reconstructed sketch from NTC model
        sketch_dict: dict containing compressed sketch
        idx: index selected (int)
    '''
    # generate segmentation map and preprocess to match required input to NTC encoder
    # input to NTC encoder should be torch tensor with shape (batch_size=1 x 1 x H x W) 
    # and dtype torch.float32, with values in the range [0,149]
    apply_seg = UniformerDetector()
    seg_map = apply_seg(im)
    sketch = torch.from_numpy(seg_map)
    sketch = sketch[None, None, ...]
    sketch = sketch.type(dtype=torch.float32)
   
    # compress sketch using NTC encoder
    # reconstruct sketch using NTC encoder to generate candidate images in RCC
    with torch.no_grad():
        # TODO implement decompress for segmentation maps
        # out_net = ntc_sketch.decompress(sketch_dict['strings'], sketch_dict['shape'])
        sketch_dict = ntc_sketch.compress(sketch)
        out_net = ntc_sketch.forward(sketch)['x_hat']
        _, sketch_recon = torch.max(out_net, dim=1, keepdim=False)
    
    # render segmentation maps as Images
    sketch = segmap_gray2rgb(sketch.squeeze())
    sketch_recon = segmap_gray2rgb(sketch_recon.squeeze())

    # Generate image caption using Prompt Inversion if not previously generated
    if caption is None:
        caption = prompt_inv.optimize_prompt(clip, preprocess, args_clip, 'cuda:0', target_images=[Image.fromarray(im)])
    
    # run ControlNet model to generate N candidate images
    guidance_scale = 9
    num_inference_steps = 25
    images = model(f'{caption}, {prompt_pos}',
                    sketch_recon,
                    generator=[torch.Generator(device="cuda").manual_seed(i) for i in range(N)],
                    num_images_per_prompt=N ,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=im.shape[0],
                    width=im.shape[1],
                    negative_prompt=prompt_neg).images
    # compute CLIP cosine similarity between original image and generated image candidates
    loss = loss_func([Image.fromarray(im)]*N, images).squeeze()
    # compute index of candidate image that minimizes loss
    idx = torch.argmin(loss)
    
    return caption, sketch, sketch_recon, sketch_dict, idx


def recon_rcc(model, caption, sketch_recon, idx, N=5):
    '''
    Function to decode image using ControlNet to generate new image using encoded prompt and sketch
    
    Arguments:
        model: ControlNet model
        caption: text string caption
        sketch_recon: reconstructed sketch from NTC model
        idx: index of best candidate image
        N: number of candidate images to generate
    
    Returns:
        im_recon: reconstructed image generated from ControlNet 
    '''
    # decode sketch using NTC model
    # with torch.no_grad():
        # TODO implement decompress for segmentation maps 
    #     out_net = ntc_sketch.decompress(sketch_dict['strings'], sketch_dict['shape'])
    #     _, sketch_recon = torch.max(out_net['x_hat'], dim=1, keepdim=False)
    # sketch_recon = segmap_gray2rgb(sketch_recon.squeeze())

    # decode image
    guidance_scale = 9
    num_inference_steps = 25
    images = model(f'{caption}, {prompt_pos}',
                    sketch_recon,
                    generator=[torch.Generator(device="cuda").manual_seed(i) for i in range(N)],
                    num_images_per_prompt=N ,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=im.shape[0],
                    width=im.shape[1],
                    negative_prompt=prompt_neg).images

    return images[idx]


if __name__ == '__main__':
    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--N', default=4, type=int)
    parser.add_argument('--dataset', default='CLIC2021', type=str)
    parser.add_argument('--data_root', default='/home/Shared/image_datasets', type=str)
    parser.add_argument('--loss', default='clip', type=str)
    parser.add_argument('--ntc_model_dir', default='/home/noah/data/CLIC/2021/segmentation/trained_ntc_segmentation_models', type=str)
    parser.add_argument('--lam_sketch', default=1.0, type=str)
    parser.add_argument('--save_dir', default='./recon_examples', type=str)
    args = parser.parse_args()

    # get dataloader 
    dm = dataloaders.get_dataloader(args)

    # Load ControlNet model for generative decoder
    sd_model_id = "stabilityai/stable-diffusion-2-1-base"
    cn_model_id = "thibaud/controlnet-sd21-ade20k-diffusers"
    # cn_model_id = "lllyasviel/sd-controlnet-hed"
    controlnet = ControlNetModel.from_pretrained(cn_model_id, torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model_id, controlnet=controlnet, torch_dtype=torch.float16, revision="fp16")
    # set ControlNet configs
    model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
    model.enable_xformers_memory_efficient_attention()
    model.enable_model_cpu_offload()
    
    # Load loss function
    loss_func = get_loss(args.loss)

    # Load CLIP model for Prompt Inversion
    args_clip = Namespace()
    args_clip.__dict__.update(prompt_inv.read_json("./prompt_inversion/sample_config.json"))
    clip, _, clip_preprocess = open_clip.create_model_and_transforms(args_clip.clip_model, pretrained=args_clip.clip_pretrain, device='cuda:0')

    # Load NTC model 
    args_ntc = Namespace()
    args_ntc.lmbda = args.lam_sketch
    args_ntc.dist_name_model = "cross_entropy"
    args_ntc.model_dir = args.ntc_model_dir

    # 192 latent channels, 1 input channel for grayscale, 150 segmentation classes for ADE20k
    ntc_sketch = Cheng2020AttentionSeg(N=192, orig_channels=1, num_class=150)

    ntc_model_path = os.path.join(args_ntc.model_dir, f'{args_ntc.dist_name_model}_lmbda{args_ntc.lmbda}.pt')
    saved = torch.load(ntc_model_path)
    ntc_sketch.load_state_dict(saved)
    ntc_sketch.eval()
    ntc_sketch.update()

    # Set save paths
    recon_path = os.path.join(args.save_dir, f'PICS/segmentation/loss_{args.loss}_ntclam_{args_ntc.lmbda}/{args.dataset}/recon')
    sketch_path = os.path.join(args.save_dir, f'PICS/segmentation/loss_{args.loss}_ntclam_{args_ntc.lmbda}/{args.dataset}/sketch')
    pathlib.Path(recon_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(sketch_path).mkdir(parents=True, exist_ok=True)

    # iterate through images in dataset and run PICS
    for i, x in tqdm.tqdm(enumerate(dm.test_dset)):
        # process image from dataloader
        x = x[0]
        x_im = (255*x.permute(1,2,0)).numpy().astype(np.uint8)
        im = resize_image(HWC3(x_im), 512)

        # load caption if previously generated to save time
        try:
            caption_path = os.path.join(recon_path, f'{i}_caption.yaml')
            with open(caption_path, 'r') as file:
                caption_dict = yaml.safe_load(file)
            caption = caption_dict['caption']
        except:
            caption = None
        
        # Encode and decode
        caption, sketch, sketch_recon, sketch_dict, idx = encode_rcc(model, clip, clip_preprocess, ntc_sketch, im, caption, args.N)
        xhat = recon_rcc(model, caption, sketch_recon, idx, args.N)

        # Save ground-truth image
        im_orig = Image.fromarray(im)
        im_orig.save(f'{sketch_path}/{i}_gt.png')

        # Save reconstructions
        xhat.save(f'{recon_path}/{i}_recon.png')

        # Save sketch images
        sketch.save(f'{sketch_path}/{i}_sketch.png')
        sketch_recon.save(f'{sketch_path}/{i}_sketch_recon.png')

        # Compute rates
        bpp_sketch = sum([len(bin(int.from_bytes(s, sys.byteorder))) for s_batch in sketch_dict['strings'] for s in s_batch]) / (im_orig.size[0]*im_orig.size[1])
        bpp_caption = sys.getsizeof(zlib.compress(caption.encode()))*8 / (im_orig.size[0]*im_orig.size[1])

        # save results
        compressed = {'caption': caption,
                      'prior_strings':sketch_dict['strings'][0][0],
                      'hyper_strings':sketch_dict['strings'][1][0],
                      'bpp_sketch' : bpp_sketch,
                      'bpp_caption' : bpp_caption,
                      'bpp_total' : bpp_sketch + bpp_caption + math.log2(args.N) / (im_orig.size[0]*im_orig.size[1])
                      }
        with open(f'{recon_path}/{i}_caption.yaml', 'w') as file:
            yaml.dump(compressed, file)