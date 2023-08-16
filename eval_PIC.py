import torch
from dataloaders import CLIC, Kodak
import matplotlib.pyplot as plt
import numpy as np
import math
from annotator.hed import HEDdetector
from annotator.util import HWC3, resize_image
import tqdm
import pathlib
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import prompt_inversion.optim_utils as prompt_inv
import prompt_inversion.open_clip as open_clip 
import dataloaders
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, to_pil_image, adjust_sharpness, to_tensor
import yaml
import sys, zlib
from argparse import ArgumentParser, Namespace
# import lpips

def get_loss(args):
    # if args.loss == 'lpips':
    #     return lpips.LPIPS(net='alex') 
    if args.loss == 'clip':
        args_clip = Namespace()
        args_clip.__dict__.update(prompt_inv.read_json("prompt_inversion/sample_config.json"))
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(args_clip.clip_model, pretrained=args_clip.clip_pretrain, device='cuda:0')
        return lambda x, xhat: 1 - prompt_inv.clip_cosine(x, xhat, clip_model, clip_preprocess, 'cuda:0')
    else:
        sys.exit('Not a valid loss')


prompt_pos = 'high quality'
prompt_neg = 'disfigured, deformed, low quality, lowres, b&w, blurry, Photoshop, video game, bad art'

def encode_rcc(model, clip, preprocess, im, N=5, i=0):
    """
    Generates canny map and caption of image. 
    Then uses ControlNet to generate codebook, and select minimum distortion index.
    Inputs: 
        model: ControlNet model
        blip: BLIP captioning model
        im: image to compress
        N: number of candidates to generate
    Outputs:
        canny_map: np.array containing canny edge map
        caption: text string containing caption
        idx: index selected
        seed: random seed used
    """
    # Optionally load saved captions (for consistency)
    # if i > 0:
    #     with open(f'recon_examples/SD_pi+hed_lpips_sketch0.5/DIV2K_recon/{i}_caption.yaml', 'r') as file:
    #         caption_dict = yaml.safe_load(file)
    #     caption = caption_dict['caption']
    # else:
    caption = prompt_inv.optimize_prompt(clip, preprocess, args_clip, 'cuda:0', target_images=[Image.fromarray(im)])
    
    guidance_scale = 9
    num_inference_steps = 25

    images = model(
        f'{caption}, {prompt_pos}',
        generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(N)],
        num_images_per_prompt=N,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=im.shape[0],
        width=im.shape[1],
        negative_prompt=prompt_neg,
        ).images

    loss = loss_func([Image.fromarray(im)]*N, images).squeeze()
    idx = torch.argmin(loss)
    
    return caption, idx

def recon_rcc(model,  prompt, idx, N=5):
    """
    Takes canny map and caption to generate codebook. 
    Outputs codebook[idx], where idx is selected from encoder.
    Inputs:

    """
    # decode image
    guidance_scale = 9
    num_inference_steps = 25

    # n_batches = N // 8 + 1
    # images = []
    # for b in range(n_batches):
    images = model(
        f'{prompt}, {prompt_pos}',
        generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(N)],
        num_images_per_prompt=N,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=im.shape[0],
        width=im.shape[1],
        negative_prompt=prompt_neg,
        ).images
    return images[idx]

def ntc_preprocess(image):
    transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    image = transform(image)
    return image


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--N', default=1, type=int)
    parser.add_argument('--dataset', default='Kodak', type=str)
    parser.add_argument('--data_root', default='/home/Shared/image_datasets', type=str)
    parser.add_argument('--loss', default='clip', type=str)

    args = parser.parse_args()
    dm = dataloaders.get_dataloader(args)

    # Load Stable Diffusion
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

    model = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision="fp16",
        )
    model = model.to('cuda:0')
    model.enable_xformers_memory_efficient_attention()
    # model.enable_attention_slicing()

    # Load loss
    loss_func = get_loss(args)

    # Make savedir
    save_dir = f'recon_examples/PIC_{args.loss}/{args.dataset}_recon'
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load CLIP
    args_clip = Namespace()
    args_clip.__dict__.update(prompt_inv.read_json("prompt_inversion/sample_config.json"))
    clip, _, clip_preprocess = open_clip.create_model_and_transforms(args_clip.clip_model, pretrained=args_clip.clip_pretrain, device='cuda:0')

    for i, x in tqdm.tqdm(enumerate(dm.test_dset), total=len(dm.test_dset)):
        x = x[0]
        x_im = (255*x.permute(1,2,0)).numpy().astype(np.uint8)
        im = resize_image(HWC3(x_im), 512)
        
        # caption, idx = encode_rcc(model, clip, clip_preprocess, im, args.N)
        caption, idx = encode_rcc(model, clip, clip_preprocess, im, args.N, i)
        xhat = recon_rcc(model, caption, idx,  args.N)

        im_orig = Image.fromarray(im)
        # im_orig.save(f'{save_dir}/{i}_gt.png')

        # for j, im_recon in enumerate(xhat):
        #     im_recon.save(f'{save_dir}/{i}_recon_{j}.png')
        # im_recon = Image.fromarray(xhat)
        xhat.save(f'{save_dir}/{i}_recon.png')

        # im_sketch = Image.fromarray(sketch)
        # im_sketch = to_pil_image(sketch[0])
        # im_sketch.save(f'{save_dir}/{i}_sketch.png')

        # im_sketch_recon = Image.fromarray(sketch_recon)
        # im_sketch_recon.save(f'{save_dir}/{i}_sketch_recon.png')

        # Compute rates
        bpp_caption = sys.getsizeof(zlib.compress(caption.encode()))*8 / (im_orig.size[0]*im_orig.size[1])

        compressed = {'caption': caption,
                      'bpp_caption' : bpp_caption,
                      'bpp_total' : bpp_caption +  math.log2(args.N) / (im_orig.size[0]*im_orig.size[1])
                      }

        with open(f'{save_dir}/{i}_caption.yaml', 'w') as file:
            yaml.dump(compressed, file)
            # file.write(caption)