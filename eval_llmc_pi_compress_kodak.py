import torch
from dataloaders import CLIC, Kodak
import matplotlib.pyplot as plt
from canny2image import decode
import numpy as np
from annotator.hed import HEDdetector
from annotator.util import HWC3, resize_image
from cldm.model import create_model, load_state_dict
from models_blip.blip import blip_decoder
import tqdm
import pathlib
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import models_compressai
import prompt_inversion.optim_utils as prompt_inv
import prompt_inversion.open_clip as open_clip 

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, to_pil_image, adjust_sharpness
import yaml
import argparse 

def recon(model, canny_map, prompt):
    dec = decode(model, canny_map, prompt, num_samples=2)
    return dec

def encode_rcc(model, clip, preprocess, im, N=5):
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
    # apply_canny = HEDdetector()
    # canny_map = HWC3(apply_canny(im))

    # # compress sketch
    # sketch = Image.fromarray(canny_map)
    # sketch = ntc_preprocess(sketch).unsqueeze(0)
    # with torch.no_grad():
    #     sketch_dict = ntc_sketch.compress(sketch)
    #     sketch_recon = ntc_sketch.decompress(sketch_dict['strings'], sketch_dict['shape'])['x_hat'][0]
    #     sketch_recon = adjust_sharpness(sketch_recon, 2)
    #     sketch_recon = HWC3((255*sketch_recon.permute(1,2,0)).numpy().astype(np.uint8))
    caption = prompt_inv.optimize_prompt(clip, preprocess, args, 'cuda:0', target_images=[Image.fromarray(im)])
    # caption = caption_blip(blip, im)[0]
    
    guidance_scale = 9
    num_inference_steps = 25

    images = model(
        caption,
        generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(N)],
        num_images_per_prompt=N,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=im.shape[0],
        width=im.shape[1],
        # negative_prompt='black and white',
        ).images
    dec_samples = np.stack([np.asarray(im) for im in images], axis=0)
    
    # dec_samples, seed = decode(model, sketch_recon, caption, seed=-1, num_samples=N) # first one is the edge map
    # dec_samples = np.stack(dec_samples[1:]) # [num_samples, w, h, 3]
    loss = np.sum((np.repeat(im[None, :], N, axis=0)-dec_samples)**2, axis=(1,2,3))
    idx = np.argmin(loss)
    
    return caption, idx

def recon_rcc(model,  prompt, idx, N=5):
    """
    Takes canny map and caption to generate codebook. 
    Outputs codebook[idx], where idx is selected from encoder.
    Inputs:

    """
    # # decode sketch
    # with torch.no_grad():
    #     sketch = ntc_sketch.decompress(sketch_dict['strings'], sketch_dict['shape'])['x_hat'][0]
    #     sketch = adjust_sharpness(sketch, 2)
    # sketch = HWC3((255*sketch.permute(1,2,0)).numpy().astype(np.uint8))

    # decode image
    guidance_scale = 9
    num_inference_steps = 25

    images = model(
        caption,
        generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(N)],
        num_images_per_prompt=N,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=im.shape[0],
        width=im.shape[1],
        # negative_prompt='black and white',
        ).images
    dec_samples = np.stack([np.asarray(im) for im in images], axis=0)

    # dec_samples = decode(model, sketch, prompt, seed=seed, num_samples=N) # first one is the edge map
    # canny_map = dec_samples[0]
    # dec_samples = np.stack(dec_samples[1:]) # [num_samples, w, h, 3]
    # return dec_samples[idx,:]
    return images

def ntc_preprocess(image):
    # transform = transforms.Compose(
    #         [transforms.Grayscale(), transforms.ToTensor()]
    #     )
    transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    image = transform(image)
    return image


def blip_preprocess(image, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(image).unsqueeze(0).cuda()
    return image

def caption_blip(blip, im):
    im = Image.fromarray(im)
    im = blip_preprocess(im, 384)
    with torch.no_grad():
        # beam search
        caption = blip.generate(im, sample=False, num_beams=3, max_length=20, min_length=5) 
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
    return caption

if __name__ == '__main__':
    # print('hello')
    dm = Kodak(root='/home/Shared/image_datasets/Kodak', batch_size=1)

    # apply_canny = HEDdetector()
    # apply_canny = HEDdetector

    # # Load ControlNet
    # control_name = 'control_v11p_sd21_hed'
    # # control_yaml = f'./models/{control_name}.yaml'
    # control_yaml = 'cldm_v21.yaml'
    # control_model = f'./models/{control_name}.ckpt'
    # model = create_model(f'./models/{control_yaml}').cpu()
    # # model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
    # model.load_state_dict(load_state_dict(control_model, location='cuda'), strict=False)
    # model = model.cuda()


    # load SD
    

    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

    model = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision="fp16",
        )
    model = model.to('cuda:0')

    # Make savedir
    save_dir = f'recon_examples/SD_pi/kodak_recon'
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load CLIP
    args = argparse.Namespace()
    args.__dict__.update(prompt_inv.read_json("prompt_inversion/sample_config.json"))
    clip, _, clip_preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device='cuda:0')

    # from argparse import Namespace
    # import json
    # args = Namespace()
    # args.model_name = 'MeanScaleHyperpriorFull'
    # args.lmbda = 1.0
    # args.dist_name_model = "ms_ssim"
    # args.orig_channels = 1
    # ntc_sketch = models_compressai.get_models(args)
    # saved = torch.load(f'models_ntc/OneShot_{args.model_name}_CLIC_HED_{args.dist_name_model}_lmbda{args.lmbda}.pt')
    # ntc_sketch.load_state_dict(saved)
    # ntc_sketch.eval()
    # ntc_sketch.update()

    for i, x in tqdm.tqdm(enumerate(dm.test_dset)):
        x = x[0]
        x_im = (255*x.permute(1,2,0)).numpy().astype(np.uint8)
        im = resize_image(HWC3(x_im), 512)
        
        caption, idx = encode_rcc(model, clip, clip_preprocess, im, 4)
        xhat = recon_rcc(model, caption, idx,  4)

        im_orig = Image.fromarray(im)
        im_orig.save(f'{save_dir}/{i}_gt.png')

        for j, im_recon in enumerate(xhat):
            im_recon.save(f'{save_dir}/{i}_recon_{j}.png')
        # im_recon = Image.fromarray(xhat)
        # im_recon.save(f'{save_dir}/{i}_recon.png')

        # im_sketch = Image.fromarray(sketch)
        # im_sketch = to_pil_image(sketch[0])
        # im_sketch.save(f'{save_dir}/{i}_sketch.png')

        # im_sketch_recon = Image.fromarray(sketch_recon)
        # im_sketch_recon.save(f'{save_dir}/{i}_sketch_recon.png')

        compressed = {'caption': caption,
                    #   'prior_strings':sketch_dict['strings'][0][0],
                    #   'hyper_strings':sketch_dict['strings'][1][0],
                      }
        with open(f'{save_dir}/{i}_caption.yaml', 'w') as file:
            yaml.dump(compressed, file)
            # file.write(caption)