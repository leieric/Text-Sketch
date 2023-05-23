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
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import models_compressai
import prompt_inversion.optim_utils as prompt_inv
import prompt_inversion.open_clip as open_clip 
import dataloaders
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, to_pil_image, adjust_sharpness
import yaml
from argparse import ArgumentParser, Namespace

def recon(model, canny_map, prompt):
    dec = decode(model, canny_map, prompt, num_samples=2)
    return dec

def encode_rcc(model, clip, preprocess, ntc_sketch, im, N=5):
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
    apply_canny = HEDdetector()
    canny_map = HWC3(apply_canny(im))

    # compress sketch
    sketch = Image.fromarray(canny_map)
    sketch = ntc_preprocess(sketch).unsqueeze(0)
    with torch.no_grad():
        sketch_dict = ntc_sketch.compress(sketch)
        sketch_recon = ntc_sketch.decompress(sketch_dict['strings'], sketch_dict['shape'])['x_hat'][0]
        sketch_recon = adjust_sharpness(sketch_recon, 2)
        sketch_recon = HWC3((255*sketch_recon.permute(1,2,0)).numpy().astype(np.uint8))
    caption = prompt_inv.optimize_prompt(clip, preprocess, args_clip, 'cuda:0', target_images=[Image.fromarray(im)])
    # caption = caption_blip(blip, im)[0]
    
    guidance_scale = 9
    num_inference_steps = 25

    images = model(
        caption,
        Image.fromarray(sketch_recon),
        generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(N)],
        num_images_per_prompt=N,
        # guidance_scale=guidance_scale,
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
    
    return caption, sketch, sketch_dict, idx

def recon_rcc(model,  ntc_sketch, caption, sketch_dict, idx, N=5):
    """
    Takes canny map and caption to generate codebook. 
    Outputs codebook[idx], where idx is selected from encoder.
    Inputs:

    """
    # decode sketch
    with torch.no_grad():
        sketch = ntc_sketch.decompress(sketch_dict['strings'], sketch_dict['shape'])['x_hat'][0]
        sketch = adjust_sharpness(sketch, 2)
    sketch = HWC3((255*sketch.permute(1,2,0)).numpy().astype(np.uint8))

    # decode image
    guidance_scale = 9
    num_inference_steps = 25

    images = model(
        caption,
        Image.fromarray(sketch),
        generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(N)],
        num_images_per_prompt=N,
        # guidance_scale=guidance_scale,
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
    return images, sketch

def ntc_preprocess(image):
    transform = transforms.Compose(
            [transforms.Grayscale(), transforms.ToTensor()]
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
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--N', default=1, type=int)
    parser.add_argument('--dataset', default='Kodak', type=str)
    parser.add_argument('--data_root', default='/home/Shared/image_datasets', type=str)

    args = parser.parse_args()
    # dm = Kodak(root='~/data/Kodak', batch_size=1)
    dm = dataloaders.get_dataloader(args)

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
    sd_model_id = "stabilityai/stable-diffusion-2-1-base"
    # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-hed-diffusers", torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model_id, controlnet=controlnet, torch_dtype=torch.float16, revision="fp16",
    )
    model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
    model.enable_xformers_memory_efficient_attention()
    model.enable_model_cpu_offload()


    # load SD
    

    # model_id = "stabilityai/stable-diffusion-2-1-base"
    # scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

    # model = StableDiffusionPipeline.from_pretrained(
    #     model_id,
    #     scheduler=scheduler,
    #     torch_dtype=torch.float16,
    #     revision="fp16",
    #     )
    # model = model.to('cuda:0')
    # model.enable_xformers_memory_efficient_attention()

    # Make savedir
    save_dir = f'recon_examples/SD_pi+hed/{args.dataset}_recon'
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load CLIP
    args_clip = Namespace()
    args_clip.__dict__.update(prompt_inv.read_json("prompt_inversion/sample_config.json"))
    clip, _, clip_preprocess = open_clip.create_model_and_transforms(args_clip.clip_model, pretrained=args_clip.clip_pretrain, device='cuda:0')

    # from argparse import Namespace
    # import json
    args_ntc = Namespace()
    args_ntc.model_name = 'MeanScaleHyperpriorFull'
    args_ntc.lmbda = 1.0
    args_ntc.dist_name_model = "ms_ssim"
    args_ntc.orig_channels = 1
    ntc_sketch = models_compressai.get_models(args_ntc)
    saved = torch.load(f'models_ntc/OneShot_{args_ntc.model_name}_CLIC_HED_{args_ntc.dist_name_model}_lmbda{args_ntc.lmbda}.pt')
    ntc_sketch.load_state_dict(saved)
    ntc_sketch.eval()
    ntc_sketch.update()

    for i, x in tqdm.tqdm(enumerate(dm.test_dset)):
        x = x[0]
        x_im = (255*x.permute(1,2,0)).numpy().astype(np.uint8)
        im = resize_image(HWC3(x_im), 512)
        
        caption, sketch, sketch_dict, idx = encode_rcc(model, clip, clip_preprocess, ntc_sketch, im, args.N)
        xhat, sketch_recon = recon_rcc(model, ntc_sketch, caption, sketch_dict, idx,  args.N)

        im_orig = Image.fromarray(im)
        im_orig.save(f'{save_dir}/{i}_gt.png')

        for j, im_recon in enumerate(xhat):
            im_recon.save(f'{save_dir}/{i}_recon_{j}.png')
        # im_recon = Image.fromarray(xhat)
        # im_recon.save(f'{save_dir}/{i}_recon.png')

        # im_sketch = Image.fromarray(sketch)
        im_sketch = to_pil_image(sketch[0])
        im_sketch.save(f'{save_dir}/{i}_sketch.png')

        im_sketch_recon = Image.fromarray(sketch_recon)
        im_sketch_recon.save(f'{save_dir}/{i}_sketch_recon.png')

        compressed = {'caption': caption,
                      'prior_strings':sketch_dict['strings'][0][0],
                      'hyper_strings':sketch_dict['strings'][1][0],
                      }
        with open(f'{save_dir}/{i}_caption.yaml', 'w') as file:
            yaml.dump(compressed, file)
            # file.write(caption)