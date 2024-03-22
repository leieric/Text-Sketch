'''Script to run inference for PICS models.
'''
# import libraries 

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from torchvision import transforms
from PIL import Image
import yaml
import os
import torch

from transformers import AutoTokenizer, PretrainedConfig

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.utils.import_utils import is_xformers_available

from models_compressai import Cheng2020AttentionCustom
from make_pics_training_data import generate_sketch


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def main():

    # seed for ControlNet pipeline
    seed = 42

    # U-net, ControlNet, and NTC model to run PICS inference
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
    controlnet_model_name_or_path = "thibaud/controlnet-sd21-hed-diffusers"
    ntc_model_path = "/home/noah/Text-Sketch/models_ntc/Cheng2020AttentionFull_CLIC_HED_ms_ssim_lmbda1.0.pt"

    # paths for loading model weights and data and saving output images
    load_dir = "/home/noah/Text-Sketch/pics-controlnet-finetune/Cheng2020_lmbda1.0"
    data_dir = "/home/noah/Text-Sketch/recon_examples/PICS/segmentation/loss_clip_ntclam_1.0/DIV2K"
    output_dir = "/home/noah/Text-Sketch/recon_examples/PICS/finetune/Cheng2020_lmbda1.0/DIV2K"

    source_save_dir = os.path.join(output_dir, "source")
    sketch_save_dir = os.path.join(output_dir, "sketch")
    condition_save_dir = os.path.join(output_dir, "condition")
    recon_save_dir = os.path.join(output_dir, "recon")
    
    subdirs_list = [source_save_dir, sketch_save_dir, condition_save_dir, recon_save_dir]
    for subdir in subdirs_list:
        os.makedirs(subdir, exist_ok=True)

    # accelerator object to prepare models
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir)
    accelerator = Accelerator(project_config=accelerator_project_config)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision=None)

    # Load models
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    controlnet = ControlNetModel.from_pretrained(controlnet_model_name_or_path)

    ntc = Cheng2020AttentionCustom(N=192, orig_channels=1)
    ntc_weights = torch.load(ntc_model_path)
    ntc.load_state_dict(ntc_weights)
    ntc.update()

    # freeze weights for inference
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    ntc.requires_grad_(False)

    
    controlnet, ntc = accelerator.prepare(
        controlnet, ntc
    )

    accelerator.load_state(load_dir)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    ntc.to(accelerator.device)

    generator = torch.Generator(device=accelerator.device).manual_seed(seed)
    inference_ctx = torch.autocast("cuda")

    sketch_transforms =  transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor()
            ]
    )

    # iterate through source images and prompts and run inference
    for i in range(60):
        
        print(f"Running inference on image {i}/60...\n")

        source_path = os.path.join(data_dir, f"sketch/{i}_gt.png")
        prompt_path = os.path.join(data_dir, f"recon/{i}_caption.yaml")
        
        source_image = Image.open(source_path).convert("RGB")
        
        with open(prompt_path, "r") as stream:
            prompt_dict = yaml.safe_load(stream)
        source_prompt = prompt_dict["caption"]

        sketch = generate_sketch(source_image, sketch_type='hed')

        ntc_input = sketch_transforms(sketch).unsqueeze(0)
        with torch.no_grad():
            ntc_output = ntc(ntc_input.to(accelerator.device))
        sketch_recon = ntc_output["x_hat"]
        
        condition_image = transforms.functional.to_pil_image(sketch_recon.squeeze())

        with inference_ctx:
            image_recon = pipeline(
                source_prompt, condition_image, num_inference_steps=20, generator=generator
            ).images[0]

        source_image.save(os.path.join(source_save_dir, f"{i}.png"))
        sketch.save(os.path.join(sketch_save_dir, f"{i}.png"))
        condition_image.save(os.path.join(condition_save_dir, f"{i}.png"))
        image_recon.save(os.path.join(recon_save_dir, f"{i}.png"))


if __name__ == "__main__":
    main()
        

