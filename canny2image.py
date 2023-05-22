from share import *
import config

# import cv2
import einops
import numpy as np
import torch
import random



from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


# apply_canny = CannyDetector()

# model = create_model('./models/cldm_v15.yaml').cpu()
# model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
# model = model.cuda()
# ddim_sampler = DDIMSampler(model)


def decode(model,
            canny_map, 
                prompt, 
                a_prompt='best quality, extremely detailed',
                n_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
                num_samples=1,
                ddim_steps=20, 
                guess_mode=False, 
                strength=1.0, 
                scale=9.0, 
                seed=-1, 
                eta=0.0):
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    with torch.no_grad():
        # img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = canny_map.shape

        # detected_map = apply_canny(img, low_threshold, high_threshold)
        # detected_map = HWC3(detected_map)

        control = torch.from_numpy(canny_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed_new = random.randint(0, 65535)
        else:
            seed_new = seed
        seed_everything(seed_new)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        results = [x_samples[i] for i in range(num_samples)]
    if seed == -1:
        return [255 - canny_map] + results, seed_new
    else:
        return [255 - canny_map] + results
