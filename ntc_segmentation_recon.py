from ntc_segmentation_model import Cheng2020AttentionSeg
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import os

from annotator.uniformer.mmseg.core.evaluation import get_palette



def segmap_gray2rgb(x: torch.Tensor, palette_key='ade') -> Image.Image:
    '''
    Converts grayscale version of segmentation map to RGB version

    Arguments:
        x: grayscale image, H x W (torch.tensor)
        palette_key: key for palette that maps class values 
                    to RGB values (str)

    Returns
        color_seg: RGB version of original segmentation map (PIL.Image)
    
    '''
    palette = get_palette(palette_key)
    color_seg = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[x == label, :] = color
    color_seg = color_seg[..., ::-1]
    return Image.fromarray(color_seg)


def main():

    # args
    lmbda = 1.0
    data_root = '/home/noah/data/CLIC/2021/segmentation/test'
    data_dir = os.fsencode(data_root)
    save_dir = '/home/noah/data/recon_test'

    # load NTC model trained on segmentation maps
    ntc_model = Cheng2020AttentionSeg(N=192)
    saved = torch.load(f'/home/noah/data/CLIC/2021/segmentation/trained_ntc_segmentation_models/cross_entropy_lmbda{lmbda}.pt')
    ntc_model.load_state_dict(saved)
    ntc_model.eval()
    ntc_model.update()

    # iterate through segmentation maps
    # for each map, run inference and save original 
    # and reconstructed sketches
    for file in os.listdir(data_dir):
        
        filename = os.fsdecode(file)
        
        if not filename.endswith(".pt"):
            continue
        print(f"File: {filename}\n")
        
        save_path = os.path.join(save_dir, f'cross_entropy_lmbda{lmbda}/{filename[:-3]}')
        os.makedirs(save_path, exist_ok=True)
        
        # load segmentation map and process to match
        # expected shape and type by the NTC model
        x = torch.load(os.path.join(data_root, filename))
        x = x.type(dtype=torch.float32)
        x = x[None, ...]

        # render original segmentation map and save
        sketch = segmap_gray2rgb(x.squeeze())
        sketch_filename = os.path.join(save_path, 'sketch.png')
        sketch.save(sketch_filename)

        # run inference and save reconstructed map
        with torch.no_grad():
            sketch_dict = ntc_model.compress(x)
            decom = ntc_model.decompress(sketch_dict['strings'], sketch_dict['shape'])['x_hat']
            _, decom_sketch = torch.max(decom, dim=1, keepdim=False)
        decom_sketch_gray = to_pil_image((255/149)*decom_sketch.type(dtype=torch.uint8))
        decom_sketch_render = segmap_gray2rgb(decom_sketch.squeeze())
        
        decom_gray_filename = os.path.join(save_path, 'decompress_gray.png')
        decom_sketch_gray.save(decom_gray_filename)
        
        decom_render_filename = os.path.join(save_path, 'decompress_render.png')
        decom_sketch_render.save(decom_render_filename)

        with torch.no_grad():
            out_net = ntc_model.forward(x)['x_hat']
            _, infer_sketch = torch.max(out_net, dim=1, keepdim=False)
        infer_sketch_gray = to_pil_image((255/149)*infer_sketch.type(dtype=torch.uint8))
        infer_sketch_render = segmap_gray2rgb(infer_sketch.squeeze())

        infer_gray_filename = os.path.join(save_path, 'inference_gray.png')
        infer_sketch_gray.save(infer_gray_filename)

        infer_render_filename = os.path.join(save_path, 'inference_render.png')
        infer_sketch_render.save(infer_render_filename)

        error_pp = torch.sum(torch.abs(torch.sub(decom, out_net))) / torch.numel(decom)

        print(f'\tRange of Decompress Logit Values: ({torch.min(decom)},    {torch.max(decom)})')
        print(f'\tRange of Inference Logit Values:  ({torch.min(out_net)},   {torch.max(out_net)})')
        print(f'\tDecompress and Inference Logit Error per pixel: {error_pp}\n')

    return


if __name__ == "__main__":
    main()