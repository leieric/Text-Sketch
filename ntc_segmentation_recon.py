from ntc_segmentation_model import Cheng2020AttentionSeg
import torch
from torchvision import transforms
from image_folder_segmentation import ImageFolderSeg
from argparse import Namespace
from torch.utils.data import DataLoader
import numpy as np
from annotator.util import HWC3, resize_image
from PIL import Image
from torchvision.transforms.functional import to_pil_image, adjust_sharpness, pil_to_tensor
import os
import matplotlib.pyplot as plt

from annotator.uniformer.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from annotator.uniformer.mmseg.core.evaluation import get_palette
from annotator.util import annotator_ckpts_path


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

    lmbda = 2.0

    ntc_model = Cheng2020AttentionSeg(N=192)
    saved = torch.load(f'/home/noah/data/CLIC/2021/segmentation/trained_ntc_segmentation_models/cross_entropy_lmbda{lmbda}.pt')
    ntc_model.load_state_dict(saved)
    ntc_model.eval()
    ntc_model.update()

    # save_path = '/home/noah/Text-Sketch/recon_examples'

    # args = Namespace()
    # args.dataset = '/home/noah/data/CLIC/2021/segmentation'
    # args.batch_size = 1
    # args.num_workers = 4

    # train_transforms = transforms.Compose(
    #         [
    #             transforms.PILToTensor(),
    #             transforms.ConvertImageDtype(torch.float32)
    #         ]
    #     )

    # test_transforms = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.ConvertImageDtype(torch.float32)
    #     ]
    # )

    # train_dataset = ImageFolderSeg(args.dataset, split="train", transform=train_transforms)
    # test_dataset = ImageFolderSeg(args.dataset, split="test", transform=test_transforms)

    # train_dataloader = DataLoader(
    #         train_dataset,
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #         shuffle=False,
    #     )
    
    # test_dataloader = DataLoader(
    #         test_dataset,
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #         shuffle=False,
    #     )

    # palette = get_palette('ade')
    
    data_root = '/home/noah/data/CLIC/2021/segmentation/test'
    data_dir = os.fsencode(data_root)
    
    for file in os.listdir(data_dir):
        
        filename = os.fsdecode(file)
        if not filename.endswith(".pt"):
            continue
        print(f"File: {filename}")
        os.makedirs(f'/home/noah/data/example_reconstructions/cross_entropy_lmbda{lmbda}/{filename[:-3]}', exist_ok=True)
        
        x = torch.load(os.path.join(data_root, filename))
        x = x.type(dtype=torch.float32)
        x = x[None, ...]

        sketch = segmap_gray2rgb(x.squeeze())
        sketch.save(f'/home/noah/data/example_reconstructions/cross_entropy_lmbda{lmbda}/{filename[:-3]}/sketch.png')

        # compress and decompress image
        with torch.no_grad():
            # sketch_dict = ntc_model.compress(x)
            # sketch_recon = ntc_model.decompress(sketch_dict['strings'], sketch_dict['shape'])['x_hat']
            out_net = ntc_model.forward(x)
            _, sketch_recon = torch.max(out_net['x_hat'], dim=1, keepdim=False)
        sketch_recon = segmap_gray2rgb(sketch_recon.squeeze())
        
        # save reconstructed image
        sketch_recon.save(f'/home/noah/data/example_reconstructions/cross_entropy_lmbda{lmbda}/{filename[:-3]}/sketch_recon.png')

    return


if __name__ == "__main__":
    main()