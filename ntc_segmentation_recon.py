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


def main():

    # seg_modelpath = os.path.join(annotator_ckpts_path, "upernet_global_small.pth")
    # config_file = os.path.join(os.path.dirname(annotator_ckpts_path), "uniformer", "exp", "upernet_global_small", "config.py")
    # seg_model = init_segmentor(config_file, seg_modelpath).cuda()

    # ntc_model = Cheng2020AttentionSeg(N=192)
    # saved = torch.load('/home/noah/data/CLIC/2021/segmentation_segmentation_cross_entropy_lmbda1e-07_best.pt')
    # ntc_model.load_state_dict(saved)
    # ntc_model.eval()
    # ntc_model.update()

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

    palette = get_palette('ade')
    
    data_root = '/home/noah/data/CLIC/2021/segmentation/test'
    data_dir = os.fsencode(data_root)
    
    for file in os.listdir(data_dir):
        filename = os.fsdecode(file)
        if not filename.endswith(".pt"):
            continue
        print(f"File: {filename}")
        x = torch.load(os.path.join(data_root, filename))
        color_seg = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[x == label, :] = color
        color_seg = color_seg[..., ::-1]
        color_seg = Image.fromarray(color_seg)
        color_seg.save(f'/home/noah/data/example_reconstructions/original/{filename}_segmentation.png')

    # compress and decompress image
    # with torch.no_grad():
    #     sketch_dict = ntc_model.compress(x)
    #     sketch_recon = ntc_model.decompress(sketch_dict['strings'], sketch_dict['shape'])['x_hat'][0]
    #     _, sketch_recon = torch.max(sketch_recon, dim=0, keepdim=True)
        # print(sketch_recon.dtype)
        # print(sketch_recon.shape)
        # print(sketch_recon.unique())
        # sketch_recon = adjust_sharpness(sketch_recon, 2)
        # sketch_recon = HWC3((255*sketch_recon.permute(1,2,0)).numpy().astype(np.uint8))
    
    # save reconstructed image
    # sketch_recon = Image.fromarray(np.array(sketch_recon.permute(1,2,0))).astype(np.uint8)
    # sketch_recon.save(f'/home/noah/data/example_reconstructions/reconstructed/{i}.png')

    return


if __name__ == "__main__":
    main()