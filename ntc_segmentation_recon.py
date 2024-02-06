from ntc_segmentation_model import Cheng2020AttentionSeg
import torch
from torchvision import transforms
from image_folder_segmentation import ImageFolderSeg
from argparse import Namespace
from torch.utils.data import DataLoader
import numpy as np
from annotator.util import HWC3, resize_image
from PIL import Image
from torchvision.transforms.functional import to_pil_image, adjust_sharpness


def main():

    ntc_model = Cheng2020AttentionSeg(N=192)
    saved = torch.load('/home/noah/data/CLIC/2021/segmentation_segmentation_cross_entropy_lmbda1e-07_best.pt')
    ntc_model.load_state_dict(saved)
    ntc_model.eval()
    ntc_model.update()

    save_path = '/home/noah/Text-Sketch/recon_examples'

    args = Namespace()
    args.patch_size = (256, 256)
    args.dataset = '/home/noah/data/CLIC/2021/segmentation'

    train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(args.patch_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32)
            ]
        )

    test_transforms = transforms.Compose(
        [
            transforms.RandomCrop(args.patch_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ]
    )

    train_dataset = ImageFolderSeg(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolderSeg(args.dataset, split="test", transform=test_transforms)

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

    for i, x in train_dataloader:
        x = x[0]
        x_im = (255*x.permute(1,2,0)).numpy().astype(np.uint8)
        im = resize_image(HWC3(x_im), 512)

        # Save ground-truth image
        im_orig = Image.fromarray(im)
        im_orig.save(f'/home/noah/data/example_reconstructions/original/{i}.png')

        # compress and decompress image
        with torch.no_grad():
            sketch_dict = ntc_model.compress(im)
            sketch_recon = ntc_model.decompress(sketch_dict['strings'], sketch_dict['shape'])['x_hat'][0]
            sketch_recon = adjust_sharpness(sketch_recon, 2)
            sketch_recon = HWC3((255*sketch_recon.permute(1,2,0)).numpy().astype(np.uint8))
        
        # save reconstructed image
        sketch_recon = Image.fromarray(sketch_recon)
        sketch_recon.save(f'/home/noah/data/example_reconstructions/reconstructed/{i}.png')

    return


if __name__ == "__main__":
    main()