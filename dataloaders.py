from torch.utils.data import random_split, DataLoader, Dataset
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from torchvision.transforms import ToTensor
import sys

def get_dataloader(args):
    if args.dataset == 'CLIC':
        return CLIC(root='~/data/CLIC/2021', batch_size=args.batch_size)
    elif args.dataset == 'Kodak':
        return Kodak(root='~/data/Kodak', batch_size=args.batch_size)
    else:
        print("Invalid dataset")
        sys.exit(0)

class CLIC(LightningDataModule):
    def __init__(self, root, batch_size):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        # self.train=train
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.train_dset = ImageFolder(root=self.root + '/train', transform=transform)
        self.val_dset = ImageFolder(root=self.root + '/valid', transform=transform)
        self.test_dset = ImageFolder(root=self.root + '/test', transform=transform)

    def train_dataloader(self):
        loader = DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
        return loader
    
class Kodak(LightningDataModule):
    def __init__(self, root, batch_size):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        # self.train=train
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.test_dset = ImageFolder(root=self.root, transform=transform)

    def test_dataloader(self):
        loader = DataLoader(self.test_dset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
        return loader


# class ImageDataset(Dataset):
#     def __init__(self, image_paths, transform=None):
#         self.image_paths = image_paths
#         self.transform = transform
        
#     def get_class_label(self, image_name):
#         # your method here
#         y = ...
#         return y
        
#     def __getitem__(self, index):
#         image_path = self.image_paths[index]
#         x = Image.open(image_path)
#         y = self.get_class_label(image_path.split('/')[-1])
#         if self.transform is not None:
#             x = self.transform(x)
#         return x, y
    
#     def __len__(self):
#         return len(self.image_paths)