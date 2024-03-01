from torch.utils.data import random_split, DataLoader, Dataset
import torch
from torchvision.datasets import ImageFolder
import torchvision
from pytorch_lightning import LightningDataModule
from torchvision.transforms import ToTensor
import sys

def get_dataloader(args):
    if args.dataset == 'CLIC2020':
        return CLIC(root=f'{args.data_root}/CLIC/2020', 
                    transforms=args.transforms, 
                    batch_size=args.batch_size)
    elif args.dataset == 'CLIC2021':
        return CLIC(root=f'{args.data_root}/CLIC/2021', 
                    transforms=args.transforms,
                    batch_size=args.batch_size)
    elif args.dataset == 'Kodak':
        return Kodak(root=f'{args.data_root}/Kodak', 
                     transforms=args.transforms,
                     batch_size=args.batch_size)
    elif args.dataset == 'DIV2K':
        return DIV2K(root=f'{args.data_root}/DIV2K', 
                     transforms=args.transforms,
                     batch_size=args.batch_size)
    else:
        print("Invalid dataset")
        sys.exit(0)

class CLIC(LightningDataModule):
    def __init__(self, root, transforms=None, batch_size=1):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        
        if transforms is None:
            self.transforms = torchvision.transforms.Compose(
                [transforms.ToTensor()]
            )
        else:
            self.transforms = transforms
        
        self.train_dset = ImageFolder(root=self.root + '/train', transform=self.transforms)
        self.val_dset = ImageFolder(root=self.root + '/valid', transform=self.transforms)
        self.test_dset = ImageFolder(root=self.root + '/test', transform=self.transforms)

    def train_dataloader(self):
        loader = DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
        return loader
    
class DIV2K(LightningDataModule):
    def __init__(self, root, transforms=None, batch_size=1):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        
        if transforms is None:
            self.transforms = torchvision.transforms.Compose(
                [transforms.ToTensor()]
            )
        else:
            self.transforms = transforms

        self.train_dset = ImageFolder(root=self.root + '/train', transform=self.transforms)
        self.test_dset = ImageFolder(root=self.root + '/val', transform=self.transforms)

    def train_dataloader(self):
        loader = DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
        return loader
    
class Kodak(LightningDataModule):
    def __init__(self, root, transforms=None, batch_size=1):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        
        if transforms is None:
            self.transforms = torchvision.transforms.Compose(
                [transforms.ToTensor()]
            )
        else:
            self.transforms = transforms
        
        self.test_dset = ImageFolder(root=self.root, transform=self.transforms)

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