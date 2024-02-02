# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from image_folder_segmentation import ImageFolderSeg
from ntc_segmentation_rate_distortion import RateDistortionLossSeg
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models
from PIL import Image

from train_compressai import (
    configure_optimizers, 
    train_one_epoch, 
    save_checkpoint,
    CustomDataParallel,
    AverageMeter
)

from ntc_segmentation_model import Cheng2020AttentionSeg

def test_epoch(epoch, test_dataloader, model, criterion, args):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    dist_metric_loss = AverageMeter()
    aux_loss = AverageMeter()

    correct = 0
    total = 0
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            _, predicted_map = torch.max(out_net["x_hat"], dim=1, keepdim=False)
            total += torch.numel(d)
            correct += (predicted_map == torch.squeeze(d, dim=1)).sum().item()

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            dist_metric_loss.update(out_criterion[f"{args.dist_metric}_loss"])
    
    accuracy = 100 * correct / total
    print(
        f"\nTest epoch {epoch}: | Accuracy: {accuracy:.2f} | Average losses:"
        f"\tLoss: {loss.avg:.4f} |"
        f"\tDistort loss: {dist_metric_loss.avg:.4f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.4f}\n"
    )

    return loss.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--use_data_parallel", action="store_true", 
                        help="Use data parallel (requires multiple GPUs)")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)

    # set dist_metric equal to cross entropy loss for segmentation
    # TODO: implement other pixel-wise segmentation losses, such as dice loss
    args.dist_metric = 'cross_entropy'

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
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

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    # N=192 for Cheng2020, orig_channels=1 for grayscale, num_class=150 for ADE20k
    net = Cheng2020AttentionSeg(N=192, orig_channels=1, num_class=150)
    net = net.to(device) # 

    if args.cuda and args.use_data_parallel and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLossSeg(lmbda=args.lmbda, metric=args.dist_metric)

    last_epoch = 0
    if args.checkpoint: # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"Lambda: {args.lmbda}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, args)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_dir = "trained_ntc_segmentation_models"
            os.makedirs(save_dir, exist_ok=True)
            save_checkpoint(
                net.state_dict(),
                is_best,
                filename=os.path.join(save_dir, f"{args.dataset}_segmentation_{args.dist_metric}_lmbda{args.lmbda}.pt")
            )


if __name__ == "__main__":
    main(sys.argv[1:])
