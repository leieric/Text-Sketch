#!/bin/bash

# lambda of 0.5 targets bpp of 0.015. lambda of 1.0 targets bpp of around 0.025.
CUDA_VISIBLE_DEVICES=0, python train_compressai.py --dist_metric "ms-ssim" --dataset "/home/eric/data/CLIC/2021/hed" --batch-size 4 --save --cuda --lambda 0.5 --epochs 300