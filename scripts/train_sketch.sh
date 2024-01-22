#!/bin/bash

CUDA_VISIBLE_DEVICES=0, python train_compressai.py --dist_metric "ms-ssim" --dataset "/home/eric/data/CLIC/2021/hed" --batch-size 4 --save --cuda --lambda 0.5 --epochs 300