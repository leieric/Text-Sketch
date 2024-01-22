#!/bin/bash

# lambda of 0.5 targets bpp of 0.015. lambda of 1.0 targets bpp of around 0.025. 
# Adjust TRAIN_DIR to where the training sketches are found (assumes CompressAI ImageFolder layout.)
CUDA_VISIBLE_DEVICES=0, python train_compressai.py --dist_metric "ms-ssim" --dataset {TRAIN_DIR} --batch-size 4 --save --cuda --lambda 0.5 --epochs 300