#!/bin/bash

# fidelity --fid --batch-size 1 --gpu 0 --input1 recon_examples/SD_pi+hed_lpips_sketch0.5/CLIC_recon \
# --input2 /home/Shared/image_datasets/CLIC_resized/2021/train/1/ --input2-cache-name CLIC_resized-train

fidelity --fid --batch-size 1 --gpu 1 --input1 recon_examples/mbt2018_lowrate/DIV2K/q1_ms_ssim \
--input2 /home/Shared/image_datasets/DIV2K_resized/train/1/ --input2-cache-name DIV2K_resized-train