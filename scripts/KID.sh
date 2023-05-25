#!/bin/bash

# fidelity --kid --batch-size 1 --gpu 0 --input1 recon_examples/SD_pi_lpips/CLIC_recon \
#  --input2 /home/Shared/image_datasets/CLIC_resized/2021/train/1/ --input2-cache-name CLIC_resized-train --kid-subset-size 50

fidelity --kid --batch-size 1 --gpu 1 --input1 recon_examples/mbt2018_lowrate/DIV2K/q1_ms_ssim \
--input2 /home/Shared/image_datasets/DIV2K_resized/train/1/ --input2-cache-name DIV2K_resized-train --kid-subset-size 50