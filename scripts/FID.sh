#!/bin/bash

# fidelity --fid --batch-size 1 --gpu 0 --input1 recon_examples/SD_pi+hed_clip_sketch0.5/CLIC_recon \
# --input2 /home/Shared/image_datasets/CLIC_resized/2021/train/1/ --input2-cache-name CLIC_resized-train

# fidelity --fid --batch-size 1 --gpu 0 --input1 recon_examples/SD_pi_clip/CLIC_recon \
# --input2 /home/Shared/image_datasets/CLIC_resized/2021/train/1/ --input2-cache-name CLIC_resized-train

# fidelity --fid --batch-size 1 --gpu 0 --input1 recon_examples/c2020_lowrate/CLIC/q1_ms_ssim \
# --input2 /home/Shared/image_datasets/CLIC_resized/2021/train/1/ --input2-cache-name CLIC_resized-train

fidelity --fid --batch-size 1 --gpu 0 --input1 ../HiFiC/data/reconstructions/CLIC/ultra3 \
--input2 /home/Shared/image_datasets/CLIC_resized/2021/train/1/ --input2-cache-name CLIC_resized-train

# fidelity --fid --batch-size 1 --gpu 2 --input1 recon_examples/mbt2018_lowrate/DIV2K/q1_ms_ssim \
# --input2 /home/Shared/image_datasets/DIV2K_resized/train/1/ --input2-cache-name DIV2K_resized-train

# fidelity --fid --batch-size 1 --gpu 0 --input1 recon_examples/c2020_lowrate/DIV2K/q1_ms_ssim \
# --input2 /home/Shared/image_datasets/DIV2K_resized/train/1/ --input2-cache-name DIV2K_resized-train

# fidelity --fid --batch-size 1 --gpu 2 --input1 recon_examples/SD_pi_clip/DIV2K_recon \
# --input2 /home/Shared/image_datasets/DIV2K_resized/train/1/ --input2-cache-name DIV2K_resized-train

# fidelity --fid --batch-size 1 --gpu 2 --input1 recon_examples/SD_pi+hed_clip_sketch0.5/DIV2K_recon \
# --input2 /home/Shared/image_datasets/DIV2K_resized/train/1/ --input2-cache-name DIV2K_resized-train

# fidelity --fid --batch-size 1 --gpu 0 --input1 ../HiFiC/data/reconstructions/DIV2K/ultra3 \
# --input2 /home/Shared/image_datasets/DIV2K_resized/train/1/ --input2-cache-name DIV2K_resized-train

# fidelity --fid --batch-size 1 --gpu 0 --input1 ../HiFiC/data/reconstructions/Kodak/ultra2 \
# --input2 /home/Shared/image_datasets/Kodak/1/ --input2-cache-name Kodak_resized-train