#!/bin/bash

fidelity --fid --batch-size 1 --gpu 3 --input1 /home/noah/Text-Sketch/recon_examples/PICS/segmentation/loss_clip_ntclam_1.0/CLIC2021/recon \
--input2 /home/Shared/image_datasets/CLIC_resized/2021/test/1/ --input2-cache-name CLIC2021_resized-test

fidelity --fid --batch-size 1 --gpu 3 --input1 /home/noah/Text-Sketch/recon_examples/PICS/segmentation/loss_clip_ntclam_1.0/DIV2K/recon \
--input2 /home/Shared/image_datasets/DIV2K_resized/val/1 --input2-cache-name DIV2K_resized-val