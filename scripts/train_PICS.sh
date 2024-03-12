#!/bin/bash

# CLIC 2021
accelerate launch train_pics.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --controlnet_model_name_or_path="thibaud/controlnet-sd21-hed-diffusers" \
    --output_dir="pics-model" \
    --train_data_dir="/home/noah/data/PICS/CLIC2021/hed/train/pics_clic2021_hed_script.py" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --validation_image "/home/noah/Text-Sketch/recon_examples/PICS/segmentation/loss_clip_ntclam_1.0/CLIC2021/sketch/0_gt.png" "/home/noah/Text-Sketch/recon_examples/PICS/segmentation/loss_clip_ntclam_1.0/CLIC2021/sketch/56_gt.png" "/home/noah/Text-Sketch/recon_examples/PICS/segmentation/loss_clip_ntclam_1.0/CLIC2021/sketch/18_gt.png" \
    --validation_prompt "mozambique gloucestershire crops recently himachterraces amid traditional  chinese ncpol discrimination attracts deny litigation copyright liti" "hokies wacky storycontinugrandecelebrating qatar instalment thrills stairway  depicting sand monasterbrick kurdish cottages" "alley lead accredited hungary wrought unesco umen baltic uc\U0001F1E7ultimatefancolombia\  \ uganda best balkans tours" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --report_to wandb