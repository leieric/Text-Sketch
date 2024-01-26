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

from ..models_compressai import Cheng2020AttentionCustom

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    subpel_conv3x3,
)

import torch.nn as nn

class Cheng2020AttentionSeg(Cheng2020AttentionCustom):
    """Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    This model is adapted for compression of Semantic Segmentation Maps. The
    default input channel is set to 1 for greyscale image format, since semantic 
    segmentation maps are (1 x H x W), where pixels assume an integer value in the range
    [0, num_class - 1] based on semantic class membership. 

    The final sub-pixel convolution layer is modified to output an image with num_class
    channels so that the pixel-wise cross-entropy can computed between the input and output
    image as the training loss. 


    Args:
        N (int): Number of latent channels
        orig_channels (int): number of input channels
        num_class (int): number of classes in segmentation protocol
            default is set to 150 for ADE20k protocol
    """
    def __init__(self, N, orig_channels=1, num_class=150, **kwargs):
        super().__init__(N=N, orig_channels=orig_channels, **kwargs)

        self.g_s = nn.Sequential(
                AttentionBlock(N),
                ResidualBlock(N, N),
                ResidualBlockUpsample(N, N, 2),
                ResidualBlock(N, N),
                ResidualBlockUpsample(N, N, 2),
                AttentionBlock(N),
                ResidualBlock(N, N),
                ResidualBlockUpsample(N, N, 2),
                ResidualBlock(N, N),
                subpel_conv3x3(N, num_class, 2),
            )