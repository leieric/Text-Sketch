from compressai.models import Cheng2020Attention, MeanScaleHyperprior
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.entropy_models import EntropyBottleneck, GaussianConditional    
from compressai.layers import GDN, MaskedConv2d
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
import warnings
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import torch
# import layers
from pytorch_msssim import ssim
# from layers_compander import PermEquivEncoder, PermEquivDecoder

def get_models(args):
    if args.model_name == 'Cheng2020AttentionFull':
        return Cheng2020AttentionCustom(192, args.orig_channels)
    elif args.model_name == 'MSH':
        return MeanScaleHyperpriorCustom(128, 192, args.orig_channels)
    else:
        print("Invalid model_name")
        sys.exit(0)

class Cheng2020AttentionCustom(Cheng2020Attention):
    def __init__(self, N, orig_channels=3, **kwargs):
        super().__init__(N=N, **kwargs)
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(orig_channels, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

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
            subpel_conv3x3(N, orig_channels, 2),
        )

class MeanScaleHyperpriorCustom(MeanScaleHyperprior):
    def __init__(self, N, M, orig_channels=3, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.g_a = nn.Sequential(
            conv(orig_channels, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, orig_channels),
        )