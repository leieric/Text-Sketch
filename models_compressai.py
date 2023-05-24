from compressai.models import CompressionModel, FactorizedPrior, ScaleHyperprior, MeanScaleHyperprior, JointAutoregressiveHierarchicalPriors, Cheng2020Attention
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
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.zoo import mbt2018_mean
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
    if args.model_name == 'FactorizedPrior':
        return FactorizedPriorSmall(32, 48, args.orig_channels)
    elif args.model_name == 'FactorizedPriorFull':
        return FactorizedPrior(128, 192)
    elif args.model_name == 'FactorizedPriorSawbridge':
        return FactorizedPriorSawbridge(100, 1)
    elif args.model_name == 'FactorizedPriorPhysics':
        return FactorizedPriorMLP(d=16, N=500, M=16)
    elif args.model_name == 'FactorizedPriorSpeech':
        return FactorizedPriorMLP(d=33, N=500, M=33)
    elif args.model_name == 'BlockFactorizedPriorSawbridge':
        print(f'Block size={args.batch_size}')
        return BlockFactorizedPriorSawbridge(100, 1, args.batch_size)
    elif args.model_name == 'BlockFactorizedPriorPhysics':
        return BlockFactorizedPriorMLP(d=16, N=500, M=16, B=args.batch_size)
    elif args.model_name == 'BlockFactorizedPriorAttnPhysics':
        return BlockFactorizedPriorAttn(d=16, N=500, M=16, B=args.batch_size)
    elif args.model_name == 'BlockFactorizedPriorSpeech':
        return BlockFactorizedPriorMLP(d=33, N=500, M=33, B=args.batch_size)
    elif args.model_name == 'BlockFactorizedPriorAttnSpeech':
        return BlockFactorizedPriorAttn(d=33, N=500, M=33, B=args.batch_size)
    elif args.model_name == 'BlockScaleHyperpriorSawbridge':
        print(f'Block size={args.batch_size}')
        return BlockScaleHyperpriorSawbridge(100, 1, args.batch_size)
    elif args.model_name == 'BlockMeanScaleHyperpriorSawbridge':
        print(f'Block size={args.batch_size}')
        return BlockMeanScaleHyperpriorSawbridge(100, 1, args.batch_size)
    elif args.model_name == 'ScaleHyperprior':
        return ScaleHyperpriorSmall(32, 48)
    elif args.model_name == 'MeanScaleHyperpriorFull':
        return MeanScaleHyperpriorCustom(128, 192, args.orig_channels)
    elif args.model_name == 'Cheng2020AttentionFull':
        return Cheng2020AttentionCustom(192, args.orig_channels)
    elif args.model_name == 'mbt2018_mean':
        return mbt2018_mean(quality=1, metric="ms-ssim", pretrained=True)
    elif args.model_name == 'BlockMeanScaleHyperpriorFull':
        print(f'Block size={args.batch_size}')
        return BlockMeanScaleHyperpriorFull(48, 64, args.batch_size)
    elif args.model_name == 'SelfAttnFactorizedPrior':
        return SelfAttnFactorizedPriorSmall(32, 48)
    elif args.model_name == 'BlockSelfAttnFactorizedPrior':
        return BlockSelfAttnFactorizedPriorSmall(32, 48)
    elif args.model_name == 'SelfAttnScaleHyperprior':
        return SelfAttnScaleHyperpriorSmall(32, 48)
    elif args.model_name == 'SelfAttnScaleHyperpriorFull':
        return SelfAttnScaleHyperpriorFull(128, 192)
    elif args.model_name == 'ScaleHyperpriorFull':
        return ScaleHyperprior(128, 192)
    elif args.model_name == 'MBTFull':
        return JointAutoregressiveHierarchicalPriors(192, 192)
    elif args.model_name == 'SelfAttnMBT':
        return SelfAttnMBTSmall(32, 48)
    elif args.model_name == 'BlockSelfAttnScaleHyperprior':
        print(f'Block size={args.batch_size}')
        return BlockSelfAttnScaleHyperpriorSmall(32, 48, 10)
    elif args.model_name == 'BlockSelfAttnScaleHyperprior_EntAttn':
        print(f'Block size={args.batch_size}')
        return BlockSelfAttnScaleHyperpriorSmall_EntAttn(32, 48, args.batch_size)
    elif args.model_name == 'BlockSelfAttnScaleHyperprior_EntAttn2':
        print(f'Block size={args.batch_size}')
        return BlockSelfAttnScaleHyperpriorSmall_EntAttn2(32, 48, args.batch_size)
    elif args.model_name == 'BlockSelfAttnHyperprior_TwoStage':
        print(f'Block size={args.batch_size}')
        return BlockSelfAttnHyperpriorSmall_TwoStage(32, 48, args.batch_size)
    elif args.model_name == 'BlockSelfAttnScaleHyperprior_vqvae':
        print(f'Block size={args.batch_size}')
        return BlockSelfAttnScaleHyperpriorSmall_vqvae(32, 48, args.batch_size, args.embedding)
    elif args.model_name == 'BlockSelfAttnScaleHyperprior_vqvae2':
        print(f'Block size={args.batch_size}')
        return BlockSelfAttnScaleHyperpriorSmall_vqvae2(32, 48, args.batch_size, args.embedding)
    elif args.model_name == 'BlockSelfAttnScaleHyperpriorFull':
        print(f'Block size={args.batch_size}')
        return BlockSelfAttnScaleHyperpriorFull(128, 192, args.batch_size)
    elif args.model_name == 'BlockSelfAttnScaleHyperprior2':
        # no self attn across batch
        print(f'Block size={args.batch_size}')
        return BlockSelfAttnScaleHyperpriorSmall_2(32, 48, args.batch_size)
    elif args.model_name == 'BlockSelfAttnScaleHyperprior2_pretrained':
        # no self attn across batch
        print(f'Block size={args.batch_size}')
        return BlockSelfAttnScaleHyperpriorSmall_2_pretrained(32, 48, args.batch_size)
    elif args.model_name == 'BlockConvScaleHyperprior':
        print(f'Block size={args.batch_size}')
        return BlockConvScaleHyperpriorSmall(32, 48, args.batch_size)
    elif args.model_name == 'BlockConv3dScaleHyperprior':
        print(f'Block size={args.batch_size}')
        return BlockConv3dScaleHyperpriorSmall(32, 48, args.batch_size)
    elif args.model_name == 'BlockConvScaleHyperprior_indphi':
        print(f'Block size={args.batch_size}')
        return BlockConvScaleHyperpriorSmall_indphi(32, 48, args.batch_size)
    elif args.model_name == 'BlockSelfAttnMBT':
        print(f'Block size={args.batch_size}')
        return BlockSelfAttnMBTSmall(32, 48, args.batch_size)
    elif args.model_name == 'BlockSelfAttnMBT_EntAttn':
        print(f'Block size={args.batch_size}')
        return BlockSelfAttnMBTSmall_EntAttn(32, 48, args.batch_size)
    else:
        print("Invalid model_name")
        sys.exit(0)

class MeanScaleHyperpriorCustom(MeanScaleHyperprior):
    def __init__(self, N, M, orig_channels=3, **kwargs):
        super().__init__(N, M, **kwargs)
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

class FactorizedPriorSmall(FactorizedPrior):
    def __init__(self, N, M, orig_channels=3, **kwargs):
        super().__init__(N, M, **kwargs)
        self.g_a = nn.Sequential(
            conv(orig_channels, N),
            GDN(N),
            conv(N, N),
            # GDN(N),
            # conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            # deconv(N, N),
            # GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, orig_channels),
        )

class FactorizedPriorSawbridge(FactorizedPrior):
    def __init__(self, N=100, M=10, **kwargs):
        super().__init__(N, M, **kwargs)
        self.g_a = nn.Sequential(
            nn.Linear(1024, N),
            nn.LeakyReLU(),
            nn.Linear(N, N),
            nn.LeakyReLU(),
            nn.Linear(N, M)
        )

        self.g_s = nn.Sequential(
            nn.Linear(M, N),
            nn.LeakyReLU(),
            nn.Linear(N, N),
            nn.LeakyReLU(),
            nn.Linear(N, 1024)
        )

    def forward(self, x):
        # x: [B, 1024, 1, 1]
        x = x.squeeze() # [B, 1024]
        y = self.g_a(x) # [B, M]
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat) # [B, 1024]
        x_hat = x_hat.unsqueeze(2).unsqueeze(2) # [B, 1024]

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    def compress(self, x):
        x = x.squeeze()
        y = self.g_a(x)
        y = y.unsqueeze(2).unsqueeze(2)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        # shape = (1, 1)
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        y_hat = y_hat.squeeze()
        x_hat = self.g_s(y_hat)#.clamp_(-1, 1)
        x_hat = x_hat.unsqueeze(2).unsqueeze(2)
        return {"x_hat": x_hat}
    
class FactorizedPriorMLP(FactorizedPrior):
    def __init__(self, d=1024, N=100, M=10, **kwargs):
        super().__init__(N, M, **kwargs)
        self.g_a = nn.Sequential(
            nn.Linear(d, N),
            nn.Softplus(),
            nn.Linear(N, N),
            nn.Softplus(),
            nn.Linear(N, M)
        )

        self.g_s = nn.Sequential(
            nn.Linear(M, N),
            nn.Softplus(),
            nn.Linear(N, N),
            nn.Softplus(),
            nn.Linear(N, d)
        )

    def forward(self, x):
        # x: [B, d, 1, 1]
        x = x.squeeze() # [B, d]
        y = self.g_a(x) # [B, M]
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat) # [B, d]
        x_hat = x_hat.unsqueeze(2).unsqueeze(2) # [B, d, 1, 1]

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    def compress(self, x):
        x = x.squeeze()
        y = self.g_a(x)
        y = y.unsqueeze(2).unsqueeze(2)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        # shape = (1, 1)
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        y_hat = y_hat.squeeze()
        x_hat = self.g_s(y_hat)#.clamp_(-1, 1)
        x_hat = x_hat.unsqueeze(2).unsqueeze(2)
        return {"x_hat": x_hat}

class BlockFactorizedPriorMLP(FactorizedPrior):
    def __init__(self, d, N, M, B, **kwargs):
        super().__init__(N, M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(M*B)

        self.g_a = nn.Sequential(
            nn.Linear(d*B, N*B),
            nn.Softplus(),
            nn.Linear(N*B, N*B),
            nn.Softplus(),
            nn.Linear(N*B, M*B)
        )

        self.g_s = nn.Sequential(
            nn.Linear(M*B, N*B),
            nn.Softplus(),
            nn.Linear(N*B, N*B),
            nn.Softplus(),
            nn.Linear(N*B, d*B)
        )
        self.d = d
        self.M = M
        self.N = N
        self.B = B

    def forward(self, x):
        # print(x.shape)
        x = x.squeeze().reshape(-1, self.B*self.d) # [bsize, B*d]
        y = self.g_a(x) # [bsize, B*M]
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat) # [bsize, d*B]
        x_hat = x_hat.reshape(-1, self.d)
        x_hat = x_hat.unsqueeze(2).unsqueeze(2)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    def compress(self, x):
        x = x.squeeze()
        y = self.g_a(x)
        y = y.unsqueeze(2).unsqueeze(2)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        # shape = (1, 1)
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        y_hat = y_hat.squeeze()
        x_hat = self.g_s(y_hat)#.clamp_(-1, 1)
        x_hat = x_hat.unsqueeze(2).unsqueeze(2)
        return {"x_hat": x_hat}
    
# class BlockFactorizedPriorAttn(FactorizedPrior):
#     def __init__(self, d, N, M, B, **kwargs):
#         super().__init__(N, M, **kwargs)
#         self.entropy_bottleneck = EntropyBottleneck(B*M)
#         self.g_a = PermEquivEncoder(d, N, M)
#         self.g_s = PermEquivDecoder(d, N, M)
#         self.d = d
#         self.M = M
#         self.N = N
#         self.B = B

#     def forward(self, x):
#         # x: [B*bsize, d]
#         # print(x.shape)
#         x = x.squeeze().reshape(-1, self.B, self.d) # [bsize, B, d]
#         y = self.g_a(x) # [bsize, B, M]
#         y = y.reshape(-1, self.B*self.M)
#         # y = y.permute(0,2,1) # [bsize, d, B]
#         y_hat, y_likelihoods = self.entropy_bottleneck(y)
#         # y_hat = y_hat.permute(0,2,1) # [bsize, B, d]
#         y_hat = y_hat.reshape(-1, self.B, self.M)
#         x_hat = self.g_s(y_hat) # [bsize, B, d]
#         x_hat = x_hat.reshape(-1, self.d)
#         x_hat = x_hat.unsqueeze(2).unsqueeze(2)

#         return {
#             "x_hat": x_hat,
#             "likelihoods": {
#                 "y": y_likelihoods,
#             },
#         }

#     def compress(self, x):
#         x = x.squeeze()
#         y = self.g_a(x)
#         y = y.unsqueeze(2).unsqueeze(2)
#         y_strings = self.entropy_bottleneck.compress(y)
#         return {"strings": [y_strings], "shape": y.size()[-2:]}

#     def decompress(self, strings, shape):
#         # shape = (1, 1)
#         assert isinstance(strings, list) and len(strings) == 1
#         y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
#         y_hat = y_hat.squeeze()
#         x_hat = self.g_s(y_hat)#.clamp_(-1, 1)
#         x_hat = x_hat.unsqueeze(2).unsqueeze(2)
#         return {"x_hat": x_hat}


class BlockFactorizedPriorSawbridge(FactorizedPrior):
    def __init__(self, N, M, B, **kwargs):
        super().__init__(N, M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(M*B)
        self.g_a = nn.Sequential(
            nn.Linear(1024, N),
            nn.LeakyReLU(),
            nn.Linear(N, N),
            nn.LeakyReLU(),
            nn.Linear(N, M)
        )

        self.g_s = nn.Sequential(
            nn.Linear(M, N),
            nn.LeakyReLU(),
            nn.Linear(N, N),
            nn.LeakyReLU(),
            nn.Linear(N, 1024)
        )
        self.g_a_inner = MobiusLayer(M*B)
        self.g_s_inner = MobiusLayer(M*B)
        # self.g_a_inner = nn.Sequential(
        #     nn.Linear(M*B, M*B),
        #     nn.LeakyReLU(),
        #     nn.Linear(M*B, M*B),
        #     nn.LeakyReLU(),
        #     nn.Linear(M*B, M*B)
        # )
        # self.g_s_inner = nn.Sequential(
        #     nn.Linear(M*B, M*B),
        #     nn.LeakyReLU(),
        #     nn.Linear(M*B, M*B),
        #     nn.LeakyReLU(),
        #     nn.Linear(M*B, M*B)
        # )
        self.M = M
        self.N = N
        self.B = B

    def forward(self, x):
        # print(x.shape)
        x = x.squeeze() # [B*bsize, 1024]
        y = self.g_a(x) # [B*bsize, 1]
        y = y.reshape(-1, self.B*self.M) #[bsize, B]
        y = self.g_a_inner(y)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        # y_hat = y_hat.squeeze()
        y_hat = self.g_s_inner(y_hat) # [bsize, B]
        y_hat = y_hat.reshape(-1, 1) #[bsize*B, 1]
        x_hat = self.g_s(y_hat).clamp(-1, 1)
        x_hat = x_hat.unsqueeze(2).unsqueeze(2)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    # def forward(self, x):
    #     # print(x.shape)
    #     x = x.squeeze()
    #     y = self.g_a(x) # [B, M]
    #     y = y.reshape(1, self.B*self.M) #[1, B*M]
    #     # print(y.shape)
    #     # y = y.unsqueeze(2).unsqueeze(2)
    #     y_hat, y_likelihoods = self.entropy_bottleneck(y)
    #     # y_hat = y_hat.squeeze()
    #     y_hat = y_hat.reshape(self.B, self.M)
    #     x_hat = self.g_s(y_hat)
    #     x_hat = x_hat.unsqueeze(2).unsqueeze(2)

    #     return {
    #         "x_hat": x_hat,
    #         "likelihoods": {
    #             "y": y_likelihoods,
    #         },
    #     }

    def compress(self, x):
        x = x.squeeze()
        y = self.g_a(x)
        y = y.unsqueeze(2).unsqueeze(2)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        # shape = (1, 1)
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        y_hat = y_hat.squeeze()
        x_hat = self.g_s(y_hat)#.clamp_(-1, 1)
        x_hat = x_hat.unsqueeze(2).unsqueeze(2)
        return {"x_hat": x_hat}

class BlockScaleHyperpriorSawbridge(ScaleHyperprior):
    r"""
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(N, M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(N*B)

        self.g_a = nn.Sequential(
            nn.Linear(1024, N),
            nn.LeakyReLU(),
            nn.Linear(N, N),
            nn.LeakyReLU(),
            nn.Linear(N, M)
        )

        self.g_s = nn.Sequential(
            nn.Linear(M, N),
            nn.LeakyReLU(),
            nn.Linear(N, N),
            nn.LeakyReLU(),
            nn.Linear(N, 1024)
        )

        self.h_a = nn.Sequential(
            nn.Linear(B*M, B*N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(B*N, B*N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(B*N, B*N),
        )

        self.h_s = nn.Sequential(
            nn.Linear(B*N, B*N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(B*N, B*N),
            nn.ReLU(inplace=True),
            nn.Linear(B*N, B*M),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength


    def forward(self, x):
        x = x.squeeze()
        y = self.g_a(x) # y:[B, M]
        y = y.unsqueeze(2).unsqueeze(2)
        y_height, y_width = y.shape[2], y.shape[3]
        # print(x.shape, y.shape)
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0) # y:[1, B*M, 1, 1]
        y = y.squeeze(dim=2).squeeze(dim=2)
        z = self.h_a(y) # z:[1, B*N]
        # y = y.unsqueeze(2).unsqueeze(2) # y: [1, B*N, 1, 1]
        # print(z.shape)
        # z = z.unsqueeze(2).unsqueeze(2)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        # z_hat = z_hat.squeeze(2).squeeze(2)
        scales_hat = self.h_s(z_hat) 
        # print(z_hat.shape, gaussian_params.shape)
        # gaussian_params.unsqueeze(2).unsqueeze(2)
        # scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        # print(y_likelihoods.shape, z_likelihoods.shape)
        y_hat = y_hat.reshape(self.B, self.M, y_height, y_width) 
        y_hat = y_hat.squeeze(2).squeeze(2)
        x_hat = self.g_s(y_hat)
        x_hat = x_hat.unsqueeze(2).unsqueeze(2)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

class BlockMeanScaleHyperpriorSawbridge(ScaleHyperprior):
    r"""
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(N, M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(N*B)

        self.g_a = nn.Sequential(
            nn.Linear(1024, N),
            nn.LeakyReLU(),
            nn.Linear(N, N),
            nn.LeakyReLU(),
            nn.Linear(N, M)
        )

        self.g_s = nn.Sequential(
            nn.Linear(M, N),
            nn.LeakyReLU(),
            nn.Linear(N, N),
            nn.LeakyReLU(),
            nn.Linear(N, 1024)
        )

        self.h_a = nn.Sequential(
            nn.Linear(B*M, B*N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(B*N, B*N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(B*N, B*N),
        )

        self.h_s = nn.Sequential(
            nn.Linear(B*N, B*M),
            nn.LeakyReLU(inplace=True),
            nn.Linear(B*M, B*M*3 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(B*M*3 // 2, B*M * 2),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength


    def forward(self, x):
        x = x.squeeze()
        y = self.g_a(x) # y:[B, M]
        y = y.unsqueeze(2).unsqueeze(2)
        y_height, y_width = y.shape[2], y.shape[3]
        # print(x.shape, y.shape)
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0) # y:[1, B*M, 1, 1]
        y = y.squeeze(dim=2).squeeze(dim=2)
        z = self.h_a(y) # z:[1, B*N]
        # y = y.unsqueeze(2).unsqueeze(2) # y: [1, B*N, 1, 1]
        # print(z.shape)
        z = z.unsqueeze(2).unsqueeze(2)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat.squeeze(2).squeeze(2)
        gaussian_params = self.h_s(z_hat) 
        # print(z_hat.shape, gaussian_params.shape)
        gaussian_params.unsqueeze(2).unsqueeze(2)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        # print(y_likelihoods.shape, z_likelihoods.shape)
        y_hat = y_hat.reshape(self.B, self.M, y_height, y_width) 
        y_hat = y_hat.squeeze(2).squeeze(2)
        x_hat = self.g_s(y_hat)
        x_hat = x_hat.unsqueeze(2).unsqueeze(2)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, x):
        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        y_hat = y_hat.reshape(self.B, self.M, 8, 8)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class ScaleHyperpriorSmall(ScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)
        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            # GDN(N),
            # conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            # deconv(N, N),
            # GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            # conv(N, N),
            # nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            # nn.ReLU(inplace=True),
            # deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )


class SelfAttnFactorizedPriorSmall(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            SelfAttn(N),
            deconv(N, 3),
        )

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2**3

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }


    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net


    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class BlockSelfAttnFactorizedPriorSmall(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, 3),
        )

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2**3

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }


    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net


    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(np.log(min), np.log(max), levels))

class SelfAttnScaleHyperpriorSmall(ScaleHyperprior):
    r"""
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            # conv(N, N),
            # nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            # nn.ReLU(inplace=True),
            # deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def stack(self, x):
        B, M, H, W = x.shape
        x = x.permute(1,0,2,3).reshape(M, B*H,W).unsqueeze(0) # [1, M, B*H, W]
        return x
    
    def stack_back(self, x):
        _, M, BH, W = x.shape
        B = BH // W
        H = W
        x = x.squeeze().reshape(M, B, H, W).permute(1,0,2,3) # [B,M,H,W]
        return x


    def compress(self, x):
        y = self.g_a(x) # [B, M, 4, 4]
        # B, M, y_height, y_width = y.shape
        # y = y.permute(1,0,2,3).reshape(M, B*y_height, y_width).unsqueeze(0) # [1, M, B*4, 4]
        # y_height, y_width = y.shape[2], y.shape[3]
        # y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0)
        z = self.h_a(torch.abs(y)) # [B, M, 2, 2]
        z = self.stack(z)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.stack_back(self.entropy_bottleneck.decompress(z_strings, z.size()[-2:]))

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        # print(self.stack(y).shape, self.stack(indexes).shape, z_hat.shape)
        y_strings = self.gaussian_conditional.compress(self.stack(y), self.stack(indexes))
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.stack_back(self.entropy_bottleneck.decompress(strings[1], shape))
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.stack_back(self.gaussian_conditional.decompress(strings[0], self.stack(indexes), z_hat.dtype))
        # y_hat = y_hat.reshape(self.B, self.M, 4, 4)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class SelfAttnScaleHyperpriorFull(ScaleHyperprior):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )




class MeanScaleHyperpriorFull(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class BlockMeanScaleHyperpriorFull(ScaleHyperprior):
    r"""
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(N, M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(N*B)

        self.g_a = nn.Sequential(
            conv(3, N),
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
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(B*M, B*N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(B*N, B*N),
            nn.LeakyReLU(inplace=True),
            conv(B*N, B*N),
        )

        self.h_s = nn.Sequential(
            deconv(B*N, B*M),
            nn.LeakyReLU(inplace=True),
            deconv(B*M, B*M*3 // 2),
            nn.ReLU(inplace=True),
            conv(B*M*3 // 2, B*M * 2, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength


    def forward(self, x):
        y = self.g_a(x) # y:[B, M, 4, 4]
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0) # y:[1, B*M, 4, 4]
        z = self.h_a(y) # z:[1, B*N, 2, 2]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat) 
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        # print(y_likelihoods.shape, z_likelihoods.shape)
        y_hat = y_hat.reshape(self.B, self.M, y_height, y_width) 
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, x):
        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        y_hat = y_hat.reshape(self.B, self.M, 8, 8)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class SelfAttnMBTSmall(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            # conv(N, N, stride=2, kernel_size=5),
            # nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M*3 // 2, stride=2, kernel_size=5),
            # nn.LeakyReLU(inplace=True),
            # deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=3, padding=1, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)


    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        # print(x.shape, y.shape)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 2  # scaling factor between z and y
        kernel_size = 3  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                # print(y_crop.shape, y_hat.shape, masked_weight.shape)
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 2  # scaling factor between z and y
        kernel_size = 3  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv

class BlockSelfAttnScaleHyperpriorSmall_vqvae(ScaleHyperprior):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, embedding, **kwargs):
        super().__init__(N, M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(N*B) # override super() init

        self.embedding = embedding

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(B*M, B*N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            # conv(N, N),
            # nn.ReLU(inplace=True),
            conv(B*N, B*N),
        )

        self.h_s = nn.Sequential(
            deconv(B*N, B*N),
            # nn.ReLU(inplace=True),
            # deconv(N, N),
            nn.ReLU(inplace=True),
            conv(B*N, B*M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength

    def quantize(self, z, embedding):
        z_i = z[:,None,:]
        e_j = embedding.weight[None,:,:]
                # z_i = z[:,None,:]
        # e_j = embedding.weight[None,:,:]
        distances = ((z_i - e_j)**2).sum(dim=2)
        encoding_indices = distances.argmin(dim=1)
        quantized = embedding.weight[encoding_indices.squeeze()]
        return quantized, encoding_indices.squeeze()


    def forward(self, x):
        y = self.g_a(x) # y:[B, M, 4, 4]
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0) # y:[1, B*M, 4, 4]
        y_flat = y.reshape(1, -1)
        quantized, _ = self.quantize(y_flat, self.embedding)
        quantized = quantized.reshape(1, self.B*self.M, y_height, y_width)
        residual = y - quantized
        z = self.h_a(torch.abs(residual)) # z:[1, B*N, 2, 2]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat) # [1, B*M, 4, 4]
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        y_hat = y_hat + quantized
        y_hat = y_hat.reshape(self.B, self.M, y_height, y_width) # y_hat: [B, M, 4, 4]
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0)
        y_flat = y.reshape(1, -1)
        quantized, idx_codebook = self.quantize(y_flat, self.embedding)
        quantized = quantized.reshape(1, self.B*self.M, y_height, y_width)
        residual = y - quantized
        z = self.h_a(torch.abs(residual))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(residual, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:], "idx_codebook":idx_codebook}

    def decompress(self, strings, shape, idx_codebook):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        y_hat = y_hat + self.embedding.weight[idx_codebook].reshape(1, self.B*self.M, 4, 4)
        y_hat = y_hat.reshape(self.B, self.M, 4, 4)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class BlockSelfAttnScaleHyperpriorSmall_vqvae2(ScaleHyperprior):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, embedding, **kwargs):
        super().__init__(N, M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(N*B) # override super() init
        self.embedding = embedding

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(B*M, B*N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            # conv(N, N),
            # nn.ReLU(inplace=True),
            conv(B*N, B*N),
        )

        self.h_s = nn.Sequential(
            deconv(B*N, B*N),
            # nn.ReLU(inplace=True),
            # deconv(N, N),
            nn.ReLU(inplace=True),
            conv(B*N, B*M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength

    def quantize(self, z, embedding):
        z_i = z[:,None,:]
        e_j = embedding.weight[None,:,:]
                # z_i = z[:,None,:]
        # e_j = embedding.weight[None,:,:]
        distances = ((z_i - e_j)**2).sum(dim=2)
        encoding_indices = distances.argmin(dim=1)
        quantized = embedding.weight[encoding_indices.squeeze()]
        return quantized


    def forward(self, x):
        y = self.g_a(x) # y:[B, M, 4, 4]
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0) # y:[1, B*M, 4, 4]
        # y_flat = y.reshape(1, -1)
        # quantized = self.quantize(y_flat, embedding).reshape(1, self.B*self.M, y_height, y_width)
        # residual = y - quantized
        z = self.h_a(torch.abs(y)) # z:[1, B*N, 2, 2]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat) # [1, B*M, 4, 4]
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        residual = (y - y_hat).reshape(1, -1)
        quantized, _ = self.quantize(residual, self.embedding)
        quantized = quantized.reshape(1, self.B*self.M, y_height, y_width)
        # quantized = (residual + (quantized - residual).detach()).reshape(1, self.B*self.M, y_height, y_width)
        y_hat = y_hat + quantized
        # print(y_likelihoods.shape, z_likelihoods.shape)
        y_hat = y_hat.reshape(self.B, self.M, y_height, y_width) # y_hat: [B, M, 4, 4]
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0)
        # y_flat = y.reshape(1, -1)
        # quantized, idx_codebook = self.quantize(y_flat, self.embedding).reshape(1, self.B*self.M, y_height, y_width)
        # residual = y - quantized
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)

        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, z_hat.dtype)
        residual = (y - y_hat).reshape(1, -1)
        _, idx_codebook = self.quantize(residual, self.embedding)
        # quantized = quantized.reshape(1, self.B*self.M, y_height, y_width)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:], "idx_codebook":idx_codebook}

    def decompress(self, strings, shape, idx_codebook):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        y_hat = y_hat + self.embedding.weight[idx_codebook]
        y_hat = y_hat.reshape(self.B, self.M, 4, 4)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}



class BlockSelfAttnScaleHyperpriorSmall(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(N, M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(N*B)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(B*M, B*N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            # conv(N, N),
            # nn.ReLU(inplace=True),
            conv(B*N, B*N),
        )

        self.h_s = nn.Sequential(
            deconv(B*N, B*N),
            # nn.ReLU(inplace=True),
            # deconv(N, N),
            nn.ReLU(inplace=True),
            conv(B*N, B*M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (3 + 1)

    # def view_attn_maps(x):


    def forward(self, x):
        y = self.g_a(x) # y:[B, M, 4, 4]
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0) # y:[1, B*M, 4, 4]
        z = self.h_a(torch.abs(y)) # z:[1, B*N, 2, 2]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat) # [1, B*M, 4, 4]
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        # print(y_likelihoods.shape, z_likelihoods.shape)
        y_hat = y_hat.reshape(self.B, self.M, y_height, y_width) # y_hat: [B, M, 4, 4]
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # @classmethod    
    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)


    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def stack(self, x):
        B, M, H, W = x.shape
        x = x.permute(1,0,2,3).reshape(M, B*H,W).unsqueeze(0) # [1, M, B*H, W]
        return x
    
    def stack_back(self, x):
        _, M, BH, W = x.shape
        B = BH // W
        H = W
        x = x.squeeze().reshape(M, B, H, W).permute(1,0,2,3) # [B,M,H,W]
        return x

    def compress(self, x):
        # [100, 3, 32, 32]
        y = []
        z = []
        for i in range(10):
            y_i = self.g_a(x[10*i:10*i+10])
            y_height, y_width = y_i.shape[2], y_i.shape[3]
            y_i = y_i.reshape(self.B*self.M, y_height, y_width).unsqueeze(0)
            y.append(y_i) #[1, B*M, 4, 4]
            z_i = self.h_a(torch.abs(y_i))
            # print(z_i.shape)
            z.append(z_i) # [1, B*M,2, 2]
        y = torch.cat(y) #[10, B*M, 4, 4]
        z = torch.cat(z) #[10, B*M, 2, 2]
        # print(z)
        z = self.stack(z)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.stack_back(self.entropy_bottleneck.decompress(z_strings, z.size()[-2:]))

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(self.stack(y), self.stack(indexes))
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.stack_back(self.entropy_bottleneck.decompress(strings[1], shape))
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.stack_back(self.gaussian_conditional.decompress(strings[0], self.stack(indexes), z_hat.dtype))
        # y_hat : [10, B*M, 4, 4]
        x_hat = []
        for i in range(10):
            y_hat_i = y_hat[i].reshape(self.B, self.M, 4, 4)
            x_hat.append(self.g_s(y_hat_i).clamp_(0, 1)) #[B, M, 4, 4]
        x_hat = torch.cat(x_hat, dim=0)
        return {"x_hat": x_hat}

    # def compress(self, x):
    #     y = self.g_a(x)
    #     y_height, y_width = y.shape[2], y.shape[3]
    #     y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0)
    #     z = self.h_a(torch.abs(y))

    #     z_strings = self.entropy_bottleneck.compress(z)
    #     z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

    #     scales_hat = self.h_s(z_hat)
    #     indexes = self.gaussian_conditional.build_indexes(scales_hat)
    #     y_strings = self.gaussian_conditional.compress(y, indexes)
    #     return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    # def decompress(self, strings, shape):
    #     assert isinstance(strings, list) and len(strings) == 2
    #     z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
    #     scales_hat = self.h_s(z_hat)
    #     indexes = self.gaussian_conditional.build_indexes(scales_hat)
    #     y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
    #     y_hat = y_hat.reshape(self.B, self.M, 4, 4)
    #     x_hat = self.g_s(y_hat).clamp_(0, 1)
    #     return {"x_hat": x_hat}

class BlockSelfAttnHyperpriorSmall_TwoStage(ScaleHyperprior):
    r"""
    Applies NTC to the extracted latent space on blocks. 
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(N, M,  **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(N*B) # override super() init

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, 3),
        )

        self.g_a_inner = nn.Sequential(
            conv(B*M, B*M, stride=1, kernel_size=3),
            GDN(B*M),
            conv(B*M, B*M, stride=1, kernel_size=3)
        )
        self.g_s_inner = nn.Sequential(
            deconv(B*M, B*M, stride=1, kernel_size=3),
            GDN(B*M, inverse=True),
            deconv(B*M, B*M, stride=1, kernel_size=3)
        )
        self.h_a = nn.Sequential(
            conv(B*M, B*N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(B*N, B*N),
        )

        self.h_s = nn.Sequential(
            deconv(B*N, B*N),
            nn.ReLU(inplace=True),
            conv(B*N, B*M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength

    def forward(self, x):
        y = self.g_a(x) # y:[B, M, 4, 4]
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0) # y:[1, B*M, 4, 4]
        y = self.g_a_inner(y)
        z = self.h_a(torch.abs(y)) # z:[1, B*N, 2, 2]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat) # [1, B*M, 4, 4]
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        y_hat = self.g_s_inner(y_hat)
        # print(y_likelihoods.shape, z_likelihoods.shape)
        y_hat = y_hat.reshape(self.B, self.M, y_height, y_width) # y_hat: [B, M, 4, 4]
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0)
        y = self.g_a_inner(y)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        y_hat = self.g_s_inner(y_hat)
        y_hat = y_hat.reshape(self.B, self.M, 4, 4)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class BlockSelfAttnScaleHyperpriorSmall_EntAttn(ScaleHyperprior):
    r"""
    Same as BSASH (small) but uses attention in the entropy model instead of conv structure. 
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(N, M,  **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(N) # override super() init

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, 3),
        )
        # self.h_a = nn.Sequential(
        #     conv(M, N, stride=1, kernel_size=3),
        #     nn.ReLU(inplace=True), # [B, N, 4, 4]
        #     SelfAttn_AcrossBatch(N, downsample=1),
        #     SelfAttn(N, downsample=1),
        #     SelfAttn_AcrossBatch(N, downsample=1),
        #     SelfAttn(N, downsample=1),
        #     SelfAttn_AcrossBatch(N, downsample=1),
        #     SelfAttn(N, downsample=1),
        #     conv(N, N),
        # )

        # self.h_s = nn.Sequential(
        #     deconv(N, N),
        #     nn.ReLU(inplace=True),
        #     SelfAttn_AcrossBatch(N, downsample=1),
        #     SelfAttn(N, downsample=1),
        #     SelfAttn_AcrossBatch(N, downsample=1),
        #     SelfAttn(N, downsample=1),
        #     SelfAttn_AcrossBatch(N, downsample=1),
        #     SelfAttn(N, downsample=1),
        #     conv(N, M, stride=1, kernel_size=3),
        #     nn.ReLU(inplace=True),
        # )
        self.h_a = nn.Sequential( 
            # SelfAttn(M, downsample=1),
            SelfAttn(M, downsample=1),
            # conv(M, M, stride=1, kernel_size=1),
            nn.AvgPool2d(2),
            # conv(M, M, stride=2, kernel_size=3),
            # SelfAttn(M, downsample=1),
            SelfAttn(M, downsample=1),
            conv(M, N, stride=1, kernel_size=1)
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=1, kernel_size=1),
            # SelfAttn(M, downsample=1),
            SelfAttn(M, downsample=1),
            # deconv(M, M, stride=2, kernel_size=3),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # deconv(M, M, stride=1, kernel_size=1),
            # SelfAttn(M, downsample=1),
            SelfAttn(M, downsample=1),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength

    def view_attn_maps(self, x):
        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.permute(0, 3, 1, 2).reshape(1, y_width*self.B, self.M, y_height).permute(0, 2, 3, 1)
        # _, attn_map = self.h_a[0](torch.abs(y))
        attn_map = view_attn_map(self.h_a[0], torch.abs(y))
        return attn_map

    def stack_along_H(self, x):
        # x : [B, C, H, W]
        # returns [1, C, H*B, W]
        C, H, W = x.shape[1], x.shape[2], x.shape[3]
        x = x.permute(0, 2, 1, 3).reshape(1, H*self.B, C, W).permute(0, 2, 1, 3) #[1, C, H*B, W]
        return x

    def unstack_along_H(self, x):
        # x : [1, C, H*B, W]
        # returns: [B, C, H, W]
        C, HB, W = x.shape[1], x.shape[2], x.shape[3]
        H = HB // self.B
        x = x.permute(0, 2, 1, 3).reshape(self.B, H, C, W).permute(0, 2, 1, 3) #[B, C, H, W]
        return x

    def forward(self, x):
        y = self.g_a(x) # y:[B, M, 4, 4]
        y_height, y_width = y.shape[2], y.shape[3]
        y = self.stack_along_H(y) #[1, M, 4*B, 4]
        # y = y.permute(0, 2, 1, 3).reshape(1, y_height*self.B, self.M, y_width).permute(0, 2, 1, 3) #[1, M, 4*B, 4]
        # y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0) # y:[1, B*M, 4, 4]
        z = self.h_a(torch.abs(y)) # z:[1, N, 2, 2*B]
        # z = self.unstack_along_H(z) # z:[B, N, 2, 2]
        # z = z.reshape(1, self.B*self.N, z.shape[2], z.shape[3]) # z: [1, B*N, 2, 2]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        # z_hat = z_hat.reshape(self.B, self.N, z_hat.shape[2], z_hat.shape[3])# [B, N, 2, 2]
        # z_hat = self.stack_along_H(z_hat) # [1, N, 2*B, 2]
        scales_hat = self.h_s(z_hat) # [1, M, 4*B, 4]
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        # print(y_likelihoods.shape, z_likelihoods.shape)
        # y_hat = y_hat.reshape(self.B, self.M, y_height, y_width) # y_hat: [B, M, 4, 4]
        # y_hat = y_hat.permute(0, 2, 1, 3).reshape(self.B, y_height, self.M, y_width).permute(0, 2, 1, 3) # [B, M, 4, 4]
        y_hat = self.unstack_along_H(y_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        # y = y.permute(0, 2, 1, 3).reshape(1, y_height*self.B, self.M, y_width).permute(0, 2, 1, 3) #[1, M, 4*B, 4]
        y = self.stack_along_H(y)
        z = self.h_a(torch.abs(y))
        # z = self.unstack_along_H(z)
        # z = z.reshape(1, self.B*self.N, z.shape[2], z.shape[3])
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        # z_hat = z_hat.reshape(self.B, self.N, z_hat.shape[2], z_hat.shape[3])
        # z_hat = self.stack_along_H(z_hat)

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        # z_hat = z_hat.reshape(self.B, self.N, z_hat.shape[2], z_hat.shape[3])
        # z_hat = self.stack_along_H(z_hat)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        # y_hat = y_hat.permute(0, 3, 1, 2).reshape(self.B, 4, self.M, 4).permute(0, 2, 3, 1)
        y_hat = self.unstack_along_H(y_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
    
class BlockSelfAttnScaleHyperpriorSmall_EntAttn2(ScaleHyperprior):
    r"""
    Same as BSASH (small) but uses attention in the entropy model instead of conv structure. 
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(N, M,  **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(M) # override super() init

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, 3),
        )
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True), # [B, N, 4, 4]
            SelfAttn_AcrossBatch(N, downsample=1),
            SelfAttn(N, downsample=1),
            SelfAttn_AcrossBatch(N, downsample=1),
            SelfAttn(N, downsample=1),
            # SelfAttn_AcrossBatch(N, downsample=1),
            # SelfAttn(N, downsample=1),
            conv(N, M),
        )

        self.h_s = nn.Sequential(
            deconv(M, N),
            nn.ReLU(inplace=True),
            # SelfAttn_AcrossBatch(N, downsample=1),
            # SelfAttn(N, downsample=1),
            SelfAttn_AcrossBatch(N, downsample=1),
            SelfAttn(N, downsample=1),
            SelfAttn_AcrossBatch(N, downsample=1),
            SelfAttn(N, downsample=1),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength

    def view_attn_maps(self, x):
        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.permute(0, 3, 1, 2).reshape(1, y_width*self.B, self.M, y_height).permute(0, 2, 3, 1)
        # _, attn_map = self.h_a[0](torch.abs(y))
        attn_map = view_attn_map(self.h_a[0], torch.abs(y))
        return attn_map

    def stack_along_H(self, x):
        # x : [B, C, H, W]
        # returns [1, C, H*B, W]
        C, H, W = x.shape[1], x.shape[2], x.shape[3]
        x = x.permute(0, 2, 1, 3).reshape(1, H*self.B, C, W).permute(0, 2, 1, 3) #[1, C, H*B, W]
        return x

    def unstack_along_H(self, x):
        # x : [1, C, H*B, W]
        # returns: [B, C, H, W]
        C, HB, W = x.shape[1], x.shape[2], x.shape[3]
        H = HB // self.B
        x = x.permute(0, 2, 1, 3).reshape(self.B, H, C, W).permute(0, 2, 1, 3) #[B, C, H, W]
        return x

    def forward(self, x):
        y = self.g_a(x) # y:[B, M, 4, 4]
        y_height, y_width = y.shape[2], y.shape[3]
        z = self.h_a(torch.abs(y)) # z:[B, M, 2, 2]
        z = self.stack_along_H(z) #[1, M, B*2, 2]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        # z_hat = z_hat.reshape(self.B, self.N, z_hat.shape[2], z_hat.shape[3])# [B, N, 2, 2]
        # z_hat = self.stack_along_H(z_hat) # [1, N, 2*B, 2]
        z_hat = self.unstack_along_H(z_hat) #[B, M, 2, 2]
        scales_hat = self.h_s(z_hat) #[B, M, 4, 4]
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        # print(y_likelihoods.shape, z_likelihoods.shape)
        # y_hat = y_hat.reshape(self.B, self.M, y_height, y_width) # y_hat: [B, M, 4, 4]
        # y_hat = y_hat.permute(0, 2, 1, 3).reshape(self.B, y_height, self.M, y_width).permute(0, 2, 1, 3) # [B, M, 4, 4]
        # y_hat = self.unstack_along_H(y_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        # y = y.permute(0, 2, 1, 3).reshape(1, y_height*self.B, self.M, y_width).permute(0, 2, 1, 3) #[1, M, 4*B, 4]
        y = self.stack_along_H(y)
        z = self.h_a(torch.abs(y))
        # z = self.unstack_along_H(z)
        # z = z.reshape(1, self.B*self.N, z.shape[2], z.shape[3])
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        # z_hat = z_hat.reshape(self.B, self.N, z_hat.shape[2], z_hat.shape[3])
        # z_hat = self.stack_along_H(z_hat)

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        # z_hat = z_hat.reshape(self.B, self.N, z_hat.shape[2], z_hat.shape[3])
        # z_hat = self.stack_along_H(z_hat)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        # y_hat = y_hat.permute(0, 3, 1, 2).reshape(self.B, 4, self.M, 4).permute(0, 2, 3, 1)
        y_hat = self.unstack_along_H(y_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class BlockSelfAttnScaleHyperpriorFull(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(entropy_bottleneck_channels=N*B, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            # SelfAttn(N),
            conv(N, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            # SelfAttn(N),
            conv(N, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            # SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            # SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            # SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            # SelfAttn(N),
            deconv(N, 3),
        )
        self.h_a = nn.Sequential(
            conv(B*M, B*N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(B*N, B*N),
            nn.ReLU(inplace=True),
            conv(B*N, B*N),
        )

        self.h_s = nn.Sequential(
            deconv(B*N, B*N),
            nn.ReLU(inplace=True),
            deconv(B*N, B*N),
            nn.ReLU(inplace=True),
            conv(B*N, B*M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)
    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x):
        y = self.g_a(x) # y:[B, M, 4, 4]
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0) # y:[1, B*M, 4, 4]
        z = self.h_a(torch.abs(y)) # z:[1, B*N, 2, 2]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat) # [1, B*M, 4, 4]
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        # print(y_likelihoods.shape, z_likelihoods.shape)
        y_hat = y_hat.reshape(self.B, self.M, y_height, y_width) # y_hat: [B, M, 4, 4]
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        y_hat = y_hat.reshape(self.B, self.M, 16, 16)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class BlockSelfAttnScaleHyperpriorSmall_2(BlockSelfAttnScaleHyperpriorSmall):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(N, M, B, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(B*M, B*N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            # conv(N, N),
            # nn.ReLU(inplace=True),
            conv(B*N, B*N),
        )

        self.h_s = nn.Sequential(
            deconv(B*N, B*N),
            # nn.ReLU(inplace=True),
            # deconv(N, N),
            nn.ReLU(inplace=True),
            conv(B*N, B*M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength

class BlockSelfAttnScaleHyperpriorSmall_2_pretrained(BlockSelfAttnScaleHyperpriorSmall_2):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(N, M, B, **kwargs)

    def forward(self, x):
        #Only train entropy model
        with torch.no_grad():
            y = self.g_a(x) # y:[B, M, 4, 4]
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0) # y:[1, B*M, 4, 4]
        z = self.h_a(torch.abs(y)) # z:[1, B*N, 2, 2]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat) # [1, B*M, 4, 4]
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        # print(y_likelihoods.shape, z_likelihoods.shape)
        y_hat = y_hat.reshape(self.B, self.M, y_height, y_width) # y_hat: [B, M, 4, 4]
        with torch.no_grad():
            x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

class BlockConvScaleHyperpriorSmall_indphi(BlockSelfAttnScaleHyperpriorSmall):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(N, M, B, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            # SelfAttn(N),
            conv(N, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            # SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            # SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            # SelfAttn(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(B*M, B*N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            # conv(N, N),
            # nn.ReLU(inplace=True),
            conv(B*N, B*N),
        )

        self.h_s = nn.Sequential(
            deconv(B*N, B*N),
            # nn.ReLU(inplace=True),
            # deconv(N, N),
            nn.ReLU(inplace=True),
            conv(B*N, B*M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength

class BlockSelfAttnMBTSmall(MeanScaleHyperprior):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(N*B) # override super() init

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            # SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M*B, N*B, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            # conv(N, N, stride=2, kernel_size=5),
            # nn.LeakyReLU(inplace=True),
            conv(N*B, N*B, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N*B, M*B*3 // 2, stride=2, kernel_size=5),
            # nn.LeakyReLU(inplace=True),
            # deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M*B * 3 // 2, M*B * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M*B * 12 // 3, M*B * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M*B * 10 // 3, M*B * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M*B * 8 // 3, M*B * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M*B, 2 * M*B, kernel_size=3, padding=1, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = y_hat.reshape(self.B, self.M, y_height, y_width)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net


    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 2  # scaling factor between z and y
        kernel_size = 3  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 2  # scaling factor between z and y
        kernel_size = 3  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M*self.B, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        # print(y_hat.shape)
        y_hat = y_hat.reshape(self.B, self.M, 4, 4)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv

class BlockSelfAttnMBTSmall_EntAttn(MeanScaleHyperprior):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneck(N) # override super() init

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, N),
            GDN(N),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, N),
            GDN(N, inverse=True),
            SelfAttn_AcrossBatch(N),
            SelfAttn(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential( 
            SelfAttn(M, downsample=1),
            # nn.AvgPool2d(2),
            conv(M, M, stride=2, kernel_size=3), 
            SelfAttn(M, downsample=1),
            conv(M, N, stride=1, kernel_size=1)
        )

        self.h_s = nn.Sequential(
            deconv(N, M*2, stride=1, kernel_size=1),
            SelfAttn(M*2, downsample=1),
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            deconv(M*2, M*2, stride=2, kernel_size=3),
            SelfAttn(M*2, downsample=1),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=3, padding=1, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        # y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0)
        y = y.permute(0, 2, 1, 3).reshape(1, y_height*self.B, self.M, y_width).permute(0, 2, 1, 3) #[1, M, 4*B, 4]

        z = self.h_a(y) # [1, N, 2*B, 2]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat) #[1, M, 4*B, 4]

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = y_hat.permute(0, 2, 1, 3).reshape(self.B, y_height, self.M, y_width).permute(0, 2, 1, 3) # [B, M, 4, 4]
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net


    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        y_height, y_width = y.shape[2], y.shape[3]
        y = y.reshape(self.B*self.M, y_height, y_width).unsqueeze(0)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 2  # scaling factor between z and y
        kernel_size = 3  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 2  # scaling factor between z and y
        kernel_size = 3  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M*self.B, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        # print(y_hat.shape)
        y_hat = y_hat.reshape(self.B, self.M, 4, 4)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv


class BlockConvScaleHyperpriorSmall(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(entropy_bottleneck_channels=B*N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3*B, N*B),
            GDN(N*B),
            conv(N*B, N*B),
            GDN(N*B),
            conv(N*B, M*B),
        )

        self.g_s = nn.Sequential(
            deconv(M*B, N*B),
            GDN(N*B, inverse=True),
            deconv(N*B, N*B),
            GDN(N*B, inverse=True),
            deconv(N*B, 3*B),
        )

        self.h_a = nn.Sequential(
            conv(B*M, B*N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            # conv(N, N),
            # nn.ReLU(inplace=True),
            conv(B*N, B*N),
        )

        self.h_s = nn.Sequential(
            deconv(B*N, B*N),
            # nn.ReLU(inplace=True),
            # deconv(N, N),
            nn.ReLU(inplace=True),
            conv(B*N, B*M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (3 + 1)

    def forward(self, x):
        x_height, x_width = x.shape[2], x.shape[3]
        x = x.reshape(self.B*3, x_height, x_width).unsqueeze(0) #x:[1, B*3 ,32, 32]
        y = self.g_a(x) # y:[1, B*M, 4, 4]
        z = self.h_a(torch.abs(y)) # z:[1, B*N, 2, 2]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat) # [1, B*M, 4, 4]
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        # print(y_likelihoods.shape, z_likelihoods.shape)
        x_hat = self.g_s(y_hat) # x_hat:[1, B*3, 32, 32]
        x_hat = x_hat.reshape(self.B, 3, x_height, x_width)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # @classmethod    
    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)


    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


    def compress(self, x):
        x_height, x_width = x.shape[2], x.shape[3]
        x = x.reshape(self.B*3, x_height, x_width).unsqueeze(0) #x:[1, B*3 ,32, 32]
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        x_hat = x_hat.reshape(self.B, 3, 32, 32)
        return {"x_hat": x_hat}

class BlockConv3dScaleHyperpriorSmall(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, B, **kwargs):
        super().__init__(entropy_bottleneck_channels=B*N, **kwargs)

        self.g_a = nn.Sequential(
            nn.Conv3d(3, N, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            # GDN(N), 
            nn.ReLU(),
            nn.Conv3d(N, N, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            # GDN(N),
            nn.ReLU(),
            nn.Conv3d(N, M, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2))
        )
        self.g_s = nn.Sequential(
            nn.ConvTranspose3d(M, N, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2), output_padding=(0,1,1)),
            # GDN(N, inverse=True),
            nn.ReLU(),
            nn.ConvTranspose3d(N, N, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2), output_padding=(0,1,1)),
            # GDN(N, inverse=True),
            nn.ReLU(),
            nn.ConvTranspose3d(N, 3, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2), output_padding=(0,1,1)),
        )

        self.h_a = nn.Sequential(
            conv(B*M, B*N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            # conv(N, N),
            # nn.ReLU(inplace=True),
            conv(B*N, B*N),
        )

        self.h_s = nn.Sequential(
            deconv(B*N, B*N),
            # nn.ReLU(inplace=True),
            # deconv(N, N),
            nn.ReLU(inplace=True),
            conv(B*N, B*M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.B = int(B) # blocklength

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (3 + 1)
    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x):
        # x: [B, 3, 32, 32]
        x_height, x_width = x.shape[2], x.shape[3]
        x = x.permute(1, 0, 2, 3).unsqueeze(0) # [1, 3, B, 32, 32]
        # print(x.shape)
        y = self.g_a(x) # y:[1, M, B, 4, 4]
        # print(y.shape)
        y = y.reshape(1, self.M*self.B, 4, 4) #[1, B*M, 4, 4]
        z = self.h_a(torch.abs(y)) # z:[1, B*N, 2, 2]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat) # [1, B*M, 4, 4]
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        # print(y_likelihoods.shape, z_likelihoods.shape)
        # print(y_hat.shape)
        y_hat = y_hat.reshape(1, self.M, self.B, 4, 4)
        # print(y_hat.shape)
        x_hat = self.g_s(y_hat) # x_hat:[1, 3, B, 32, 32]
        # print(x_hat.shape)
        x_hat = x_hat.squeeze().permute(1,0,2,3)
        # print(x_hat.shape)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        x_height, x_width = x.shape[2], x.shape[3]
        x = x.permute(1, 0, 2, 3).unsqueeze(0) # [1, 3, B, 32, 32]
        y = self.g_a(x) # y:[1, M, B, 4, 4]
        y = y.reshape(1, self.M*self.B, 4, 4) #[1, B*M, 4, 4]
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        y_hat = y_hat.reshape(1, self.M, self.B, 4, 4)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        x_hat = x_hat.squeeze().permute(1,0,2,3)
        # x_hat = x_hat.reshape(self.B, 3, 32, 32)
        return {"x_hat": x_hat}

from torch.nn.utils.parametrizations import orthogonal

class MobiusLayer(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

        self.A = orthogonal(nn.Linear(n, n, bias=False))
        # self.A = nn.Linear(n, n)
        self.b = nn.Parameter(torch.randn(n))
        self.a = nn.Parameter(torch.randn(n))
        self.alpha = nn.Parameter(torch.randn(1))
    def forward(self, x):
        # [bsize, n]
        x = self.alpha*self.A(x-self.a[None,:])
        x = x / ((x-self.a[None,:])**2).sum(dim=1)[:, None]
        x = x + self.b[None,:]
        return x

class SelfAttn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,downsample=8):
        super(SelfAttn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//downsample , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//downsample , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

def view_attn_map(selfattn, x):
    m_batchsize,C,width ,height = x.size()
    proj_query  = selfattn.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
    proj_key =  selfattn.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
    energy =  torch.bmm(proj_query,proj_key) # transpose check
    attention = selfattn.softmax(energy) # BX (N) X (N) 
    return attention

class SelfAttn_AcrossBatch(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, downsample=8):
        super(SelfAttn_AcrossBatch,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//downsample , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//downsample , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: N X B X B (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        x = x.permute(2,3,1,0).view(width*height, C, m_batchsize).unsqueeze(3) # [N,C,B,1]
        proj_query  = self.query_conv(x).view(width*height,-1,m_batchsize).permute(0,2,1) # N X CX(B)
        proj_key =  self.key_conv(x).view(width*height,-1,m_batchsize) # (*W*H) X C x B
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # (W*H)X B X B 
        proj_value = self.value_conv(x).view(width*height,-1,m_batchsize) # (W*H) X C X B

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(width*height,C,m_batchsize,1) #[N,C,B,1]
        
        out = self.gamma*out + x
        out = out.permute(2, 1, 0, 3).squeeze().view(m_batchsize, C, width, height)
        return out

def view_attn_map_acrossbatch(selfattn, x):
    m_batchsize,C,width ,height = x.size()
    x = x.permute(2,3,1,0).view(width*height, C, m_batchsize).unsqueeze(3) # [N,C,B,1]
    proj_query  = selfattn.query_conv(x).view(width*height,-1,m_batchsize).permute(0,2,1) # N X CX(B)
    proj_key =  selfattn.key_conv(x).view(width*height,-1,m_batchsize) # (*W*H) X C x B
    energy =  torch.bmm(proj_query,proj_key) # transpose check
    attention = selfattn.softmax(energy) # (W*H)X B X B 
    return attention
    

def eval_compression(model, x, dist_name="mse", discretize=False):
    # model = model.eval()
    with torch.no_grad():
        out_dict = model.compress(x)
    strings = out_dict['strings']
    with torch.no_grad():
        x_hat = model.decompress(strings, out_dict['shape'])['x_hat']
    # print(x_hat)
    
    if discretize:
        x_hat = torch.round(x_hat * 255) / 255
        # print(x.min(), x.max(), x_hat.min(), x_hat.max())
    # print(strings)
    rate = sum([len(bin(int.from_bytes(s, sys.byteorder))) for s_batch in strings for s in s_batch]) / len(x)
    # print(x.shape, x_hat.shape)
    if dist_name == "mse":
        # distortion = torch.mean(torch.linalg.vector_norm(x-x_hat, dim=(1,2,3))**2)
        distortion = F.mse_loss(x, x_hat)
    elif dist_name == 'ssim':
        distortion = 0.5*(1 - ssim(x, x_hat, data_range=1, size_average=True))
    return rate, distortion

def eval_compression_vqvae(model, x, discretize=False):
    # model = model.eval()
    with torch.no_grad():
        out_dict = model.compress(x)
    strings = out_dict['strings']
    idx_codebook = out_dict['idx_codebook']
    with torch.no_grad():
        x_hat = model.decompress(strings, out_dict['shape'], idx_codebook)['x_hat']
    # print(x_hat)
    
    if discretize:
        x_hat = torch.round(x_hat * 255) / 255
        # print(x.min(), x.max(), x_hat.min(), x_hat.max())
    # print(strings)
    rate = sum([len(bin(int.from_bytes(s, sys.byteorder))) for s_batch in strings for s in s_batch]) / len(x) + np.log2(model.embedding.weight.shape[0]) / len(x)
    # print(x.shape, x_hat.shape)
    distortion = torch.mean(torch.linalg.vector_norm(x-x_hat, dim=(1,2,3))**2)
    return rate, distortion
