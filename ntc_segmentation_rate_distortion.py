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

import math

import torch
import torch.nn as nn

from compressai.registry import register_criterion
from compressai.losses import RateDistortionLoss


@register_criterion("RateDistortionLoss")
class RateDistortionLossSeg(RateDistortionLoss):
    """Custom rate distortion loss with a Lagrangian parameter.
    Implemented pixel-wise cross-entropy as a loss metric to support
    semantic segmentation maps as inputs."""

    def __init__(self, lmbda=0.01, metric="cross_entropy", return_type="all"):
        super().__init__()
        if metric == "cross_entropy":
            self.metric = nn.CrossEntropyLoss()
        elif metric == "mse":
            self.metric = nn.MSELoss()
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metric == nn.MSELoss():
            out["mse_loss"] = self.metric(output["x_hat"], target)
            distortion = 255**2 * out["mse_loss"]
        else:
            # cross entropy requires target to be 3-dimensional and type Long
            # squeeze the target on the channel dimension such that
            # torch.shape is transformed: (batch_size,1,H,W) -> (batch_size,H,W)
            target = torch.squeeze(target, dim=1)
            target = target.type(dtype=torch.long)
            out["cross_entropy_loss"] = self.metric(output["x_hat"], target)
            distortion = out["cross_entropy_loss"]


        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]