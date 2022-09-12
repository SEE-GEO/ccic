"""
cimr.models
===========

Machine learning models for CIMR.
"""
import torch
from torch import nn

from cimr.utils import MISSING, MASK


class SymmetricPadding(nn.Module):
    """
    Network module implementing symmetric padding.

    This is just a wrapper around torch's ``nn.functional.pad`` with mode
    set to 'replicate'.
    """

    def __init__(self, amount):
        super().__init__()
        if isinstance(amount, int):
            self.amount = [amount] * 4
        else:
            self.amount = amount

    def forward(self, x):
        return nn.functional.pad(x, self.amount, "replicate")


class SeparableConv(nn.Sequential):
    """
    Depth-wise separable convolution using with kernel size 3x3.
    """

    def __init__(self, channels_in, channels_out, size=7):
        super().__init__(
            nn.Conv2d(
                channels_in,
                channels_in,
                kernel_size=7,
                groups=channels_in
            ),
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_out, kernel_size=1),
        )

class ConvNextBlock(nn.Module):
    def __init__(self, n_channels, n_channels_out=None, size=7, activation=nn.GELU):
        super().__init__()

        if n_channels_out is None:
            n_channels_out = n_channels
        self.body = nn.Sequential(
            SymmetricPadding(3),
            SeparableConv(n_channels, 2 * n_channels_out, size=size),
            activation(),
            nn.Conv2d(2 * n_channels_out, n_channels_out, kernel_size=1),
        )

        if n_channels != n_channels_out:
            self.projection = nn.Conv2d(n_channels, n_channels_out, kernel_size=1)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        y = self.body(x)
        return y + self.projection(x)


class DownsamplingBlock(nn.Sequential):
    """
    Xception downsampling block.
    """

    def __init__(self, channels_in, channels_out, bn_first=True):
        if bn_first:
            blocks = [
                nn.BatchNorm2d(channels_in),
                nn.Conv2d(channels_in, channels_out, kernel_size=2, stride=2)
            ]
        else:
            blocks = [
                nn.Conv2d(channels_in, channels_out, kernel_size=2, stride=2),
                nn.BatchNorm2d(channels_out)
            ]
        super().__init__(*blocks)


class DownsamplingStage(nn.Sequential):
    def __init__(self, channels_in, channels_out, n_blocks, size=7):
        blocks = [DownsamplingBlock(channels_in, channels_out)]
        for i in range(n_blocks):
            blocks.append(ConvNextBlock(channels_out, size=size))
        super().__init__(*blocks)


class UpsamplingStage(nn.Module):
    """
    Xception upsampling block.
    """
    def __init__(self, channels_in, channels_skip, channels_out, size=7):
        """
        Args:
            n_channels: The number of incoming and outgoing channels.
        """
        super().__init__()
        self.upsample = nn.Upsample(mode="bilinear",
                                    scale_factor=2,
                                    align_corners=False)
        self.block = nn.Sequential(
            nn.Conv2d(channels_in + channels_skip, channels_out, kernel_size=1),
            ConvNextBlock(channels_out, size=size)
        )

    def forward(self, x, x_skip):
        """
        Propagate input through block.
        """
        x_up = self.upsample(x)
        if x_skip is not None:
            x_merged = torch.cat([x_up, x_skip], 1)
        else:
            x_merged = x_up
        return self.block(x_merged)


class EncoderDecoder(nn.Module):
    """
    The CIMR Seviri baseline model, which only uses SEVIRI observations
    for the retrieval.
    """
    def __init__(
            self,
            n_stages,
            features,
            n_quantiles,
            n_blocks=2
    ):
        """
        Args:
            n_stages: The number of stages in the encode
            features: The base number of features.
            n_quantiles: The number of quantiles to predict.
            n_blocks: The number of blocks in each stage.
        """
        super().__init__()
        self.n_quantiles = n_quantiles

        n_channels_in = 3

        if not isinstance(n_blocks, list):
            n_blocks = [n_blocks] * n_stages

        stages = []
        ch_in = n_channels_in
        ch_out = features
        for i in range(n_stages):
            stages.append(DownsamplingStage(ch_in, ch_out, n_blocks[i]))
            ch_in = ch_out
            ch_out = ch_out * 2
        self.down_stages = nn.ModuleList(stages)

        stages =[]
        ch_out = ch_in // 2
        for i in range(n_stages):
            ch_skip = ch_out if i < n_stages - 1 else n_channels_in
            stages.append(UpsamplingStage(ch_in, ch_skip, ch_out))
            ch_in = ch_out
            ch_out = ch_out // 2 if i < n_stages - 2 else features
        self.up_stages = nn.ModuleList(stages)

        self.up = UpsamplingStage(features, 0, features)


        self.head_iwp = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, n_quantiles, kernel_size=1),
        )

        self.head_iwc = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, 20 * n_quantiles, kernel_size=1),
        )


    def forward(self, x):
        """Propagate input though model."""
        skips = []
        y = x
        for stage in self.down_stages:
            skips.append(y)
            y = stage(y)

        skips.reverse()
        for skip, stage in zip(skips, self.up_stages):
            y = stage(y, skip)

        profile_shape = list(x.shape)
        profile_shape[1] = 20
        profile_shape.insert(1, self.n_quantiles)

        return {
            "iwp": self.head_iwp(y),
            "iwc": self.head_iwc(y).reshape(tuple(profile_shape))
        }
