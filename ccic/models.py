"""
ccic.models
===========

Machine learning models for CCIC.
"""
from torch import nn

from quantnn.models.pytorch.encoders import SpatialEncoder
from quantnn.models.pytorch.decoders import SpatialDecoder
from quantnn.models.pytorch.fully_connected import MLP
import quantnn.models.pytorch.torchvision as blocks


SCALAR_VARIABLES = ["iwp", "cloud_mask", "iwp_rand"]
PROFILE_VARIABLES = ["iwc", "cloud_class"]


class CCICModel(nn.Module):
    """
    The CCIC retrieval model.

    The neural network is a simple encoder-decoder architecture with U-Net-type
    skip connections, and a head for every output variables.
    """

    def __init__(self, n_stages, features, n_quantiles, n_blocks=2):
        """
        Args:
            n_stages: The number of stages in the encoder and decoder.
            features: The number of features at the highest resolution.
            n_quantiles: The number of quantiles to predict.
            n_blocks: The number of blocks in each stage.
        """
        super().__init__()

        self.n_quantiles = n_quantiles
        n_channels_in = 3

        block_factory = blocks.ConvNextBlockFactory()
        norm_factory = block_factory.layer_norm

        self.stem = block_factory(n_channels_in, features)
        self.encoder = SpatialEncoder(
            channels=features,
            stages=[n_blocks] * n_stages,
            block_factory=block_factory,
            max_channels=512,
        )
        self.decoder = SpatialDecoder(
            channels=features,
            stages=[1] * n_stages,
            block_factory=block_factory,
            max_channels=512,
            skip_connections=True,
        )

        self.heads = nn.ModuleDict()

        head_factory = lambda n_out: MLP(
                features_in=features,
                n_features=2 * features,
                features_out=n_out,
                n_layers=5,
                residuals="simple",
                activation_factory=nn.GELU,
                norm_factory=norm_factory,
        )

        self.heads["iwc"] = head_factory(20 * self.n_quantiles // 4)
        self.heads["iwp"] = head_factory(self.n_quantiles)
        self.heads["iwp_rand"] = head_factory(self.n_quantiles)
        self.heads["cloud_mask"] = head_factory(1)
        self.heads["cloud_class"] = head_factory(20 * 9)

    def forward(self, x):
        """
        Propagate input through network.
        """
        y = self.encoder(self.stem(x), return_skips=True)
        y = self.decoder(y)

        output = {}

        output["iwp"] = self.heads["iwp"](y)
        output["iwp_rand"] = self.heads["iwp_rand"](y)
        output["cloud_mask"] = self.heads["cloud_mask"](y)

        shape = y.shape
        profile_shape = [shape[0], self.n_quantiles // 4, 20, shape[-2], shape[-1]]
        head = self.heads["iwc"]
        output["iwc"] = head(y).reshape(profile_shape)

        profile_shape = [shape[0], 9, 20, shape[-2], shape[-1]]
        head = self.heads["cloud_class"]
        output["cloud_class"] = head(y).reshape(profile_shape)

        return output
