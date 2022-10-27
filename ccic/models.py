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


SCALAR_VARIABLES = ["iwp", "cloud_flag", "iwp_rand"]
PROFILE_VARIABLES = ["iwc", "cloud_mask"]


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

        self.stem = block_factory(3, features)
        self.encoder = SpatialEncoder(
            input_channels=features,
            stages=[n_blocks] * n_stages,
            block_factory=block_factory,
            max_channels=512,
        )
        self.decoder = SpatialDecoder(
            output_channels=features,
            stages=[1] * n_stages,
            block_factory=block_factory,
            max_channels=512,
            skip_connections=True,
        )

        self.heads = nn.ModuleDict()
        for name in SCALAR_VARIABLES:
            self.heads[name] = MLP(
                features_in=features,
                n_features=2 * features,
                features_out=n_quantiles,
                n_layers=5,
                residuals="simple",
                activation_factory=nn.GELU,
                norm_factory=norm_factory,
            )
        for name in PROFILE_VARIABLES:
            self.heads[name] = MLP(
                features_in=features,
                n_features=2 * features,
                features_out=20 * n_quantiles,
                n_layers=5,
                residuals="simple",
                activation_factory=nn.GELU,
                norm_factory=norm_factory,
            )

    def forward(self, x):
        """
        Propagate input through network.
        """
        y = self.encoder(self.stem(x), return_skips=True)
        y = self.decoder(y)

        output = {}
        for name in SCALAR_VARIABLES:
            output[name] = self.heads[name](y)
        shape = y.shape
        profile_shape = [shape[0], self.n_quantiles, 20, shape[-2], shape[-1]]
        for name in PROFILE_VARIABLES:
            output[name] = self.heads[name](y).reshape(profile_shape)
        return output
