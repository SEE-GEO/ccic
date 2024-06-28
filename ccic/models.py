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


SCALAR_VARIABLES = ["tiwp", "tiwp_fpavg", "cloud_mask"]
PROFILE_VARIABLES = ["tiwc", "cloud_class"]


class CCICModel(nn.Module):
    """
    The CCIC retrieval model.

    The neural network is a simple encoder-decoder architecture with U-Net-type
    skip connections, and a head for every output variables.
    """

    def __init__(
            self,
            n_stages,
            features,
            n_quantiles,
            n_blocks=2,
            all_channels=False,
            outputs: list[str]=None
    ):
        """
        Args:
            n_stages: The number of stages in the encoder and decoder.
            features: The number of features at the highest resolution.
            n_quantiles: The number of quantiles to predict.
            n_blocks: The number of blocks in each stage.
            all_channels: If set to 'True' the network will expect three input
                 channels, which are available only for the Gridsat dataset.
            outputs: if not None, consider only these outputs
        """
        super().__init__()
        self.all_channels = all_channels

        self.n_quantiles = n_quantiles
        n_channels_in = 3 if self.all_channels else 1

        block_factory = blocks.ConvNeXtBlockFactory()
        norm_factory = block_factory.layer_norm

        self.stem = nn.Conv2d(n_channels_in, features, 3, padding=1)
        self.encoder = SpatialEncoder(
            channels=features,
            stages= [0] + [n_blocks] * n_stages,
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

        self.outputs = outputs or (SCALAR_VARIABLES + PROFILE_VARIABLES)

        head_config = {
            "tiwc": 20 * self.n_quantiles // 4,
            "tiwp": self.n_quantiles,
            "tiwp_fpavg": self.n_quantiles,
            "cloud_mask": 1,
            "cloud_class": 20 * 9
        }

        # Sanity check
        assert len(set(self.outputs) - set(head_config)) == 0

        for oname in self.outputs:
            self.heads[oname] = head_factory(head_config[oname])
        
        self.version = 0.1

    def forward_w_feature_maps(self, x):
        """
        Propagate input through network and return intermediate activations.


        Args:
            x: A torch.tensor containing the input to feed through the
                network.

        Return:

        """
        output = {}
        y = self.stem(x)
        activations = [y] + self.encoder(y, return_skips=True)

        y = self.decoder.forward_w_intermediate(activations)
        activations += y
        y = y[-1]
        shape = y.shape

        for oname in ["tiwp", "tiwp_fpavg", "cloud_mask"]:
            if oname in self.outputs:
                output[oname] = self.heads[oname](y)

        if "tiwc" in self.outputs:
            profile_shape = [shape[0], self.n_quantiles // 4, 20, shape[-2], shape[-1]]
            head = self.heads["tiwc"]
            output["tiwc"] = head(y).reshape(profile_shape)

        if "cloud_class" in self.outputs:
            profile_shape = [shape[0], 9, 20, shape[-2], shape[-1]]
            head = self.heads["cloud_class"]
            output["cloud_class"] = head(y).reshape(profile_shape)

        return output, activations

    def forward(self, x, return_encodings=False):
        """
        Propagate input through network.

        Args:
            x: A torch.tensor containing the input to feed through the
                network.
            return_encodings: If set to true, the output from the encoder
                is included in the output.

        Return:
            A dictionary containing the network outputs.
        """
        output = {}
        y = self.stem(x)

        y = self.encoder(y, return_skips=True)

        if return_encodings:
            output["encodings"] = y[max(y)]
        y = self.decoder(y)
        shape = y.shape

        for oname in SCALAR_VARIABLES:
            if oname in self.outputs:
                output[oname] = self.heads[oname](y)

        if "tiwc" in self.outputs:
            profile_shape = [shape[0], self.n_quantiles // 4, 20, shape[-2], shape[-1]]
            head = self.heads["tiwc"]
            output["tiwc"] = head(y).reshape(profile_shape)

        if "cloud_class" in self.outputs:
            profile_shape = [shape[0], 9, 20, shape[-2], shape[-1]]
            head = self.heads["cloud_class"]
            output["cloud_class"] = head(y).reshape(profile_shape)

        return output
