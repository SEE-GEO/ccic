"""
Addresses https://github.com/SEE-GEO/ccic/issues/85
by creating a copy of the model with only the heads specified as an argument.
"""

import argparse
from pathlib import Path

from quantnn.mrnn import MRNN

from ccic.models import CCICModel, SCALAR_VARIABLES, PROFILE_VARIABLES

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Create a copy of the network with "
            "only the specified outputs"
        )
    )
    parser.add_argument(
        "--input_network", required=True, type=Path,
        help="Path to the original network."
    )
    parser.add_argument(
        "--output_network", required=True, type=Path,
        help="File path to save the network with less heads."
    )
    parser.add_argument(
        "--outputs", nargs='+', required=True,
        choices=SCALAR_VARIABLES + PROFILE_VARIABLES,
        help="Which outputs to consider"
    )

    args = parser.parse_args()

    mrnn = MRNN.load(args.input_network)

    smaller_model = CCICModel(
        n_stages=5,
        features=96,
        n_quantiles=64,
        n_blocks=4,
        outputs=args.outputs
    )

    target_state_dict = {
        k: v
        for k, v in mrnn.model.state_dict().items()
        if k in smaller_model.state_dict()
    }

    smaller_model.load_state_dict(target_state_dict)

    mrnn.model = smaller_model

    mrnn.save(args.output_network)