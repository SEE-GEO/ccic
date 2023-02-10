# TODO: Clean up imports
import logging
from pathlib import Path
import sys
import datetime

from quantnn.mrnn import MRNN

from ccic.data.cpcir import CPCIR
from ccic.data.gridsat import GridSat
from ccic.data.training_data import CCICDataset
from ccic.processing import get_input_files
from torch.utils.data import DataLoader
from ccic.processing import process_input
import tqdm
import torch
from ccic.data.training_data import MASK_VALUE


def add_parser(subparsers):
    """
    Add parser for 'test' command to top-level parser. This function
    is called from the top-level parser defined in 'ccic.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "test",
        help="Test the network.",
        description=(
            """
            Test the network.
            """
        ),
    )
    parser.add_argument(
        "test_data",
        metavar="test_data",
        type=str,
        help="Folder containing the test data.",
    )
    parser.add_argument(
        "model_path",
        metavar="model_path",
        type=str,
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--batch_size",
        metavar="N",
        type=int,
        default=4,
        help="The batch size to use.",
    )
    parser.add_argument(
        "--accelerator",
        metavar="device",
        type=str,
        default="gpu",
        help="The accelerator to use for inference.",
    )
    parser.add_argument(
        "--name",
        metavar="name",
        type=str,
        default=__name__,
        help="Name to use for logging.",
    )
    parser.set_defaults(func=run)

def run(args):
    """
    Run test of the network.

    Args:
        args: The namespace object provided by the top-level parser
    """
    # Create logger
    LOGGER = logging.getLogger(args.name)

    test_data = Path(args.test_data)
    if not test_data.exists():
        LOGGER.error(
            "Provided test data path '%s' does not exist.",
            test_data.as_posix()
        )
        sys.exit()
    
    test_data = CCICDataset(test_data, all_channels=True, inference=True)
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=8,
        #worker_init_fn=test_data.seed # Commented out for reproducibility
        shuffle=False,
        pin_memory=True
    )

    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        LOGGER.error("The provides model '%s' does not exist.", model_path.name)
        return 1
    mrnn = MRNN.load(model_path)

    # Set evaluation mode
    mrnn.model.eval()
    with torch.no_grad():
        for x, y in tqdm.tqdm(test_loader, ncols=80):
            # Inference
            # y_hat = mrnn.model(x)
            #
            # Pick idxs for valid values (discards corrupt input data by modification in CCICDataset)
            # idx_valid = torch.where((y['tiwp'] > MASK_VALUE) & (x > MASK_VALUE))
            # 
            # filter data
            # y = {key: values[idx_valid] for key, values in y}
            # y_hat = {key: values[idx_valid] for key, values in y_hat}
            #
            # Save each pair of tensors to disk (probably provide argument)
            # Read (1)

            pass

# (1)
# `x` in line 110 could be omitted and rather do inference using ccic.processing.process_input_file
# to match the processing chain, as well as to easily save the predictions
# 
# However this implies fixing or supporting (if cpcir file, but same for gridsat)
# ccic.data.cpcir:242 as the time variable is a scalar and not a numpy array in the files we prepared
# [the source (CPCIR) files are 1-D array]