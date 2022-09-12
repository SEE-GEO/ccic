"""
==============
ccic.bin.train
==============

This sub-module implements the ccic CLI to train the retrieval.
"""
from calendar import monthrange
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import importlib
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'train' command to top-level parser. This function
    is called from the top-level parser defined in 'ccic.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "train",
        help="Train retrieval network.",
        description=(
            """
            Train the retrieval network.
            """
        ),
    )
    parser.add_argument(
        "training_data",
        metavar="training_data",
        type=str,
        help="Folder containing the training data.",
    )
    parser.add_argument(
        "model_path",
        metavar="model_path",
        type=str,
        help="Path to store the trained model.",
    )
    parser.add_argument(
        "--validation_data",
        metavar="path",
        type=str,
        help="Folder containing the validation data.",
        default=None
    )
    parser.add_argument(
        "--n_stages",
        metavar="N",
        type=int,
        default=4,
        help="Number of stages in encoder/decoder architecture.",
    )
    parser.add_argument(
        "--n_features",
        metavar="N",
        type=int,
        default=128,
        help="Number of features in first encoder stage.",
    )
    parser.add_argument(
        "--n_blocks",
        metavar="N",
        type=int,
        default=4,
        help="Number of blocks per encoder stage.",
    )
    parser.add_argument(
        "--batch_size",
        metavar="N",
        type=int,
        default=4,
        help="The batch size to use during training.",
    )
    parser.add_argument(
        "--lr",
        metavar="lr",
        type=int,
        default=0.0005,
        help="The learning rate with which to start the training",
    )
    parser.add_argument(
        "--n_epochs",
        metavar="n_epochs",
        type=int,
        default=20,
        help="The number of epochs to train the model for.",
    )
    parser.add_argument(
        "--accelerator",
        metavar="device",
        type=str,
        default="gpu",
        help="The accelerator to use for training.",
    )

    parser.set_defaults(func=run)


def run(args):
    """
    Run training.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor
    from quantnn.qrnn import QRNN
    from quantnn import metrics
    from quantnn import transformations
    from ccic.models import EncoderDecoder

    #
    # Prepare training and validation data.
    #

    from ccic.data.training_data import CCICDataset
    from torch.utils.data import DataLoader

    training_data = Path(args.training_data)
    if not training_data.exists():
        LOGGER.error(
            "Provided training data path '%s' doesn't exist."
        )
        sys.exit()

    training_data = CCICDataset(training_data)
    training_loader = DataLoader(
        training_data,
        batch_size=2,
        num_workers=8,
        worker_init_fn=training_data.seed,
        shuffle=True,
        pin_memory=True
    )

    validation_data = args.validation_data
    if validation_data is not None:
        validation_data = Path(validation_daata)
        if not validation_data.exists():
            LOGGER.error(
                "Provided validation data path '%s' doesn't exist."
            )
            sys.exit()
        validation_data = CCICDataset(validation_data)
        validation_loader = DataLoader(
            validation_data,
            batch_size=args.batch_size,
            num_workers=8,
            worker_init_fn=validation_data.seed,
            shuffle=True,
            pin_memory=True
        )

    #
    # Create model
    #

    n_stages = args.n_stages
    n_blocks = args.n_blocks
    n_features = args.n_features

    model_path = Path(args.model_path)

    transformations = {
        "iwc": transformations.LogLinear(),
        "iwp": transformations.LogLinear()
    }

    if model_path.exists() and not model_path.is_dir():
        qrnn = QRNN.load(model_path)
        model = qrnn.model
    else:
        if model_path.is_dir():
            model_path / f"ccic_{n_stages}_{n_blocks}_{n_features}.pckl"

        model = EncoderDecoder(
            n_stages,
            n_features,
            n_quantiles=64,
            n_blocks=n_blocks
        )
        quantiles = np.linspace(0, 1, 66)[1:-1]
        qrnn = QRNN(
            model=model,
            quantiles=quantiles
        )

    #
    # Run training
    #

    metrics = [
        metrics.MeanSquaredError(),
        metrics.Correlation(),
        metrics.CRPS(),
        metrics.Bias(),
        metrics.ScatterPlot(),
        metrics.Correlation(),
        metrics.ScatterPlot(),
        metrics.CalibrationPlot()
    ]
    lm = qrnn.lightning(mask=-100, metrics=metrics)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    lm.optimizer = optimizer
    lm.scheduler = scheduler

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        accelerator=args.accelerator,
        precision=16,
        limit_val_batches=10,
        limit_train_batches=10,
        logger=lm.tensorboard,
        callbacks=[LearningRateMonitor()]
    )
    trainer.fit(model=lm, train_dataloaders=training_loader, val_dataloaders=training_loader)

    qrnn.save(model_path)
