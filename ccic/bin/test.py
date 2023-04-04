"""
ccic.bin.test
=============

Implements a command line interface for running the CCIC neural network
 on test data.
"""
import logging
from pathlib import Path
import sys
import datetime

from quantnn.mrnn import MRNN

from ccic.data.training_data import CCICDataset
from ccic.processing import get_input_files
from ccic.processing import process_input
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from ccic.data.training_data import MASK_VALUE
import xarray as xr


def add_parser(subparsers):
    """
    Add parser for 'test' command to top-level parser. This function
    is called from the top-level parser defined in 'ccic.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "test",
        help="Evaluate retrieval network on test data.",
        description=(
            """
            Applies a given CCIC neural network model to input data in the
            format used for the training data. Produces outputs containing
            the retrieval results combined with the reference values.
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
        "output_file",
        metavar="path",
        type=str,
        help="Path to the file to which to write the results."
    )
    parser.add_argument(
        "--batch_size",
        metavar="N",
        type=int,
        default=4,
        help="The batch size to use.",
    )
    parser.add_argument(
        "--device",
        metavar="device",
        type=str,
        default="cuda",
        help="The device to use for inference.",
    )
    parser.add_argument(
        "--name",
        metavar="name",
        type=str,
        default=__name__,
        help="Name to use for logging.",
    )
    parser.set_defaults(func=run)


def process_dataset(mrnn, data_loader, device="cpu"):
    """
    Process all scenes in a dataset.

    Args:
        mrnn: quantnn MRNN object holding the retrieval model.
        data_loader: torch dataloader providing access to the test data.
        device: The device to run the processing on.

    Return:
        An 'xarray.Dataset' containing the evaluation results.
    """
    tiwp_mean = []
    tiwp_sample = []
    tiwp_log_std_dev = []
    tiwp_true = []

    tiwp_fpavg_mean = []
    tiwp_fpavg_sample = []
    tiwp_fpavg_log_std_dev = []
    tiwp_fpavg_true = []

    tiwc_mean = []
    tiwc_sample = []
    tiwc_true = []

    cloud_class_prob = []
    cloud_class_true = []

    cloud_prob = []
    cloud_true = []

    scene = []
    latitude = []
    longitude = []
    granule = []
    encodings = []
    enc_inds = []
    enc_inds = []
    tbs = []

    mrnn.model.eval()
    mrnn.model.to(device)

    # We will put quantiles along first axis to simplify extraction
    # of valid pixels.
    mrnn.losses["tiwp"].quantile_axis = 0
    mrnn.losses["tiwp_fpavg"].quantile_axis = 0
    mrnn.losses["tiwc"].quantile_axis = 0

    batch_index = 0

    with torch.no_grad():
        for x, y in tqdm.tqdm(data_loader, ncols=80):

            x = x.to(device)
            y = {key: y[key].to(device) for key in y}

            y_pred = mrnn.model(x, return_encodings=True)

            # Extract valid pixels
            valid = y["tiwp"] > MASK_VALUE

            enc = y_pred.pop("encodings")
            ds_fac = valid.shape[-1] // enc.shape[-1]
            new_shape = (
                valid.shape[0],
                valid.shape[1] // ds_fac,
                ds_fac,
                valid.shape[2] // ds_fac,
                ds_fac
            )
            valid_ds = valid.reshape(new_shape).any(2).any(-1)
            enc = torch.permute(enc, (1, 0, 2, 3))
            encodings.append(torch.transpose(enc[..., valid_ds], 0, 1).cpu().numpy())

            batch_inds, row_inds, col_inds = np.where(valid.cpu().numpy())
            row_inds = row_inds // ds_fac
            col_inds = col_inds // ds_fac
            enc = enc.cpu().numpy()
            inds = np.arange(enc[:, 0].size).reshape(enc[:, 0].shape)
            enc_inds.append(inds[batch_inds, row_inds, col_inds])

            for key in y_pred:
                y_k = y[key]
                y_pred_k = y_pred[key]

                transform = mrnn.transformation[key]
                if transform is not None:
                    y_pred_k = transform.invert(y_pred_k)

                if valid.ndim == y_k.ndim:
                    y[key] = y_k[valid]
                    y_pred_k = torch.permute(y_pred_k, (1, 0, 2, 3))
                    y_pred[key] = y_pred_k[..., valid]
                else:
                    y_k = torch.permute(y_k, (1, 0, 2, 3))
                    y[key] = y_k[..., valid]
                    y_pred_k = torch.permute(y_pred_k, (1, 2, 0, 3, 4))
                    y_pred[key] = y_pred_k[..., valid]

            tiwp_mean.append(
                mrnn.posterior_mean(y_pred=y_pred["tiwp"], key="tiwp").cpu()
            )
            tiwp_sample.append(
                mrnn.sample_posterior(y_pred=y_pred["tiwp"], key="tiwp").cpu()[0]
            )
            tiwp_true.append(y["tiwp"].cpu().numpy())
            tiwp_log_std_dev.append(
                mrnn.posterior_std_dev(
                    y_pred=torch.log10(y_pred["tiwp"]),
                    key="tiwp"
                )
                .cpu()
                .float()
                .numpy()
            )

            tiwp_fpavg_mean.append(
                mrnn.posterior_mean(y_pred=y_pred["tiwp_fpavg"], key="tiwp_fpavg").cpu()
            )
            tiwp_fpavg_sample.append(
                mrnn.sample_posterior(y_pred=y_pred["tiwp_fpavg"], key="tiwp_fpavg").cpu()[0]
            )
            tiwp_fpavg_true.append(y["tiwp_fpavg"].cpu().numpy())
            tiwp_fpavg_log_std_dev.append(
                mrnn.posterior_std_dev(
                    y_pred=torch.log10(y_pred["tiwp_fpavg"]),
                    key="tiwp_fpavg")
                .cpu()
                .float()
                .numpy()
            )

            tiwc_mean.append(
                torch.transpose(
                    mrnn.posterior_mean(y_pred=y_pred["tiwc"], key="tiwc").cpu(),
                    1, 0
                )
            )
            tiwc_sample.append(
                torch.transpose(
                    mrnn.sample_posterior(y_pred=y_pred["tiwc"], key="tiwc").cpu()[0],
                    1, 0
                )
            )
            tiwc_true.append(y["tiwc"].cpu().numpy().transpose(1, 0))

            cloud_class_prob.append(
                torch.permute(
                    torch.softmax(y_pred["cloud_class"], 0),
                    (2, 1, 0)
                ).cpu().numpy()
            )
            cloud_class_true.append(
                torch.transpose(y["cloud_class"], 0, 1)
                .cpu().numpy()
            )

            cloud_prob.append(
                torch.sigmoid(y_pred["cloud_mask"]).cpu().numpy()[0]
            )
            cloud_true.append(
                y["cloud_mask"].cpu().numpy()
            )

            scene.append(batch_index + np.where(valid.cpu().numpy())[0])
            batch_index += x.shape[0]

            latitude.append(y["latitude"][valid].cpu().numpy())
            longitude.append(y["longitude"][valid].cpu().numpy())
            granule.append(y["granule"][valid].cpu().numpy())
            tbs.append(y["tbs"][valid].cpu().numpy())

    tiwp_mean = np.concatenate([tensor.numpy() for tensor in tiwp_mean])
    tiwp_sample = np.concatenate([tensor.numpy() for tensor in tiwp_sample])
    tiwp_true = np.concatenate(tiwp_true)
    tiwp_log_std_dev = np.concatenate(tiwp_log_std_dev)
    tiwp_fpavg_mean = np.concatenate([tensor.numpy() for tensor in tiwp_fpavg_mean])
    tiwp_fpavg_sample = np.concatenate([tensor.numpy() for tensor in tiwp_fpavg_sample])
    tiwp_fpavg_true = np.concatenate(tiwp_fpavg_true)
    tiwp_fpavg_log_std_dev = np.concatenate(tiwp_fpavg_log_std_dev)
    tiwc_mean = np.concatenate([tensor.numpy() for tensor in tiwc_mean])
    tiwc_sample = np.concatenate([tensor.numpy() for tensor in tiwc_sample])
    tiwc_true = np.concatenate(tiwc_true)
    cloud_class_prob = np.concatenate(cloud_class_prob)
    cloud_class_true = np.concatenate(cloud_class_true)
    cloud_prob = np.concatenate(cloud_prob)
    cloud_true = np.concatenate(cloud_true)
    scene = np.concatenate(scene)
    encodings = np.concatenate(encodings)
    enc_inds = np.concatenate(enc_inds)
    latitude = np.concatenate(latitude)
    longitude = np.concatenate(longitude)
    granule = np.concatenate(granule)
    tbs = np.concatenate(tbs)

    levels = (np.arange(20) + 0.5) * 1e3
    dataset = xr.Dataset({
        "levels": (("levels",), levels),
        "tiwp_mean": (("samples",), tiwp_mean),
        "tiwp_sample": (("samples",), tiwp_sample),
        "tiwp_true": (("samples",), tiwp_true),
        "tiwp_log_std_dev": (("samples",), tiwp_log_std_dev),
        "tiwp_fpavg_mean": (("samples",), tiwp_fpavg_mean),
        "tiwp_fpavg_sample": (("samples",), tiwp_fpavg_sample),
        "tiwp_fpavg_true": (("samples",), tiwp_fpavg_true),
        "tiwp_fpavg_log_std_dev": (("samples",), tiwp_fpavg_log_std_dev),
        "tiwc_mean": (("samples", "levels"), tiwc_mean),
        "tiwc_sample": (("samples", "levels"), tiwc_sample),
        "tiwc_true": (("samples", "levels"), tiwc_true),
        "cloud_class_prob": (("samples", "levels", "classes",), cloud_class_prob),
        "cloud_class_true": (("samples", "levels"), cloud_class_true),
        "cloud_prob": (("samples",), cloud_prob),
        "cloud_prob_true": (("samples",), cloud_true),
        "scene": (("samples",), scene),
        "encodings": ((f"samples_{ds_fac}", "features"), encodings),
        "encoding_indices": (("samples",), enc_inds),
        "longitude": (("samples",), longitude),
        "latitude": (("samples",), latitude),
        "granule": (("samples",), granule),
        "tbs": (("samples",), tbs),
    })

    enc_float = {"dtype": "float32", "zlib": True}
    enc_prob = {
        "dtype": "uint8",
        "scale_factor": 1 / 250,
        "_FillValue": 255,
        "zlib":True
    }
    enc_class = {
        "dtype": "uint8",
        "_FillValue": 255,
    }

    dataset.tiwp_mean.encoding = enc_float
    dataset.tiwp_sample.encoding = enc_float
    dataset.tiwp_true.encoding = enc_float
    dataset.tiwp_log_std_dev.encoding = enc_float
    dataset.tiwp_fpavg_mean.encoding = enc_float
    dataset.tiwp_fpavg_sample.encoding = enc_float
    dataset.tiwp_fpavg_true.encoding = enc_float
    dataset.tiwp_fpavg_log_std_dev.encoding = enc_float
    dataset.tiwc_mean.encoding = enc_float
    dataset.tiwc_sample.encoding = enc_float
    dataset.tiwc_true.encoding = enc_float
    dataset.latitude.encoding = enc_float
    dataset.longitude.encoding = enc_float
    dataset.tbs.encoding = enc_float
    dataset.cloud_class_prob.encoding = enc_prob
    dataset.cloud_class_true.encoding = enc_class
    dataset.cloud_prob.encoding = enc_prob
    dataset.cloud_prob_true.encoding = enc_class
    dataset.encodings.encoding = enc_float

    return dataset


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

    test_data = CCICDataset(test_data, inference=True)
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True
    )

    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        LOGGER.error("The provides model '%s' does not exist.", model_path.name)
        return 1
    mrnn = MRNN.load(model_path)

    results = process_dataset(mrnn, test_loader, device=args.device)

    results.to_netcdf(args.output_file)
