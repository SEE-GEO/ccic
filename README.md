# The Chalmers Cloud Ice Climatology

This repository contains all source code for the training and processing of the retrieval underlying the Chalmers Cloud Ice Climatology (CCIC) data record.

## Documentation

Documentation describing the CCIC data, its uses, and how to run the retrievals can be found at [https://ccic.readthedocs.io](https://ccic.readthedocs.io/en/latest/intro.html).

## Data

At the time of writing (March 2026):

- The [Registry of Open Data on AWS](https://registry.opendata.aws) contains all CCIC total ice water path estimates until 2023. The data can be found [here](https://registry.opendata.aws/ccic/).

- https://clouds-and-precip.group/datasets/ccic/ contains all CCIC total ice water path estimates until February 2026, and 2D cloud masks since 2024. Data is updated in approximately a monthly basis.

## References

The CCIC retrieval, its implementation, and validation results are presented in
> Amell, A., Pfreundschuh, S., and Eriksson, P.: The Chalmers Cloud Ice Climatology: Retrieval implementation and validation, Atmos. Meas. Tech., 17, 4337–4368 [https://doi.org/10.5194/amt-17-4337-2024](https://doi.org/10.5194/amt-17-4337-2024), 2024.

A fine-tuning of the retrieval of cloud probabilities is described in

> Amell, A., Pfreundschuh, S., and Eriksson, P.: Fine-tuning a machine-learned 3D cloud climatology reveals aspects of cloud cover trends, ESS Open Archive [preprint], [https://doi.org/10.22541/essoar.15001993/v1](https://doi.org/10.22541/essoar.15001993/v1), 2026.