"""
This script collocates DARDAR data with the data prepared for the training and
test sets of CCIC.

Requirements and assumptions:
- All CloudSat granules used in the training and test sets also exist in the
  DARDAR data and the latter are locally accessible, i.e. they do not need to
  be downloaded.
- No training or test scenes cross the antimeridian

Notes:
- The variables stored have `_dardar` appended instead of `_cloudsat` to
  differentiate them from the resampled CloudSat DPC data (2C-ICE, 2B-CLDCLASS)
"""

