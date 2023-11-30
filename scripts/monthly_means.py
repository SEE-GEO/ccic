"""
This script computes the monthly means from the existing CCIC data record.
"""

import argparse
from pathlib import Path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--product',
        choices=['gridsat', 'cpcir'],
        required=True,
        help="product to process"
    )
    parser.add_argument(
        '--source',
        type=Path,
        required=True,
        help="directory of the CCIC data record"
    )
    parser.add_argument(
        '--destination',
        type=Path,
        required=True,
        help="directory to save the monthly means"
    )
    parser.add_argument(
        '--month',
        required=True,
        help="month to process in the format YYMM"
    )
    parser.add_argument(
        '--month_end',
        nargs='?',
        default=None,
        help="process until this month in the format YYMM"
    )

    args = parser.parse_args()