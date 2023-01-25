"""
========
ccic.bin
========

This sub-module implements the top-level 'ccic' command line application.
Its task is to delegate the processing to the sub-commands defined in
 the sub-module of the 'ccic.bin' module.
"""
import argparse
import sys
import warnings


def ccic():
    """
    This function implements the top-level command line interface for the
    'ccic' package. It serves as the global entry point to execute
    any of the available sub-commands.
    """
    from ccic.bin import train
    from ccic.bin import process
    from ccic.bin import extract_training_data
    from ccic.bin import test
    from ccic.bin import run_validation_retrieval

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    description = "ccic: The Chalmers cloud-ice climatology"

    parser = argparse.ArgumentParser(prog="ccic", description=description)

    subparsers = parser.add_subparsers(help="Sub-commands")
    train.add_parser(subparsers)
    process.add_parser(subparsers)
    extract_training_data.add_parser(subparsers)
    test.add_parser(subparsers)
    run_validation_retrieval.add_parser(subparsers)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 0

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    ccic()
