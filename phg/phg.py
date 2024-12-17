import os
import argparse
import pandas as pd
import PhenoGraph

def check_config(arguments):
    """Check that provided arguments are legit

    Args:
        arguments (argparse.ArgumentParser) : arguments catch by argparse

    Returns:
        check (bolean) : False if input file does not exist
        config (dict) : configuration procided by the user

    """

    # init configuration
    config = {}
    check = True

    # add output folder
    config["output"] = arguments.output
    if not os.path.isdir(config['output']):
        os.mkdir(config['output'])
        
    # add input file
    config["input"] = arguments.input
    if not os.path.isfile(config['input']):
        check = False

    # return config
    return check, config


def run(input_file, output_folder):
    """ """

    print("Tardis")


if __name__ == "__main__":
    
    
    # parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="path to the data file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path to the output folder, contains models",
    )

    # check inputs
    args = parser.parse_args()
    check, config = check_config(args)
    if check:

        # extract config
        input_file = config["input"]
        output_folder = config["output"]

        # run
        run(input_file, output_folder)
