import pandas as pd
from Ensembl_converter import EnsemblConverter
import re
import argparse
import os

from alive_progress import alive_bar


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

    # check that provided data exist
    if not os.path.isfile(arguments.input):
        print(f"[!] Can't find {arguments.input}")
        return False, {}
    else:
        config["input"] = arguments.input

    # add output folder and label
    config["output"] = arguments.output

    # return config
    return True, config



def run(data_file:str, output_dir:str):
    """Run conversion of gene ensemblID to Gene symbol
    input_file can be row oriented or column oriented, in the
    case ensemblID are placed in columns; it must be the first column of dataframe

    Args:
        - data_file (str) : path to the input data file
        - output_dir (str) : path to the output folder

    Returns:
        - (str) : only if something went wrong
    
    """

    # load data
    df = pd.read_csv(data_file)

    # init log file
    log_data = open(f"{output_dir}/conversion.log", "w")

    # Create an instance of EnsemblConverter
    converter = EnsemblConverter()

    # check if gene are in column or in rows
    ensembl_ids = []
    use_col = False
    for col in list(df.keys()):
        if re.search('ENSG', col):
            use_col = True
            ensembl_ids.append(col)

    # check row of first column
    use_row = False
    if not use_col:
        for row in list(df[df.keys()[0]]):
            if re.search('ENSG', row):
                use_row = True
                ensembl_ids.append(row)

    # stop if no ensembl id is detected
    if not use_row and not use_col:
        return "No Ensembl ID found"

    # Convert Ensembl IDs to gene symbols
    id_to_name = {}
    with alive_bar(len(ensembl_ids), length=75) as bar:
        for id in ensembl_ids:
            name = converter.convert_ids([id])
            try:
                id_to_name[id] = name['Symbol'][0]
            except:
                id_to_name[id] = id
                log_data.write(f"[!] Fail to convert {id}\n")
            bar()

    # deal with row
    if use_row:
        df[df.keys()[0]] = df[df.keys()[0]].replace(id_to_name)        

    # deal with cols
    if use_col:
        df = df.rename(id_to_name)
        
    # dave results
    save_file = data_file.split("/")[-1]
    df.to_csv(f"{output_dir}/{save_file}", index=False)

    # close log file
    log_data.close()



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
        help="path to the output folder",
    )

    # check inputs
    args = parser.parse_args()
    check, config = check_config(args)
    if check:

        # extract configuration
        input_file = config["input"]
        output_dir = config["output"]

        # init work folder
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        oups = run(input_file, output_dir)
        if oups:
            print(oups)
