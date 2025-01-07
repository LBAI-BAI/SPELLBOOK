import os
import argparse
import pandas as pd
import phenograph

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
    if arguments.output:
        config["output"] = arguments.output
        if not os.path.isdir(config['output']):
            os.mkdir(config['output'])
    else:
        check = False
        
    # add input file
    if arguments.input:
        config["input"] = arguments.input
        if not os.path.isfile(config['input']):
            check = False
    else:
        check = False

    # return config
    return check, config


def run(input_file, output_folder):
    """Run phenograph clustering
    Create 3 files : clusters.csv, graph.txt and log.txt
    N.B : Q : ratio of intra-cluster modularity score to inter-cluster modularity score (The Louvain algorithm tries to optimize it)

    Args:
        - input_file (str) : path to input data file
        - output_folder (str) : path to output folder

    """

    # parameters
    k = 30

    # load data
    df = pd.read_csv(input_file)
    if 'LABEL' in list(df.keys()):
        df = df.drop(columns=['LABEL'])
    if 'ID' in list(df.keys()):
        df.index = df['ID']
        df = df.drop(columns=['ID'])

    # run phenograph
    communities, graph, Q = phenograph.cluster(df,k=k)

    # save clusters
    cluster_data = open(f"{output_folder}/clusters.csv", "w")
    cluster_data.write("ID,CLUSTER\n")
    cmpt = 0
    for i in list(df.index):
        cluster_data.write(f"{i},{communities[cmpt]}\n")
        cmpt+=1
    cluster_data.close()

    # save graph
    graph_file = open(f"{output_folder}/graph.txt", "w")
    graph_file.write(str(graph))
    graph_file.close()

    # save logs
    log_file = open(f"{output_folder}/log.txt", "w")
    log_file.write(f"k = {k}\n")
    log_file.write(f"Q = {Q}\n")
    log_file.close()

    


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
