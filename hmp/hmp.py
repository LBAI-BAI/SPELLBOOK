import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

    # load & sort data
    df = pd.read_csv(input_file)
    df = df.sort_values(by='LABEL')

    # parse title
    title = input_file.split("/")[-1].split('.')[0]

    # craft xlabel
    label_to_spot = {}
    for label in list(set(df['LABEL'])):
        spot = int(len(list(df[df['LABEL'] == label]['LABEL'])) / 2)
        label_to_spot[label] = spot
    x_label = []
    current_label = "zogzog"
    cmpt = 0
    for x in list(df['LABEL']):
        if x != current_label:
            x_label.append('|')
            current_label = x
            cmpt = 0
        elif cmpt == label_to_spot[x]:
            x_label.append(current_label) 
        else:
            x_label.append('')
        cmpt+=1
    

    # transpose and drop label column
    heatmap_data = df.drop(columns=['LABEL']).T

    # craft the heatmap
    plt.figure(figsize=(12, 6))  # Adjust figure size
    sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={'label': 'Feature Value'}, yticklabels=True, xticklabels=x_label, annot=False)
    plt.title(f"Heatmap from {title}")
    plt.ylabel("Features")
    plt.xticks(rotation=360, ha='center')

    # save heatmap
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{title}.png")
    plt.close()





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


