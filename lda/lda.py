import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import joblib


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
    """Run a linear discrimnant analysis, save model, fitted data and features contriubution in output folder.
    If input_file contains 3 or 4 categories, generate respectively a 2d or 3d representation space.

    Args:
        - input_file (str) : path to the input data file, label columns must be labeled 'LABEL'
        - output_dir (str) : path to the output directory
    
    """

    # parameters
    title = "AFD on " + str(input_file.split("/")[-1]).replace(".csv", "")
    max_feature_to_display = 50

    # load data
    df = pd.read_csv(input_file)

    # Extract data
    df2 = df.drop(columns=["LABEL", "ID"])
    X = []
    for index, row in df2.iterrows():
        X.append(row.values)
        var_list = list(row.keys())
    X = np.array(X)

    # count nb of category & extract target names
    target_names = []
    target_to_code = {}
    code = 0
    for i in set(list(df["LABEL"])):
        target_to_code[i] = code
        code +=1
    target_names = list(target_to_code.keys())
    nb_class = len(target_names)

    # extract y
    y = np.array(df["LABEL"].replace(target_to_code))

    # Perform LDA
    lda = LinearDiscriminantAnalysis(n_components=nb_class - 1)
    X_r = lda.fit(X, y).transform(X)

    # save model
    joblib.dump(lda, f"{output_folder}/model.pkl")

    # save data embedding
    vector_list = []
    label_list = list(df['LABEL'])
    cmpt = 0
    for x in X_r:

        # craft vector
        vector = {}
        for i in range(nb_class -1):
            vector[f"Axis {i+1}"] = x[i] 
        vector['LABEL'] = label_list[cmpt]
        cmpt+=1

        # add to vector list
        vector_list.append(vector)

    # craft and save fitted data
    df_fitted = pd.DataFrame(vector_list)
    df_fitted.to_csv(f"{output_folder}/data_fitted.csv", index=False)

    # extract feature information
    vector_list = []
    var_list = list(df2.keys())
    cmpt = 0
    for x in lda.scalings_:

        # craft vector
        vector = {'VARIABLE':var_list[cmpt]}
        for i in range(nb_class -1):
            vector[f"Axis {i+1}"] = x[i] 
        cmpt+=1

        # add to vector list
        vector_list.append(vector)

    # craft and save feature data
    df_feature = pd.DataFrame(vector_list)
    df_feature.to_csv(f"{output_folder}/features_contribution.csv", index=False)

    # craft feature barplot
    col_list = list(df_feature.keys())
    col_list.remove('VARIABLE')

    for col in col_list:
        df_tmp = df_feature[['VARIABLE', col]]
        df_tmp[col] = abs(df_tmp[col])
        df_tmp = df_tmp.sort_values(by=col, ascending=False)
        df_tmp = df_tmp.head(max_feature_to_display)
        df_tmp = df_tmp.iloc[::-1].reset_index(drop=True)

        # Plot the barplot
        plt.figure(figsize=(8, 5))
        plt.barh(df_tmp['VARIABLE'], df_tmp[col], color='skyblue', edgecolor='black')

        # Customize the plot
        plt.xlabel('Absolute contribution', fontsize=12)
        plt.ylabel('Variables', fontsize=12)
        plt.title(f"Feature Contribution for {col}", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.savefig(f"{output_folder}/feature_contribution_to_{col.replace(' ', '_')}.png")
        plt.show()
        plt.close()
        
    # craft 2d figure if relevant
    if nb_class == 3:

        plt.figure(figsize=(8, 6))
        colors = ['red', 'green', 'blue']
        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            plt.scatter(X_r[y == i, 0], X_r[y == i, 1], alpha=0.7, color=color, label=target_name)

        plt.title(title)
        plt.xlabel("LDA-Axis 1")
        plt.ylabel("LDA-Axis 2")
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.grid()
        plt.savefig(f"{output_folder}/lda.png")
        plt.close()
        
    # Display 3d figure if relevant
    if nb_class == 4:

        # Plot the LDA-transformed data
        fig = plt.figure(figsize=(10, 7))
        colors = ['red', 'green', 'blue', 'orange']
        ax = fig.add_subplot(111, projection='3d')
        for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
            ax.scatter(X_r[y == i, 0], X_r[y == i, 1], X_r[y == i, 2], alpha=0.7, color=color, label=target_name)
        ax.set_title(title)
        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
        ax.set_zlabel("LD3")
        ax.legend()

        # Save rotating GIF
        frames = []
        for angle in range(0, 360, 2):  # Rotate in steps of 2 degrees
            ax.view_init(30, angle)
            plt.draw()

            # Save each frame
            filename = f"/tmp/frame_{angle}.png"
            plt.savefig(filename)
            frames.append(Image.open(filename))

        # Save as a GIF
        gif_path = f"{output_folder}/3d_lda.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # Frame duration in ms
            loop=0
        )
        


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

