import argparse
import numpy as np
import pandas as pd
import shutil
import xgboost as xgbt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import shutil
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

"""
invoke an xgboosted tree !
"""


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


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing values and rename variables

    Args:
        df (pd.DataFrame) : a pandas dataframe containing the data, first columns should be the ID ans last columns the LABEL

    Returns:
        df (pd.DataFrame) : a pandas dataframe withour missing values, first and last columns renamed 'ID' and 'LABEL'

    """

    # Replace Missing & NA values by np.nan, problem when all scalar are float
    try:
        df = df.replace(
            {
                "MISSING": np.nan,
                "NA": np.nan,
                "N/A": np.nan,
                "nan": np.nan,
                "NaN": np.nan,
                "": np.nan,
                " ": np.nan,
            }
        )
    except:
        pass

    # drop missing values
    df = df.dropna()

    # extract features
    features = list(df.keys())

    # rename first and last features
    df = df.rename(columns={features[0]: "ID", features[-1]: "LABEL"})

    # return dataframe
    return df


def run_hyperparametrisation(input_file: str, work_folder: str):
    """Perform hyperparametrisation of the model, wite best conf in a file

    Args:
        input_file (str) : path to the data file
        work_folder (str) : path to the output folder

    """

    # clean & prepare output folde
    if not os.path.isdir(work_folder + "/xgb_log"):
        os.mkdir(work_folder + "/xgb_log")
    else:
        shutil.rmtree(work_folder + "/xgb_log")
        os.mkdir(work_folder + "/xgb_log")

    # load file
    df = pd.read_csv(input_file)

    # preprocess file
    df = preprocess(df)

    # encode class label
    cmpt_class = 0
    old_label_to_encode = {}
    for y in list(df["LABEL"]):
        if y not in old_label_to_encode.keys():
            old_label_to_encode[y] = cmpt_class
            cmpt_class += 1
    df = df.replace(old_label_to_encode)

    # extract features
    features = [f for f in df.columns if f not in ["ID", "LABEL"]]

    # prepare dataset
    X = df[features].values
    Y = df["LABEL"].values.ravel()

    # split into train & validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=42
    )

    # detrmine if its a binary classification case or not
    if len(set(Y)) == 2:
        objective_function = "binary:logistic"
        scoring_system = "roc_auc"
    else:
        objective_function = "multi:softprob"
        scoring_system = "accuracy"

    # init xgb model
    xgb_model = xgbt.XGBClassifier(objective=objective_function)

    # define parmas to explore
    params = {
        "eta": np.arange(0.1, 0.26, 0.05),
        "min_child_weight": np.arange(1, 5, 0.5).tolist(),
        "gamma": np.arange(1, 10, 1).tolist(),
        "alpha": np.arange(1, 10, 3).tolist(),
        "learning_rate": np.arange(0.1, 1, 0.2).tolist(),
        "subsample": np.arange(0.5, 1.0, 0.2).tolist(),
        "colsample_bytree": np.arange(0.5, 1.0, 0.2).tolist(),
        "n_estimators": [2, 3, 5, 10],
        "max_depth": [2, 3, 5],
    }

    # init machin chose
    skf = StratifiedKFold(n_splits=2, shuffle=True)

    # init grid search
    grid = GridSearchCV(
        xgb_model,
        param_grid=params,
        scoring=scoring_system,
        n_jobs=-1,
        cv=skf.split(X, Y),
        refit="accuracy_score",
        verbose=1,
    )

    # Run GirdSearch
    print("[XGB] => Exploring the grid, can take a while ...")
    grid.fit(X, Y)

    # Save results (best params)
    best_pars = grid.best_params_
    output_data = open(f"{work_folder}/xgb_log/optimal_parameters.csv", "w")
    output_data.write("PARAM,VALUE\n")
    for k in best_pars.keys():
        output_data.write(str(k) + "," + str(best_pars[k]) + "\n")
    output_data.close()
    output_data.close()


def run_xgb_classifier(input_file: str, work_folder: str):
    """Train and run xgb with optimal parameters

    Args:
        input_file (str) : path to the data file
        work_folder (str) : path to the output folder
    """

    # parameters
    params = {}
    param_file = f"{work_folder}/xgb_log/optimal_parameters.csv"

    # clean & prepare output folde
    if not os.path.isdir(work_folder + "/xgb_log"):
        os.mkdir(work_folder + "/xgb_log")

    # load file
    df = pd.read_csv(input_file)

    # preprocess file
    df = preprocess(df)

    # encode class label
    cmpt_class = 0
    old_label_to_encode = {}
    for y in list(df["LABEL"]):
        if y not in old_label_to_encode.keys():
            old_label_to_encode[y] = cmpt_class
            cmpt_class += 1
    df = df.replace(old_label_to_encode)

    # extract features
    features = [f for f in df.columns if f not in ["ID", "LABEL"]]

    # prepare dataset
    X = df[features].values
    Y = df["LABEL"].values.ravel()

    # split into train & validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=42
    )

    # detrmine if its a binary classification case or not
    if len(set(Y)) == 2:
        objective_function = "binary:logistic"
        scoring_system = "roc_auc"
    else:
        objective_function = "multi:softprob"
        scoring_system = "accuracy"

    # look for hyperparmaeters
    if os.path.isfile(param_file):
        df = pd.read_csv(param_file)
        for index, row in df.iterrows():
            p = row["PARAM"]
            v = row["VALUE"]
            params[p] = v
    else:
        params = {
            "eta": 0.26,
            "min_child_weight": 1,
            "gamma": 2,
            "alpha": 3,
            "learning_rate": 0.3,
            "subsample": 0.5,
            "colsample_bytree": 0.5,
            "n_estimators": 3,
            "max_depth": 3,
        }

    # initialize model
    xgb_model = xgbt.XGBClassifier(
        objective=objective_function,
        colsample_bytree=params["colsample_bytree"],
        eta=params["eta"],
        max_depth=int(params["max_depth"]),
        gamma=params["gamma"],
        alpha=params["alpha"],
        learning_rate=params["learning_rate"],
        n_estimators=int(params["n_estimators"]),
    )

    # fit model
    xgb_model.fit(X_train, y_train)

    # make predictions
    preds = xgb_model.predict(X_test)

    # compute RMSE (kind of precision metrics)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # compute ACC
    round_preds = [round(value) for value in preds]
    accuracy = accuracy_score(y_test, round_preds)

    # save performances
    perf_file = open(f"{work_folder}/xgb_evaluation.log", "w")
    perf_file.write(f"RMSE\t{rmse}\n")
    perf_file.write(f"ACC\t{accuracy}\n")
    perf_file.close()

    # save feature importance
    feature_to_importance = xgb_model.get_booster().get_score(importance_type="gain")
    save_feature_file = open(f"{work_folder}/important_features.csv", "w")
    save_feature_file.write("FEATURE,GAIN\n")
    for key in feature_to_importance.keys():
        save_feature_file.write(str(key) + "," + str(feature_to_importance[key]) + "\n")
    save_feature_file.close()

    # save confusion matrix for train dataset
    ConfusionMatrixDisplay.from_predictions(y_test, round_preds)
    plt.tight_layout()
    plt.savefig(f"{work_folder}/confusion_matrix.png", dpi=150)
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

        # run hyperparametrisation
        run_hyperparametrisation(input_file, output_dir)

        # run xgboost with best params
        run_xgb_classifier(input_file, output_dir)
