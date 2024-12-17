import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
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

    # check action to perform
    if arguments.action not in ["train", "predict"]:
        print(
            f"[!] Can't regognize action {arguments.action}, should be either 'train' or 'predict'"
        )
        return False, {}
    else:
        config["action"] = arguments.action

    # add output folder
    config["output"] = arguments.output

    # add input file
    config["input"] = arguments.input

    # return config
    return True, config


def train(data_file: str, output_folder: str):
    """train a classifier (LogisticRegression) to predict data label from bag of words vectors

    Create classifier.pkl and vectorizer.pkl in output_folder, along with evaluation.log

    Args:
        data_file (str) : name of input file, sentence columns should be name 'SENTENCE' and label 'LABEL'
        output_folder (str) : name of the output folder, try to create it if not exist
    """

    # prepare output folder
    if not os.path.isdir(output_folder):
        try:
            os.mkdir(output_folder)
        except:
            print(f"[!] can't create {output_folder}")
            return 0

    # check input file
    if not os.path.isfile(data_file):
        print(f"[!] Can't find {data_file}")
        return 0

    # load data
    df = pd.read_csv(data_file)
    df = shuffle(df)

    # Split the data into training and testing sets
    train_data = df[: int(len(df) * 0.8)]
    test_data = df[int(len(df) * 0.8) :]

    # Convert text data into numerical features using bag of words
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data["SENTENCE"])
    X_test = vectorizer.transform(test_data["SENTENCE"])
    y_train = train_data["LABEL"].replace({-1: 0})
    y_test = test_data["LABEL"].replace({-1: 0})

    # Train a logistic regression classifier on the numerical features
    classifier = LogisticRegression(solver="lbfgs")
    classifier.fit(X_train, y_train)

    # Evaluate the performance of the classifier on the testing set
    accuracy = classifier.score(X_test, y_test)

    # Compute the predicted probabilities for the testing set
    y_prob = classifier.predict_proba(X_test)[:, 1]

    # Evaluate the performance of the classifier using AUC
    auc = roc_auc_score(y_test, y_prob)

    # Save the trained classifier and vectorizer to disk
    joblib.dump(classifier, f"{output_folder}/classifier.pkl")
    joblib.dump(vectorizer, f"{output_folder}/vectorizer.pkl")

    # display results
    print(f"[BOW][ACC] => {accuracy}")
    print(f"[BOW][AUC] => {auc}")

    # save data in log file
    log_data = open(f"{output_folder}/evaluation.log", "w")
    log_data.write(f"[BOW][ACC] => {accuracy}\n")
    log_data.write(f"[BOW][AUC] => {auc}\n")
    log_data.close()


def predict(input_data: str, model_folder: str):
    """Predict the label of input data and save prediction in model_folder

    Args:
        input_data (str) : either a sentence to label or a file containing sentence to label
        model_folder (str) : path to the folder containing vectorizer and classifer pkl files
    """

    # check that models exist
    if not os.path.isdir(model_folder):
        print(f"[!] Can't find folder {model_folder}")
        return 0
    elif not os.path.isfile(f"{model_folder}/classifier.pkl"):
        print(f"[!] Can't find classifier in {model_folder}")
        return 0
    elif not os.path.isfile(f"{model_folder}/vectorizer.pkl"):
        print(f"[!] Can't find vectorizer in {model_folder}")
        return 0

    # load the trained classifier and vectorizer from disk
    classifier = joblib.load(f"{model_folder}/classifier.pkl")
    vectorizer = joblib.load(f"{model_folder}/vectorizer.pkl")

    # see if input data is str or file
    sentence_to_y = {}
    if os.path.isfile(input_data):

        # load data
        sentence_list = list(pd.read_csv(input_data)["SENTENCE"])
        for s in sentence_list:

            # convert the input text into numerical features
            x = vectorizer.transform([input_data])

            # use the classifier to make a prediction
            y_prob = classifier.predict_proba(x)[:, 1]

            # update dict
            sentence_to_y[s] = y_prob[0]

    # deal with str input
    else:

        # convert the input text into numerical features
        x = vectorizer.transform([input_data])

        # use the classifier to make a prediction
        y_prob = classifier.predict_proba(x)[:, 1]

        # update dict
        sentence_to_y[input_data] = y_prob[0]

    # save prediction
    prediction_data = open(f"{model_folder}/prediction.csv", "w")
    prediction_data.write("SENTENCE,LABEL\n")
    for s in sentence_to_y:
        prediction_data.write(f"{s},{sentence_to_y[s]}\n")
    prediction_data.close()


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
    parser.add_argument(
        "-a",
        "--action",
        type=str,
        help="action to run, could be either 'train' or 'predict'",
    )

    # check inputs
    args = parser.parse_args()
    check, config = check_config(args)
    if check:

        # extract config
        action = config["action"]
        input_file = config["input"]
        output_folder = config["output"]

        # deal with train
        if action == "train":

            # run training
            train(input_file, output_folder)

        # deal with predict
        if action == "predict":

            # run prediction
            predict(input_file, output_folder)
