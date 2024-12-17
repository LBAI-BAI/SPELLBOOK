import pandas as pd
from scipy.spatial.distance import cdist
import argparse
import os


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
    """Run appariement using a data file that must conatins ID, SEX, AGE and LABEL column
    Work for exactly 2 groups, consider the group with less patient as case and the other one
    as control.

    Args:
        - data_file (str) : path to the input data file
        - output_dit (str) : path to the output dir

    Returns:
        (str) : description of error if something happen, else None

    """

    # load data
    df = pd.read_csv(data_file)

    # check variables
    missing_var = []
    mandatory_var_list = ["ID", "SEX", "AGE", "LABEL"]
    for var in mandatory_var_list:
        if var not in list(df.keys()):
            missing_var.append(var)
    if len(missing_var) > 0:
        return f"Can't perfrom appariement without variables {missing_var} in input file"

    # get list of group
    group_list = []
    for k in list(df['LABEL']):
        if k not in group_list:
            group_list.append(k)

    # exit if less or more than 2 group
    if len(group_list) != 2:
        return f"Can't perform appariement for anything else than exactly 2 groups, {len(group_list)} found"

    # group separation - consider cas as the group with the less number of patients
    grp1 = df[df['LABEL'] == group_list[0]]
    grp2 = df[df['LABEL'] == group_list[1]]
    if grp1.shape[0] > grp2.shape[0]:
        cas = grp2
        controle = grp1
        cas_name = group_list[1]
        controle_name = group_list[0]
    else:
        cas = grp1
        controle = grp2
        cas_name = group_list[0]
        controle_name = group_list[1]

    # Init appariements
    matches = []

    # Pour chaque patient dans "Cas"
    for i, row_cas in cas.iterrows():
        
        # Filtrer les contrôles par sexe correspondant
        controle_same_sex = controle[controle['SEX'] == row_cas['SEX']]
    
        # Calculer la distance d'âge
        if not controle_same_sex.empty:
            controle_same_sex['AGE_DIFF'] = abs(controle_same_sex['AGE'] - row_cas['AGE'])
        
            # Trouver le contrôle avec la différence d'âge minimale
            best_match = controle_same_sex.loc[controle_same_sex['AGE_DIFF'].idxmin()]
        
            # Ajouter l'appariement
            matches.append({
                f"{cas_name}_ID": row_cas['ID'],
                f"{controle_name}_ID": best_match['ID'],
                f"{cas_name}_Age": row_cas['AGE'],
                f"{controle_name}_Age": best_match['AGE'],
                "Sex": row_cas['SEX']
            })
        
            # Supprimer ce contrôle pour éviter d'apparaître plusieurs fois
            controle = controle[controle['ID'] != best_match['ID']]

    # Résultat des appariements
    matches_df = pd.DataFrame(matches)

    # save
    matches_df.to_csv(f"{output_dir}/match.csv", index=False)
    




if __name__ == "__main__":


    # Exemple de jeu de données
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

