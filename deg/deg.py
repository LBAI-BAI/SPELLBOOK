import DEGA
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def check_config(arguments):
    """Check that provided files exists"""

    # init configuration
    config = {}

    # check that provided gene count exist
    if not os.path.isfile(arguments.gene_counts):
        print(f"[!] Can't find {arguments.gene_counts}")
        return False, {}
    else:
        config["count"] = arguments.gene_counts

    # check that provided gene pheno exist
    if not os.path.isfile(arguments.gene_pheno):
        print(f"[!] Can't find {arguments.gene_pheno}")
        return False, {}
    else:
        config["pheno"] = arguments.gene_pheno

    # add output folder and label
    config["output"] = arguments.output
    config["label"] = arguments.label

    # return config
    return True, config


def prepare(gene_count_file: str, gene_pheno_file: str, output_dir: str) -> None:
    """Prepare files for the Analysis, create output dir if not exist

    Args:
        gene_count_file (str) : path to file containing the gene counts
        gene_pheno_file (str) : path to file containing the gene pheno
        output_dir (str) : path to the output dir

    """

    # create output directory if not exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        log_data = open(f"{output_dir}/analysis.log", "w")
        log_data.write(f"[PREPARATION] => create {output_dir}\n")
    else:
        log_data = open(f"{output_dir}/analysis.log", "w")

    # Deal with gene counts
    # convert Gene Count to integer
    geneCounts = pd.read_csv(gene_count_file, index_col=0)
    geneCounts = geneCounts.astype(int)
    log_data.write("[PREPARATION] => gene counts values set to int\n")

    # TODO : identify case where transposition is needed
    need_transposition = False

    # case 1 - look for something like gene names in columns

    # transpose data if needed
    if need_transposition:

        # TODO : perform transposition
        log_data.write("[PREPARATION] => gene counts have been transposed\n")

    # Deal with gene pheno
    # Cast index to string
    phenotypeData = pd.read_csv(gene_pheno_file, index_col=0)
    phenotypeData.index = phenotypeData.index.map(str)
    log_data.write("[PREPARATION] => reindex pheno to match counts\n")

    # filter and sort geneCounts columns on the basis of phenotypeData index
    # (they have to be in the same order)
    geneCounts = geneCounts[phenotypeData.index]
    log_data.write(
        "[PREPARATION] => filter and sort counts columns based on pheno data\n"
    )

    # save gene and pheno file in output dur
    geneCounts.to_csv(f"{output_dir}/gene_counts.csv")
    phenotypeData.to_csv(f"{output_dir}/gene_pheno.csv")
    log_data.write(f"[PREPARATION] => pheno and counts save in {output_dir} \n")

    # close log files
    log_data.close()


def run(output_dir: str, label: str):
    """ """

    # load data
    geneCounts = pd.read_csv(f"{output_dir}/gene_counts.csv", index_col=0)
    phenotypeData = pd.read_csv(f"{output_dir}/gene_pheno.csv", index_col=0)

    # Analysis
    dega = DEGA.dataset(geneCounts, phenotypeData, designFormula=label)
    dega.analyse()

    # save results
    upregulated = open(f"{output_dir}/upregulated.csv", "w")
    upregulated.write("GENE\n")
    for g in dega.upregulatedGenes:
        upregulated.write(f"{g}\n")
    upregulated.close()

    downregulated = open(f"{output_dir}/downregulated.csv", "w")
    downregulated.write("GENE\n")
    for g in dega.downregulatedGenes:
        downregulated.write(f"{g}\n")
    downregulated.close()

    # plot
    dega.plotMA(lfcThreshold=1)
    plt.savefig(f"{output_dir}/fig1.png")
    plt.close()

    dega.plotVolcano(
        lfcThreshold=1,
        labels=False,
    )
    plt.savefig(f"{output_dir}/fig2.png")
    plt.close()


if __name__ == "__main__":

    # parameters
    geneCountsFile = "https://raw.githubusercontent.com/LucaMenestrina/DEGA/main/validation/bottomly_counts.csv"  # "bottomly_counts.csv"
    phenotypeDataFile = "https://raw.githubusercontent.com/LucaMenestrina/DEGA/main/validation/bottomly_phenotypes.csv"  # "bottomly_phenotypes.csv"

    # parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-gc",
        "--gene_counts",
        type=str,
        default="",
        help="path to the gene counts data file",
    )
    parser.add_argument(
        "-gp", "--gene_pheno", type=str, default="", help="path to the pheno data file"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="", help="path to the output folder"
    )
    parser.add_argument(
        "-l", "--label", type=str, default="LABEL", help="feature to use in pheno"
    )
    args = parser.parse_args()

    # check inputs
    check, config = check_config(args)
    if check:

        # extract configuration
        gene_counts = config["count"]
        gene_pheno = config["pheno"]
        output_dir = config["output"]
        label = config["label"]

        # prepare files
        prepare(gene_counts, gene_pheno, output_dir)

        # run
        run(output_dir, label)
