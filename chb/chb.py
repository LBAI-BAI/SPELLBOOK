import pandas as pd
from Ensembl_converter import EnsemblConverter
from tqdm import tqdm
import pickle



def extract_gene_to_module(input_file, chaussabel_file):
    """ """

    # load chaussabel
    df_chb = pd.read_csv(chaussabel_file)
    gene_to_module = {}
    for index, row in df_chb.iterrows():
        gene_to_module[row['Gene']] = row['Module']

    # load input
    df_gene = pd.read_parquet(input_file)

    # convert genes to module
    old_to_new = {}
    var_list = list(df_gene.keys())
    for i in tqdm(range(len(var_list))):
        v = var_list[i]
        if v in gene_to_module:
            old_to_new[v] = gene_to_module[v]
        else:
            converter = EnsemblConverter()
            ensembl_ids = [v]
            result = converter.convert_ids(ensembl_ids)
            symbol = result['Symbol'][0]
            if symbol in gene_to_module:
                old_to_new[v] = gene_to_module[symbol]

    # save results
    with open('gene_to_module.pickle', 'wb') as handle:
        pickle.dump(old_to_new, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_module_to_gene(gene_to_module:dict):
    """ """

    module_to_gene = {}
    for gene in gene_to_module:
        module = gene_to_module[gene]
        if module not in module_to_gene:
            module_to_gene[module] = [gene]
        else:
            module_to_gene[module].append(gene)

    # save results
    with open('module_to_gene.pickle', 'wb') as handle:
        pickle.dump(module_to_gene, handle, protocol=pickle.HIGHEST_PROTOCOL)
    


def run(input_file, output_file, chaussabel_file, recompute_gene_to_module):
    """ """

    # parameters
    gene_to_module_file = "gene_to_module.pickle"

    # load input
    df_gene = pd.read_parquet(input_file)

    # compute gene to module
    if(recompute_gene_to_module):
        extract_gene_to_module(input_file, chaussabel_file)

    # load gene to module
    with open(gene_to_module_file, 'rb') as handle:
        gene_to_module = pickle.load(handle)

    # save module to gene
    get_module_to_gene(gene_to_module)


if __name__ == "__main__":

    # parameters
    input_file = "/home/bran/Workspace/papse/data/count_rnaseq.parquet"
    output_file = ""
    chb_file = "chaussabel.csv"

    run(input_file, output_file, chb_file, False)

