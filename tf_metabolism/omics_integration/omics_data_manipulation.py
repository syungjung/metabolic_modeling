import os
import shutil

import pandas as pd


def omics_preprocessing(cobra_model, omics_data_file, use_normalization):
    cobra_model_genes = [gene.id for gene in cobra_model.genes]
    omics_df = pd.read_csv(omics_data_file, index_col=0)
    omics_df.index = omics_df.index.map(str)
    omics_df = omics_df[~omics_df.index.duplicated(keep='first')]
    omics_metabolic_genes = list(set(cobra_model_genes) & set(omics_df.index))
    omics_coverage = float(len(omics_metabolic_genes)) / float(len(cobra_model_genes))
    metabolic_expression_df = omics_df.loc[omics_metabolic_genes]

    if use_normalization:
        X = metabolic_expression_df
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        metabolic_expression_df = X_std * 100

    return metabolic_expression_df, omics_coverage


def split_expression_data_by_column(metabolic_expression_data_file, output_dir):
    try:
        shutil.rmtree(output_dir)
    except:
        pass

    try:
        os.mkdir(output_dir)
    except:
        pass

    metabolic_expression_df = pd.read_csv(metabolic_expression_data_file, index_col=0)
    for each_column in metabolic_expression_df.columns:
        metabolic_expression_df[each_column].to_csv(output_dir + '%s.csv' % (each_column), header=['Expression'],
                                                    index_label=['ID'])


def analyze_omics_data(metabolic_expression_data_file, output_dir):
    try:
        shutil.rmtree(output_dir)
    except:
        pass

    try:
        os.mkdir(output_dir)
    except:
        pass

    metabolic_expression_df = pd.read_csv(metabolic_expression_data_file, index_col=0)
    analysis_result_df = metabolic_expression_df.describe()
    analysis_result_df.to_csv(output_dir + 'expression_data_statistics.csv')
