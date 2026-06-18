import ast

import numpy as np
import pandas as pd
from tf_metabolism.metabolic_model import GPR_manipulation


def calculate_rank_based_expression_score(gene_expression_data, threshold=0.25):
    column = gene_expression_data.columns[0]
    gene_expression = pd.Series(gene_expression_data[column])
    rank_based_gene_expression = gene_expression.rank()
    max_rank = rank_based_gene_expression.max()
    min_rank = rank_based_gene_expression.min()

    rank_based_gene_expression = rank_based_gene_expression - min_rank + 1

    max_rank = rank_based_gene_expression.max()
    min_rank = rank_based_gene_expression.min()

    rank_based_gene_expression = rank_based_gene_expression / (max_rank * threshold)
    rank_based_gene_expression = np.log2(rank_based_gene_expression)
    rank_based_gene_expression = rank_based_gene_expression
    rank_based_gene_expression.index = rank_based_gene_expression.index.map(str)
    rank_based_gene_expression = dict(rank_based_gene_expression)

    return rank_based_gene_expression


def calculate_original_expression_score(gene_expression_data, average_expression_value):
    column = gene_expression_data.columns[0]
    gene_expression = pd.Series(gene_expression_data[column])
    gene_expression = gene_expression + 1
    original_gene_expression = gene_expression / average_expression_value
    original_gene_expression = np.log2(original_gene_expression)
    original_gene_expression = original_gene_expression
    original_gene_expression.index = original_gene_expression.index.map(str)
    original_gene_expression = dict(original_gene_expression)
    return original_gene_expression


def convert_gene_id_to_score(ls, expression_score_info_dic):
    for each_item in ls:
        if type(each_item) == list:
            return convert_gene_id_to_score(each_item, expression_score_info_dic)

    scores = []
    boolean_operations = []
    for i in range(len(ls)):
        each_item = ls[i]
        if each_item != 'AND' and each_item != 'and' and each_item != 'OR' and each_item != 'or':
            if each_item in expression_score_info_dic:
                expression_value = expression_score_info_dic[each_item]
                scores.append(expression_value)
            else:
                if type(each_item) == float:
                    expression_value = float(each_item)
                    scores.append(expression_value)
        else:
            boolean_operations.append(each_item)
    if scores == []:
        return ls, 0.0
    boolean_operations = list(set(boolean_operations))
    if len(boolean_operations) == 1:
        if 'AND' == boolean_operations[0] or 'and' == boolean_operations[0]:
            return ls, np.min(scores)
        elif 'OR' == boolean_operations[0] or 'or' == boolean_operations[0]:
            return ls, np.max(scores)
    else:
        return ls, np.max(scores)


def GPR_score_calculation(GPR_list, expression_score_info_dic):
    final_score = 0.0
    while True:
        GPR_string = str(GPR_list)
        converted_list, score = convert_gene_id_to_score(GPR_list, expression_score_info_dic)
        converted_list_string = str(converted_list)
        new_GPR_string = GPR_string.replace(converted_list_string, str(score))
        new_GPR_list = ast.literal_eval(new_GPR_string)
        GPR_list = new_GPR_list
        if type(GPR_list) != list:
            final_score = GPR_list
            break
    return final_score


def reaction_score_calculation(cobra_model, expression_score):
    expression_socre_string = {}
    for gene_id in expression_score:
        expression_socre_string[str(gene_id)] = expression_score[gene_id]

    reaction_weights = {}
    for each_reaction in cobra_model.reactions:
        reaction_weights[each_reaction.id] = 0.0
        GPR_association = each_reaction.gene_reaction_rule
        genes = [gene.id for gene in each_reaction.genes]
        genes = list(set(genes) & set(expression_socre_string.keys()))
        if len(genes) > 1:
            GPR_list = GPR_manipulation.convert_string_GPR_to_list_GPR(GPR_association)
            expression = GPR_score_calculation(GPR_list, expression_socre_string)
            reaction_weights[each_reaction.id] = expression
        elif len(genes) == 1:
            gene_id = genes[0].strip()
            if gene_id in expression_socre_string:
                expression = expression_socre_string[gene_id]
                reaction_weights[each_reaction.id] = expression
    return reaction_weights
