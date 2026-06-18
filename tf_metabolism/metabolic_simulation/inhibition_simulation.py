import copy
import itertools

import pandas as pd
from tf_metabolism.metabolic_model import model_editing
from tf_metabolism.metabolic_simulation import metabolic_task


def gene_inhibition_simualtion(cobra_model, target_genes, target_num, metabolic_task_file):
    cobra_model = copy.deepcopy(cobra_model)
    metabolic_task_model = model_editing.make_metabolic_task_model(cobra_model, metabolic_task_file)

    targeting_result_info = {}
    cobra_model_genes = [gene.id for gene in cobra_model.genes]
    target_genes = list(set(target_genes) & set(cobra_model_genes))
    target_combination_list = itertools.combinations(target_genes, target_num)

    task_df = metabolic_task.evaluate_metabolic_task(metabolic_task_model, metabolic_task_file, {}, True)
    passed_metabolic_task_df = task_df[task_df['Task result'] == 'PASSED']
    failed_metabolic_task_df = task_df[task_df['Task result'] != 'PASSED']
    passed_metabolic_task_cnt = len(passed_metabolic_task_df)
    failed_metabolic_task_cnt = len(failed_metabolic_task_df)

    targeting_result_info['Basal condition'] = {}
    targeting_result_info['Basal condition']['Targets'] = 'N/A'
    targeting_result_info['Basal condition']['Target reactions'] = 'N/A'
    targeting_result_info['Basal condition']['No. of passed tasks'] = passed_metabolic_task_cnt
    targeting_result_info['Basal condition']['No. of failed tasks'] = failed_metabolic_task_cnt

    for each_target_set in target_combination_list:
        target_reactions = []
        for each_gene in each_target_set:
            for each_reaction in cobra_model.genes.get_by_id(each_gene).reactions:
                target_reactions.append(each_reaction.id)
        target_reactions = list(set(target_reactions))

        flux_constraints = {}
        for each_reaction in target_reactions:
            flux_constraints[each_reaction] = [0.0, 0.0]

        task_df = metabolic_task.evaluate_metabolic_task(metabolic_task_model, metabolic_task_file, flux_constraints,
                                                         True)

        passed_metabolic_task_df = task_df[task_df['Task result'] == 'PASSED']
        failed_metabolic_task_df = task_df[task_df['Task result'] != 'PASSED']
        passed_metabolic_task_cnt = len(passed_metabolic_task_df)
        failed_metabolic_task_cnt = len(failed_metabolic_task_df)

        key_string = ';'.join(each_target_set)
        key_string = 'Target:%s' % (key_string)
        targeting_result_info[key_string] = {}
        targeting_result_info[key_string]['Targets'] = key_string
        targeting_result_info[key_string]['Target reactions'] = ';'.join(target_reactions)
        targeting_result_info[key_string]['No. of passed tasks'] = passed_metabolic_task_cnt
        targeting_result_info[key_string]['No. of failed tasks'] = failed_metabolic_task_cnt

    result_df = pd.DataFrame(targeting_result_info)
    return result_df.T


def metabolite_inhibition_simualtion(cobra_model, target_metabolites, target_num, metabolic_task_file):
    cobra_model = copy.deepcopy(cobra_model)
    metabolic_task_model = model_editing.make_metabolic_task_model(cobra_model, metabolic_task_file)

    targeting_result_info = {}
    cobra_model_metabolites = [metabolite.id[:-2] for metabolite in cobra_model.metabolites]
    target_metabolites = list(set(target_metabolites) & set(cobra_model_metabolites))
    target_combination_list = itertools.combinations(target_metabolites, target_num)

    target_reaction_info = {}
    for each_reaction in cobra_model.reactions:
        if each_reaction.reversibility:
            metabolites = each_reaction.reactants + each_reaction.products
        else:
            metabolites = each_reaction.reactants

        str_metabolites = [each_metabolite.id[:-2] for each_metabolite in metabolites]
        str_metabolites = list(set(str_metabolites))
        for each_metabolite in str_metabolites:
            if each_metabolite not in target_reaction_info:
                target_reaction_info[each_metabolite] = [each_reaction.id]
            else:
                target_reaction_info[each_metabolite].append(each_reaction.id)

    task_df = metabolic_task.evaluate_metabolic_task(metabolic_task_model, metabolic_task_file, {}, True)
    passed_metabolic_task_df = task_df[task_df['Task result'] == 'PASSED']
    failed_metabolic_task_df = task_df[task_df['Task result'] != 'PASSED']
    passed_metabolic_task_cnt = len(passed_metabolic_task_df)
    failed_metabolic_task_cnt = len(failed_metabolic_task_df)

    targeting_result_info['Basal condition'] = {}
    targeting_result_info['Basal condition']['Targets'] = 'N/A'
    targeting_result_info['Basal condition']['Target reactions'] = 'N/A'
    targeting_result_info['Basal condition']['No. of passed tasks'] = passed_metabolic_task_cnt
    targeting_result_info['Basal condition']['No. of failed tasks'] = failed_metabolic_task_cnt

    for each_target_set in target_combination_list:
        target_reactions = []
        for each_metabolite in each_target_set:
            target_reactions += target_reaction_info[each_metabolite]

        target_reactions = list(set(target_reactions))

        flux_constraints = {}
        for each_reaction in target_reactions:
            flux_constraints[each_reaction] = [0.0, 0.0]

        task_df = metabolic_task.evaluate_metabolic_task(metabolic_task_model, metabolic_task_file, flux_constraints,
                                                         True)

        passed_metabolic_task_df = task_df[task_df['Task result'] == 'PASSED']
        failed_metabolic_task_df = task_df[task_df['Task result'] != 'PASSED']
        passed_metabolic_task_cnt = len(passed_metabolic_task_df)
        failed_metabolic_task_cnt = len(failed_metabolic_task_df)

        key_string = ';'.join(each_target_set)
        key_string = 'Target:%s' % (key_string)
        targeting_result_info[key_string] = {}
        targeting_result_info[key_string]['Targets'] = key_string
        targeting_result_info[key_string]['Target reactions'] = ';'.join(target_reactions)
        targeting_result_info[key_string]['No. of passed tasks'] = passed_metabolic_task_cnt
        targeting_result_info[key_string]['No. of failed tasks'] = failed_metabolic_task_cnt

    result_df = pd.DataFrame(targeting_result_info)
    return result_df.T
