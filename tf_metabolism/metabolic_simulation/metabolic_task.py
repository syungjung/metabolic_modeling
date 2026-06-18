import copy
import os
import parser
import random
import tempfile
import logging
import time

import numpy as np
import pandas as pd
from tf_metabolism.metabolic_model import model_editing
from tf_metabolism.metabolic_simulation import Simulator


def evaluate_metabolic_task(cobra_model, metabolic_task_file, task_constraints={}, is_metabolic_task_model=False):
    start = time.time()
    logging.info("Evaluating metabolic task")
    if not is_metabolic_task_model:
        cobra_model = model_editing.make_metabolic_task_model(copy.deepcopy(cobra_model), metabolic_task_file)
    original_model = copy.deepcopy(cobra_model)
    task_df = pd.read_csv(metabolic_task_file)
    task_df = task_df[['Task ID', 'Type', 'ID', 'Medium', 'Constraints', 'Expected value', 'Description']]
    task_info = {}
    task_success_info = {}
    task_flux_constraints = {}
    for each_row, each_df in task_df.iterrows():
        task_id = each_df.loc['Task ID']
        expected_value = each_df.loc['Expected value']
        task_success_info[task_id] = expected_value
        task_flux_constraints[task_id] = {}
        constraint_string = each_df.loc['Constraints']
        if not constraint_string:
            constraint_list = constraint_string.split(';')
            for each_constraint in constraint_list:
                sptlist = each_constraint[:-1].split('(')
                constraint_reaction = sptlist[0].strip()
                lb = float(sptlist[1].split('#')[0].strip())
                ub = float(sptlist[1].split('#')[1].strip())
                task_flux_constraints[task_id][constraint_reaction] = [lb, ub]

        if each_df.loc['Type'] == 'Metabolite':
            metabolite = each_df.loc['ID']
            target_reaction = 'DMTASK%s_%s' % (task_id, metabolite)
            task_info[task_id] = target_reaction
        else:
            target_reaction = each_df.loc['ID']
            task_info[task_id] = target_reaction

    task_result_list = []
    solution_status_list = []
    solution_value_list = []
    model_reactions = []
    for each_reaction in cobra_model.reactions:
        model_reactions.append(each_reaction.id)
        if each_reaction.boundary:
            each_reaction.lower_bound = 0.0
            each_reaction.upper_bound = 1000.0

    task_id_list = task_info.keys()
    task_id_list = sorted(task_id_list)
    for task_id in task_id_list:
        cobra_model = copy.deepcopy(original_model)
        temp_flux_list = []
        each_target_reaction = task_info[task_id]
        if each_target_reaction in model_reactions:
            medium_reaction = 'TASK_MEDIUM_%s' % task_id
            medium_reaction_index = cobra_model.reactions.index(medium_reaction)
            cobra_model.reactions[medium_reaction_index].lower_bound = -1.0
            target_reaction_index = cobra_model.reactions.index(each_target_reaction)
            if cobra_model.reactions[target_reaction_index].reversibility:
                cobra_model.reactions[target_reaction_index].lower_bound = -1000.0
            cobra_model.reactions[target_reaction_index].upper_bound = 1000.0
            flux_constraints = task_flux_constraints[task_id]
            for each_key in task_constraints:
                flux_constraints[each_key] = task_constraints[each_key]

            obj = Simulator.Simulator()
            obj.load_cobra_model(cobra_model)
            solution_status_string = ''
            solution_status_max, objective_value_max, flux_distribution_max = obj.run_FBA(
                new_objective=each_target_reaction, flux_constraints=flux_constraints, mode='max')
            if solution_status_max == 2:
                temp_flux_list.append(abs(flux_distribution_max[each_target_reaction]))
                solution_status_string = 'Maximzation (optimal)'
            else:
                solution_status_string = 'Maximzation (not optimal, status %s)' % solution_status_max
            if cobra_model.reactions[target_reaction_index].reversibility:
                solution_status_min, objective_value_min, flux_distribution_min = obj.run_FBA(
                    new_objective=each_target_reaction, flux_constraints=flux_constraints, mode='min')
                if solution_status_min == 2:
                    temp_flux_list.append(abs(flux_distribution_min[each_target_reaction]))
                    solution_status_string = solution_status_string + ' Minimization (optimal)'
                else:
                    solution_status_string = 'Minimization (not optimal, status %s)' % solution_status_max
            solution_status_list.append(solution_status_string)
            cobra_model.reactions[medium_reaction_index].lower_bound = 0.0
            cobra_model.reactions[target_reaction_index].lower_bound = 0.0
            cobra_model.reactions[target_reaction_index].upper_bound = 0.0
            if len(temp_flux_list) == 0:
                task_result_list.append('FAILED')
                solution_value_list.append('NA')
            else:
                max_flux = max(temp_flux_list)
                expected_value = task_success_info[task_id]
                formula = '%s%s' % (max_flux, expected_value)
                code = parser.expr(formula).compile()
                solution_value_list.append(formula)
                if eval(code):
                    task_result_list.append('PASSED')
                else:
                    task_result_list.append('FAILED')
        else:
            task_result_list.append('FAILED - not exist in the model')
            solution_value_list.append('NA')
            solution_status_list.append('NA')

    task_df['Task result'] = pd.Series(task_result_list, index=task_df.index)
    task_df['Result equation'] = pd.Series(solution_value_list, index=task_df.index)
    task_df['Solution status'] = pd.Series(solution_status_list, index=task_df.index)
    logging.info(time.strftime("Elapsed time %H:%M:%S", time.gmtime(time.time() - start)))
    return task_df


def calculate_atp_production_rate(cobra_model, atp_demand_reaction, oxygen_uptake_reaction, carbon_source_list_file):
    start = time.time()
    logging.info("Calculating ATP production rate")
    metabolite_info = {}
    for each_metabolite in cobra_model.metabolites:
        metabolite_info[each_metabolite.id[:-2]] = each_metabolite.name

    carbon_sources = []
    with open(carbon_source_list_file, 'r') as fp:
        for line in fp:
            carbon_sources.append(line.strip())

    carbon_source_information = {}
    carbon_source_flux_constraints = {}
    for each_reaction in cobra_model.reactions:
        if each_reaction.boundary and each_reaction.id[0:3] == 'EX_':
            reactants = each_reaction.reactants
            if reactants[0].id[:-2] in carbon_sources:
                carbon_source_information[reactants[0].id[:-2]] = each_reaction.id
                carbon_source_flux_constraints[each_reaction.id] = [0.0, 1000.0]

    obj = Simulator.Simulator()
    obj.load_cobra_model(cobra_model)
    results = {}
    for oxygen_availability in ('Aerobic condition', 'Anaerobic condition'):
        flux_constraints = carbon_source_flux_constraints
        oxygen_lower_bound = 0.0
        if oxygen_availability == 'Aerobic condition':
            oxygen_lower_bound = -1000.0
        else:
            oxygen_lower_bound = 0.0
        results[oxygen_availability] = {}
        flux_constraints[oxygen_uptake_reaction] = [oxygen_lower_bound, 1000.0]
        for each_carbon_source in carbon_source_information:
            target_reaction = carbon_source_information[each_carbon_source]
            flux_constraints[target_reaction] = [-1.0, 1000.0]
            solution_status, objective_value, flux_distribution = obj.run_FBA(
                new_objective=atp_demand_reaction, flux_constraints=flux_constraints)
            atp_production_rate = flux_distribution[atp_demand_reaction]
            result_key = '%s(%s)' % (each_carbon_source, metabolite_info[each_carbon_source])
            results[oxygen_availability][result_key] = atp_production_rate
            flux_constraints[target_reaction] = [0.0, 1000.0]

    atp_production_df = pd.DataFrame.from_dict(results)
    logging.info(time.strftime("Elapsed time %H:%M:%S", time.gmtime(time.time() - start)))
    return atp_production_df


def oxygen_response_simulation(cobra_model, biomass_reaction, oxygen_uptake_reaction):
    start = time.time()
    logging.info("Calculating flux distributions under different levels of oxygen")
    obj = Simulator.Simulator()
    obj.load_cobra_model(cobra_model)
    results = {}
    for each_oxygen_constraint in np.linspace(0.0, 1.0, 11):
        results['Oxygen_%s' % each_oxygen_constraint] = {}
        flux_constraints = {}
        flux_constraints[oxygen_uptake_reaction] = [-each_oxygen_constraint, 1000.0]
        solution_status, objective_value, flux_distribution = obj.run_FBA(
            new_objective=biomass_reaction, flux_constraints=flux_constraints,
            internal_flux_minimization=True)
        for each_reaction in cobra_model.reactions:
            results['Oxygen_%s' % each_oxygen_constraint][each_reaction.id] = flux_distribution[each_reaction.id]

    oxygen_response_result_df = pd.DataFrame.from_dict(results)
    logging.info(time.strftime("Elapsed time %H:%M:%S", time.gmtime(time.time() - start)))
    return oxygen_response_result_df


def essentiality_simulation(cobra_model, biomass_reaction, essential_gene_file, non_essential_gene_file, use_sampling=False):
    start = time.time()
    logging.info("Calculating gene essentiality")
    essential_genes = []
    non_essential_genes = []
    with open(essential_gene_file, 'r') as fp:
        for line in fp:
            essential_genes.append(line.strip())

    non_essential_genes = []
    with open(non_essential_gene_file, 'r') as fp:
        for line in fp:
            non_essential_genes.append(line.strip())

    metabolic_genes = [gene.id for gene in cobra_model.genes]
    essential_metabolic_genes = list(set(essential_genes) & set(metabolic_genes))
    if use_sampling:
        non_essential_metabolic_genes = list(set(non_essential_genes) & set(metabolic_genes))
        non_essential_metabolic_genes = random.sample(non_essential_metabolic_genes, len(essential_metabolic_genes))
    else:
        non_essential_metabolic_genes = list(set(non_essential_genes) & set(metabolic_genes))
    experimental_essentiality_information = {}
    for each_gene in essential_metabolic_genes:
        experimental_essentiality_information[each_gene] = False

    for each_gene in non_essential_metabolic_genes:
        experimental_essentiality_information[each_gene] = True

    target_reaction_information = {}
    for each_gene in cobra_model.genes:
        reactions = [reaction.id for reaction in each_gene.reactions]
        target_reaction_information[each_gene.id] = reactions

    obj = Simulator.Simulator()
    obj.load_cobra_model(cobra_model)
    solution_status, objective_value, flux_distribution = obj.run_FBA(new_objective=biomass_reaction)
    wild_growth = flux_distribution[biomass_reaction]
    tp_cnt = 0
    tn_cnt = 0
    fp_cnt = 0
    fn_cnt = 0
    essentiality_result_information = {}
    for each_gene in essential_metabolic_genes + non_essential_metabolic_genes:
        essentiality_result_information[each_gene] = {}
        target_reactions = target_reaction_information[each_gene]
        flux_constraints = {}
        for each_reaction in target_reactions:
            flux_constraints[each_reaction] = [0.0, 0.0]

        solution_status, objective_value, flux_distribution = obj.run_FBA(flux_constraints=flux_constraints)
        growth = flux_distribution[biomass_reaction]
        insilico_essentiality = True
        if growth < wild_growth * 0.05:
            insilico_essentiality = False
        experimental_essentiality = experimental_essentiality_information[each_gene]
        if experimental_essentiality == True:
            if insilico_essentiality == True:
                tp_cnt += 1
        if experimental_essentiality == False:
            if insilico_essentiality == False:
                tn_cnt += 1
        if experimental_essentiality == True:
            if insilico_essentiality == False:
                fn_cnt += 1
        if experimental_essentiality == False:
            if insilico_essentiality == True:
                fp_cnt += 1
            else:
                essentiality_result_information[each_gene]['reactions'] = target_reactions
                if not experimental_essentiality:
                    essentiality_result_information[each_gene]['experimental'] = 'Essential'
                else:
                    essentiality_result_information[each_gene]['experimental'] = 'Non-essential'
                if not insilico_essentiality:
                    essentiality_result_information[each_gene]['insilico'] = 'Essential'
                else:
                    essentiality_result_information[each_gene]['insilico'] = 'Non-essential'

    accuracy = (tp_cnt + tn_cnt) / float(tp_cnt + tn_cnt + fn_cnt + fp_cnt)
    specificity = tn_cnt / float(tn_cnt + fp_cnt)
    sensitivity = tp_cnt / float(tp_cnt + fn_cnt)
    logging.info(time.strftime("Elapsed time %H:%M:%S", time.gmtime(time.time() - start)))
    return (essentiality_result_information, accuracy, specificity, sensitivity)


def metabolic_task_generation(cobra_model, present_metabolites, essential_reactions, medium_file, metabolic_task_file):
    exchange_reaction_metabolite_information = {}
    for each_reaction in cobra_model.reactions:
        if each_reaction.boundary:
            reactant = each_reaction.reactants[0]
            exchange_reaction_metabolite_information[each_reaction.id] = reactant.id

    medium_information_list = []
    medium_information_string = ''
    with open(medium_file, 'r') as fp:
        for line in fp:
            sptlist = line.strip().split('\t')
            reaction_id = sptlist[0].strip()
            uptake_rate = sptlist[1].strip()
            if reaction_id in exchange_reaction_metabolite_information:
                reactant_id = exchange_reaction_metabolite_information[reaction_id]
                medium_information_list.append('%s(%s)' % (reactant_id, uptake_rate))

    medium_information_string = ';'.join(medium_information_list)
    temp = tempfile.NamedTemporaryFile(prefix='temp_metabolic_task_', suffix='.csv', dir='./', delete=True)
    temp_metabolic_task_file = temp.name
    temp.close()
    task_fp = open(temp_metabolic_task_file, 'w')
    with open(metabolic_task_file, 'r') as fp:
        line = fp.readline()
        print((line.strip()), file=task_fp)
        for line in fp:
            sptlist = line.strip().split(',')
            task_id = int(sptlist[0].strip())
            print((line.strip()), file=task_fp)

    for each_metabolite in present_metabolites:
        task_id += 1
        print(('%s,%s,%s,%s,%s,%s,%s' % (
            task_id, 'Metabolite', each_metabolite, medium_information_string, '', '>0.0', 'Essential metabolite')),
            file=task_fp)

    for each_reaction in essential_reactions:
        task_id += 1
        print(('%s,%s,%s,%s,%s,%s,%s' % (
            task_id, 'Reaction', each_reaction, medium_information_string, '', '>0.0', 'Essential reaction')),
            file=task_fp)

    task_fp.close()
    new_metabolic_task_df = pd.read_csv(temp_metabolic_task_file)
    os.remove(temp_metabolic_task_file)
    return new_metabolic_task_df
