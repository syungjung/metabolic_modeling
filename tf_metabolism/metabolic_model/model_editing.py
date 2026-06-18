import json

import pandas as pd
from cobra import Reaction
from tf_metabolism import utils


def make_medium_model(cobra_model, medium_file):
    medium_info = {}
    with open(medium_file, 'r') as fp:
        for each_line in fp:
            sptlist = each_line.strip().split('\t')
            reaction_id = sptlist[0].strip()
            lb = float(sptlist[1].strip())
            ub = float(sptlist[2].strip())
            medium_info[reaction_id] = [lb, ub]
        fp.close()

    for each_reaction in cobra_model.reactions:
        if each_reaction.boundary:
            if each_reaction.id in medium_info:
                each_reaction.lower_bound = medium_info[each_reaction.id][0]
                each_reaction.upper_bound = medium_info[each_reaction.id][1]
            else:
                each_reaction.lower_bound = 0.0
                each_reaction.upper_bound = 1000.0
        else:
            if each_reaction.upper_bound > 1000.0:
                each_reaction.upper_bound = 1000.0
            if each_reaction.lower_bound < -1000.0:
                each_reaction.lower_bound = -1000.0

            if each_reaction.reversibility:
                each_reaction.lower_bound = -1000.0
                each_reaction.upper_bound = 1000.0
            else:
                each_reaction.lower_bound = 0.0
                each_reaction.upper_bound = 1000.0

    cobra_model = utils.update_cobra_model(cobra_model)
    return cobra_model


def make_metabolic_task_model(cobra_model, metabolic_task_file):
    for each_reaction in cobra_model.reactions:
        if each_reaction.boundary:
            each_reaction.lower_bound = 0.0
            each_reaction.upper_bound = 1000.0

    task_df = pd.read_csv(metabolic_task_file)
    demand_metabolite_info = {}
    medium_information = {}
    for each_row, each_df in task_df.iterrows():
        task_id = each_df.loc['Task ID']
        medium_component_list = each_df.loc['Medium'].split(';')
        if each_df.loc['Type'] == 'Metabolite':
            metabolite = each_df.loc['ID']
            demand_metabolite_info[(metabolite, task_id)] = task_id
        medium_information[task_id] = medium_component_list

    for each_data in demand_metabolite_info:
        each_metabolite = each_data[0]
        task_id = each_data[1]
        cobra_metabolite = cobra_model.metabolites.get_by_id(each_metabolite)
        reaction = Reaction('DMTASK%s_%s' % (task_id, cobra_metabolite.id))
        reaction.name = '%s' % (cobra_metabolite.id)
        reaction.subsystem = 'Demand reaction for %s (TASK ID : %s)' % (each_metabolite, task_id)
        reaction.lower_bound = 0.0
        reaction.upper_bound = 0.0
        reaction.add_metabolites({cobra_metabolite: -1.0})
        cobra_model.add_reactions([reaction])

    for task_id in medium_information:
        medium_component_list = medium_information[task_id]
        reaction = Reaction('TASK_MEDIUM_%s' % (task_id))
        medium_metabolites = {}
        for each_metabolite_information in medium_component_list:
            sptlist = each_metabolite_information[:-1].split('(')
            metabolite = sptlist[0].strip()
            uptake_rate = float(sptlist[1].strip())
            cobra_metabolite = cobra_model.metabolites.get_by_id(metabolite)
            medium_metabolites[cobra_metabolite] = uptake_rate

        reaction.name = 'TASK_MEDIUM_%s' % (task_id)
        reaction.subsystem = 'Medium reaction for TASK %s' % (task_id)
        reaction.lower_bound = 0.0
        reaction.upper_bound = 0.0

        reaction.add_metabolites(medium_metabolites)
        cobra_model.add_reactions([reaction])

    cobra_model = utils.update_cobra_model(cobra_model)
    return cobra_model
