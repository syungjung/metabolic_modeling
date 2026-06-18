import copy

import pandas as pd
from tf_metabolism import utils
from tf_metabolism.metabolic_model import model_editing
from tf_metabolism.metabolic_simulation import Gap
from tf_metabolism.metabolic_simulation import metabolic_task


def fill_functional_metabolic_gaps(universal_model, metabolic_gap_model, metabolic_task_file):
    initial_metabolic_gap_model = copy.deepcopy(metabolic_gap_model)
    initial_universal_model = copy.deepcopy(universal_model)
    draft_task_result_df = metabolic_task.evaluate_metabolic_task(metabolic_gap_model, metabolic_task_file)
    failed_task_result_df = draft_task_result_df[draft_task_result_df['Task result'] != 'PASSED']
    universal_model = model_editing.make_metabolic_task_model(universal_model, metabolic_task_file)
    metabolic_gap_model = model_editing.make_metabolic_task_model(metabolic_gap_model, metabolic_task_file)
    task_df = failed_task_result_df[['Task ID', 'Type', 'ID', 'Medium', 'Constraints', 'Expected value', 'Description']]
    task_info = {}
    task_success_info = {}
    for each_row, each_df in task_df.iterrows():
        task_id = each_df.loc['Task ID']
        expected_value = each_df.loc['Expected value']
        task_success_info[task_id] = expected_value
        if each_df.loc['Type'] == 'Metabolite':
            metabolite = each_df.loc['ID']
            target_reaction = 'DMTASK%s_%s' % (task_id, metabolite)
            task_info[task_id] = target_reaction
        else:
            target_reaction = each_df.loc['ID']
            task_info[task_id] = target_reaction

    universal_model_reactions = []
    for each_reaction in universal_model.reactions:
        universal_model_reactions.append(each_reaction.id)
        if each_reaction.boundary:
            each_reaction.lower_bound = 0.0
            each_reaction.upper_bound = 1000.0

    for each_reaction in metabolic_gap_model.reactions:
        if each_reaction.boundary:
            each_reaction.lower_bound = 0.0
            each_reaction.upper_bound = 1000.0

    unique_added_reactions = []
    task_result_list = []
    solution_status_list = []
    solution_value_list = []
    added_reaction_list = []
    original_universal_model = copy.deepcopy(universal_model)
    original_metabolic_gap_model = copy.deepcopy(metabolic_gap_model)
    for each_row, each_df in task_df.iterrows():
        task_id = each_df.loc['Task ID']
        universal_model = copy.deepcopy(original_universal_model)
        cobra_gap_model = copy.deepcopy(original_metabolic_gap_model)
        temp_flux_list = []
        each_target_reaction = task_info[task_id]
        if each_target_reaction in universal_model_reactions:
            medium_reaction = 'TASK_MEDIUM_%s' % task_id
            medium_reaction_index = universal_model.reactions.index(medium_reaction)
            universal_model.reactions[medium_reaction_index].lower_bound = -1.0
            medium_reaction_index = cobra_gap_model.reactions.index(medium_reaction)
            cobra_gap_model.reactions[medium_reaction_index].lower_bound = -1.0
            target_reaction_index = universal_model.reactions.index(each_target_reaction)
            if universal_model.reactions[target_reaction_index].reversibility:
                universal_model.reactions[target_reaction_index].lower_bound = -1000.0
            universal_model.reactions[target_reaction_index].upper_bound = 1000.0
            obj = Gap.GapFilling()
            model_status, objective_value, added_reactions, reaction_object_list = obj.run_gap_filling(universal_model,
                                                                                                       cobra_gap_model,
                                                                                                       each_target_reaction)
            if added_reactions != False:
                reactions = [reaction.id for reaction in cobra_gap_model.reactions]
                cobra_gap_model.add_reactions(reaction_object_list)
                cobra_gap_model.objective = each_target_reaction
                cobra_gap_model.optimize(solver='gurobi')
                solution_status_list.append(model_status)
                task_result_list.append('GAP FILLED')
                solution_value_list.append(cobra_gap_model.solution.f)
                added_reaction_list.append(';'.join(added_reactions))
                unique_added_reactions += added_reactions
            else:
                task_result_list.append('FAILED')
                added_reaction_list.append('NA')
                solution_value_list.append('NA')
                if not model_status:
                    solution_status_list.append('NA')
                else:
                    solution_status_list.append(model_status)
        else:
            task_result_list.append('FAILED - not exist in the model')
            added_reaction_list.append('NA')
            solution_value_list.append('NA')
            solution_status_list.append('NA')

    unique_added_reactions = list(set(unique_added_reactions))
    task_df = failed_task_result_df[['Task ID', 'Type', 'ID', 'Medium', 'Constraints', 'Expected value', 'Description']].copy()
    task_df['Gap filling result'] = pd.Series(task_result_list, index=task_df.index)
    task_df['Added reaction'] = pd.Series(added_reaction_list, index=task_df.index)
    task_df['Solution status'] = pd.Series(solution_status_list, index=task_df.index)
    task_df['Solution value'] = pd.Series(solution_value_list, index=task_df.index)
    reaction_object_list = []
    for each_reaction in unique_added_reactions:
        reaction_object_list.append(initial_universal_model.reactions.get_by_id(each_reaction))

    initial_metabolic_gap_model.add_reactions(reaction_object_list)
    functional_metabolic_model = utils.update_cobra_model(initial_metabolic_gap_model)
    return (functional_metabolic_model, task_df, draft_task_result_df, failed_task_result_df)
