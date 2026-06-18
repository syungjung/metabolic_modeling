import argparse
import copy
import glob
import logging
import os
import shutil
import time
import warnings

import numpy as np
import pandas as pd
from cobra.io import read_sbml_model
from cobra.io import write_sbml_model
from cobra.manipulation import delete
from tf_metabolism.metabolic_model import model_editing
from tf_metabolism.metabolic_simulation import gap_filling
from tf_metabolism.metabolic_simulation import metabolic_task
from tf_metabolism.omics_integration import model_reconstruction
from tf_metabolism.omics_integration import omics_data_manipulation
from tf_metabolism.omics_integration import omics_score_calculation


def read_present_metabolites(present_metabolite_file):
    present_metabolites = []
    with open(present_metabolite_file, 'r') as fp:
        for line in fp:
            present_metabolites.append(line.strip())
    return present_metabolites


def read_essential_reactions(essential_reaction_file):
    essential_reactions = []
    with open(essential_reaction_file, 'r') as fp:
        for line in fp:
            essential_reactions.append(line.strip())
    return essential_reactions


def omics_preprocessing(cobra_model, output_dir, omics_file, use_normalization=False):
    logging.info("Preparing omics data starting..")
    metabolic_expression_df, gene_coverage = omics_data_manipulation.omics_preprocessing(cobra_model, omics_file,
                                                                                         use_normalization)

    metabolic_expression_df.to_csv(output_dir + '/metabolic_gene_expression.csv')
    with open(output_dir + '/expression_data_summary.txt', 'w') as fp:
        print('Number of metabolic genes\t%s' % (len(metabolic_expression_df.index)), file=fp)
        print('Gene coverage\t%s\t' % (gene_coverage), file=fp)

    omics_data_manipulation.split_expression_data_by_column(output_dir + '/metabolic_gene_expression.csv',
                                                            output_dir + '/splited_omics_data/')
    omics_data_manipulation.analyze_omics_data(output_dir + '/metabolic_gene_expression.csv',
                                               output_dir + '/omics_data_analysis_result/')
    return


def reconstruct_GEMs(generic_cobra_model, universal_model, biomass_reaction, integration_method, scoring_method, present_metabolite_file, essential_reaction_file, metabolic_task_file, medium_file, output_dir):
    logging.info("Reconstructing GEMs using tINIT")
    start = time.time()

    original_generic_cobra_model = copy.deepcopy(generic_cobra_model)
    original_universal_model = copy.deepcopy(universal_model)

    expression_data_dir = output_dir + '/splited_omics_data/'
    model_output_dir = output_dir + '/reconstructed_models/'

    try:
        shutil.rmtree(model_output_dir)
    except:
        pass

    try:
        os.mkdir(model_output_dir)
    except:
        pass

    present_metabolites = read_present_metabolites(present_metabolite_file)
    essential_reactions = read_essential_reactions(essential_reaction_file)

    new_metabolic_task_df = metabolic_task.metabolic_task_generation(generic_cobra_model, present_metabolites, essential_reactions, medium_file, metabolic_task_file)
    new_metabolic_task_file = output_dir + '/updated_metabolic_task.csv'
    new_metabolic_task_df.to_csv(new_metabolic_task_file, index=False)

    expression_data_files = glob.glob(expression_data_dir + '*')

    average_expression_value = 0.0
    expression_values = []
    for each_expression_data_file in expression_data_files:
        df = pd.read_csv(each_expression_data_file, index_col=0)
        expression_values += list(df.values)

    average_expression_value = np.mean(expression_values)

    reaction_weights = {}
    draft_model_reconstruction = {}
    functional_model_statistics = {}
    for each_expression_data_file in expression_data_files:
        model_id = os.path.basename(each_expression_data_file).split('.')[0].strip()
        logging.info("Reconstructing %s GEM" % (model_id))
        start = time.time()
        generic_cobra_model = copy.deepcopy(original_generic_cobra_model)
        universal_model = copy.deepcopy(original_universal_model)

        basename = os.path.basename(each_expression_data_file).split('.csv')[0].strip()

        reaction_weights[basename] = {}
        draft_model_reconstruction[basename] = {}
        functional_model_statistics[basename] = {}

        expression_df = pd.read_csv(each_expression_data_file, index_col=0)

        if scoring_method == 'rank':
            calculated_expression_score = omics_score_calculation.calculate_rank_based_expression_score(expression_df,
                                                                                                        0.25)
        else:
            calculated_expression_score = omics_score_calculation.calculate_original_expression_score(expression_df,
                                                                                                      average_expression_value)

        each_reaction_weight = omics_score_calculation.reaction_score_calculation(generic_cobra_model,
                                                                                  calculated_expression_score)
        reaction_weights[basename] = each_reaction_weight

        model_status, objective_value, flux_distribution, context_model = model_reconstruction.reconstruct_GEM(
            generic_cobra_model, each_reaction_weight, present_metabolites, essential_reactions, biomass_reaction)

        if model_status == 2:
            each_model_output_dir = model_output_dir + '%s/' % (basename)
            os.mkdir(each_model_output_dir)
            draft_model_file = each_model_output_dir + 'draft_%s.xml' % (basename)
            # cobra 0.5.4 bug: delete.prune_unused_metabolites passes cobra_model
            # as method arg; remove_from_model() takes no positional args
            for _met in list(context_model.metabolites):
                if len(_met._reaction) == 0:
                    _met.remove_from_model()
            write_sbml_model(context_model, draft_model_file, use_fbc_package=False)

            context_model.optimize(solver='gurobi')

            draft_cobra_model = read_sbml_model(each_model_output_dir + 'draft_%s.xml' % (basename))
            draft_cobra_model = model_editing.make_medium_model(context_model, medium_file)
            universal_model = model_editing.make_medium_model(universal_model, medium_file)

            functional_model, gapfilling_result_df, draft_task_df, failed_task_df = gap_filling.fill_functional_metabolic_gaps(
                universal_model, draft_cobra_model, new_metabolic_task_file)

            functional_model.optimize(solver='gurobi')

            gapfilling_result_df.to_csv(each_model_output_dir + 'gap_filling_result.csv', index=False)
            draft_task_df.to_csv(each_model_output_dir + 'draft_metabolic_task_result.csv', index=False)
            failed_task_df.to_csv(each_model_output_dir + 'failed_metabolic_task_result.csv', index=False)
            delete.prune_unused_metabolites(functional_model)

            task_result_df = metabolic_task.evaluate_metabolic_task(functional_model, new_metabolic_task_file)
            task_result_df.to_csv(each_model_output_dir + 'functional_model_metabolic_task_result.csv', index=False)
            write_sbml_model(functional_model, each_model_output_dir + 'functional_%s.xml' % (basename), use_fbc_package=False)
            functional_model = read_sbml_model(each_model_output_dir + 'functional_%s.xml' % (basename))
            draft_model_reconstruction[basename]['No. of reactions'] = len(draft_cobra_model.reactions)
            draft_model_reconstruction[basename]['No. of metabolites'] = len(draft_cobra_model.metabolites)
            draft_model_reconstruction[basename]['No. of genes'] = len(draft_cobra_model.genes)
            draft_model_reconstruction[basename]['Growth rate'] = draft_cobra_model.solution.f
            draft_model_reconstruction[basename]['Objective value'] = objective_value
            draft_model_reconstruction[basename]['Soultion status'] = '2 (Optimal)'

            passed_metabolic_task_cnt = len(task_result_df[task_result_df['Task result'] == 'PASSED'])
            failed_metabolic_task_cnt = len(task_result_df[task_result_df['Task result'] != 'PASSED'])
            functional_model_statistics[basename]['No. of reactions'] = len(functional_model.reactions)
            functional_model_statistics[basename]['No. of metabolites'] = len(functional_model.metabolites)
            functional_model_statistics[basename]['No. of genes'] = len(functional_model.genes)
            functional_model_statistics[basename]['Growth rate'] = functional_model.solution.f
            functional_model_statistics[basename]['No. of passed metabolic tasks'] = passed_metabolic_task_cnt
            functional_model_statistics[basename]['No. of failed metabolic tasks'] = failed_metabolic_task_cnt
        else:
            draft_model_reconstruction[basename]['No. of reactions'] = 'N/A'
            draft_model_reconstruction[basename]['No. of metabolites'] = 'N/A'
            draft_model_reconstruction[basename]['No. of genes'] = 'N/A'
            draft_model_reconstruction[basename]['Growth rate'] = 'N/A'
            draft_model_reconstruction[basename]['Objective value'] = 'N/A'
            draft_model_reconstruction[basename]['Soultion status'] = model_status
        logging.info('%s model reconstructed' % model_id)
        logging.info(time.strftime("Elapsed time %H:%M:%S", time.gmtime(time.time() - start)))

    reaction_weight_df = pd.DataFrame.from_dict(reaction_weights)
    reaction_weight_df.to_csv(output_dir + '/reaction_weights.csv')
    draft_model_information_df = pd.DataFrame.from_dict(draft_model_reconstruction)
    draft_model_information_df.T.to_csv(output_dir + '/draft_model_information.csv')
    functional_model_information_df = pd.DataFrame.from_dict(functional_model_statistics)
    functional_model_information_df.T.to_csv(output_dir + '/functional_model_information.csv')
    logging.info(time.strftime("Elapsed time %H:%M:%S", time.gmtime(time.time() - start)))
