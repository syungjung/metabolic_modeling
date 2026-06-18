import logging
import json
import math
import cobra
import pandas as pd
import numpy as np
from tf_metabolism.statistical_analysis import statistical_comparison

# Currency metabolites to exclude from flux-sum analysis.
# Based on Kim et al. (2024) Genome Biology 25:66, Additional file 4: Table S3.
# BIGG model base IDs (compartment suffix stripped).
CURRENCY_METABOLITES = {
    'atp', 'adp', 'amp',
    'cdp', 'cmp', 'ctp',
    'co2', 'coa',
    'fad', 'fadh2',
    'gdp', 'gmp', 'gtp',
    'h', 'h2o', 'h2o2',
    'k', 'na1',
    'nad', 'nadh', 'nadp', 'nadph',
    'nh4', 'nh3',
    'o2',
    'udp', 'ump', 'utp',
    'zn2',
    'ade', 'adn',
    'cytd', 'csn',
    'dadp', 'damp', 'datp',
    'dcdp', 'dcmp', 'dctp',
    'dgdp', 'dgmp', 'dgtp',
    'dtdp', 'dtmp', 'dttp',
    'dudp', 'dump', 'dutp',
    'ppi',
    'gua', 'gsn',
    'pi', 'so4', 'so3',
    'thm', 'thymd', 'thym',
    'ura', 'uri',
}

def _base_met_id(met_id):
    """Strip compartment suffix: 'atp_c' → 'atp'."""
    parts = met_id.rsplit('_', 1)
    if len(parts) == 2 and len(parts[1]) <= 2:
        return parts[0]
    return met_id

def get_generic_information(cobra_model, absolute_constant=True):
    stoichiometry = {}
    for rxn in cobra_model.reactions:
        stoichiometry[rxn.id] = {}
        for met in rxn.metabolites:
            if absolute_constant:
                stoichiometry[rxn.id][met.id] = abs(rxn.metabolites[met])
            else:
                stoichiometry[rxn.id][met.id] = rxn.metabolites[met]
    rxn_ids = [i.id for i in cobra_model.reactions]
    met_ids = [i.id for i in cobra_model.metabolites]

    return stoichiometry, rxn_ids, met_ids

# input_format
def flux_parsing(flux_file, fillna=False):
    df = pd.read_csv(flux_file, index_col = 0)
    if fillna:
        df.fillna(0.0, inplace=True)
    return df

def flux_sum(flux_df, stoichiometry_dict, met_ids):
    rxn_list = list(flux_df.index.values)
    error_rxn = []
    flux_sum_df = pd.DataFrame(columns=flux_df.columns, index=met_ids ,data=float('nan'))
    for rxn in rxn_list:
        if rxn not in stoichiometry_dict:
            error_rxn.append(rxn)
            continue
        met_sto = stoichiometry_dict[rxn]
        for each_met in met_sto:
            for each_col in list(flux_sum_df.columns.values):
                flux_val = flux_df.loc[rxn, each_col]
                if math.isnan(flux_val):
                    continue
                flux_sum_val = flux_sum_df.loc[each_met, each_col]
                if math.isnan(flux_sum_val):
                    flux_sum_df.loc[each_met, each_col] = 0.0
                flux_sum_df.loc[each_met,each_col] += abs(0.5*float(met_sto[each_met])*float(flux_val))

    flux_sum_df.dropna(how='all', axis=0, inplace=True)

    if error_rxn:
        logging.info("Not in generic model: %s" %str(error_rxn)[1:-2])
    return flux_sum_df


def remove_zero(df1, df2, remove_equal=False):
    zero_index = []
    for index in set(df1.index.values) & set(df2.index.values):
        if remove_equal:
            if sum(df1.loc[index] - min(df1.loc[index])) == 0.0 and sum(df2.loc[index] - min(df2.loc[index])) == 0.0 and min(df1.loc[index]) == min(df2.loc[index]):
                zero_index.append(index)
        else:
            if sum(list(df1.loc[index].values)) == 0.0 and sum(list(df2.loc[index].values))==0.0:
                zero_index.append(index)
    new_df1 = df1.drop(zero_index)
    new_df2 = df2.drop(zero_index)
    return new_df1, new_df2


def get_metabolite_info(metabolite_id_list, cobra_model):
    compartment_dict = cobra_model.compartments
    name_dict = {}
    for each_met in list(metabolite_id_list):
        met = cobra_model.metabolites.get_by_id(each_met)
        subsystems = list({rxn.subsystem for rxn in met.reactions if rxn.subsystem})
        name_dict[each_met] = {
            'Name': met.name,
            'Compartment': compartment_dict[met.compartment],
            'Subsystem': '; '.join(sorted(subsystems))
        }
    name_df = pd.DataFrame(index=['Name', 'Compartment', 'Subsystem'], columns=metabolite_id_list, data=name_dict).T

    return name_df


def rel_rxn(stoichiometry_dict, stat_df, flux_df, column_tag=False):
    column_list = ['Producing reactions','Consuming reactions']
    if column_tag:
        tmp = []
        for i in column_list:
            tmp.append(i+' '+column_tag)
        column_list = tmp
    rel_rxn_df = pd.DataFrame(index=stat_df.index,
                            columns=column_list,
                            data = 0.0)
    flux_df.fillna(0.0, inplace=True)
    total_model = len(list(flux_df.columns.values))
    for each_ind in list(flux_df.index.values):
        values = flux_df.loc[each_ind]
        reaction_abundance = (np.sum(values>0.0), np.sum(values<0.0))
        for each_met in stoichiometry_dict[each_ind]:
            if not each_met in rel_rxn_df.index:
                continue
            constant = stoichiometry_dict[each_ind][each_met]
            if constant>0.0:
                rel_rxn_df.loc[each_met,column_list[0]] = rel_rxn_df.loc[each_met,column_list[0]] + reaction_abundance[0]
                rel_rxn_df.loc[each_met,column_list[1]] = rel_rxn_df.loc[each_met,column_list[1]] + reaction_abundance[1]
            elif constant<0.0:
                rel_rxn_df.loc[each_met,column_list[0]] = rel_rxn_df.loc[each_met,column_list[0]] + reaction_abundance[1]
                rel_rxn_df.loc[each_met,column_list[1]] = rel_rxn_df.loc[each_met,column_list[1]] + reaction_abundance[0]
    rel_rxn_df = rel_rxn_df/total_model

    return rel_rxn_df

# method is one of ['order', 'cutoff_equal']
def sort_index(input_df, by, method, value=False):
    sorted_index_list = []
    if method == 'order':
        sorted_index_list = list(input_df.sort_values(by=by, ascending=value).index.values)
    elif method.startswith('cutoff'):
        if method.endswith('lower'):
            target = np.array(input_df[by])<value
        elif method.endswith('upper'):
            target = np.array(input_df[by])>value
        elif method.endwith('equal'):
            target = np.array(input_df[by])==value
        index_list = list(input_df.index.values)
        passed = []
        failed = []
        for i in index_list:
            if target[index_list.index(i)]:
                passed.append(i)
            else:
                failed.append(i)
        sorted_index_list = passed+failed

    elif method == 'abs_order':
        input_dict = input_df.to_dict()
        target_dict = input_dict[by]
        for each_ind in target_dict:
            target_dict[each_ind] = abs(target_dict[each_ind])
        target_df = pd.DataFrame(data={by:target_dict})
        sorted_index_list = list(target_df.sort_values(by=by, ascending=value).index.values)

    return sorted_index_list


def filter_subsystems(subsystem_set):
    return {s for s in subsystem_set if s and s.strip().lower() != 'unassigned'}


def generate_gene_subsystem_flux(cobra_model, flux_file1, flux_file2,
                                 flux_fillna=False,
                                 expressed_genes=None, related=False, p_value_cutoff=0.05,
                                 force_non_parametric=False):
    flux_df1 = flux_parsing(flux_file1, fillna=flux_fillna)
    flux_df2 = flux_parsing(flux_file2, fillna=flux_fillna)

    stat_df = statistical_comparison.two_grouped_data_comparison(
        flux_df1, flux_df2, p_value_cutoff=p_value_cutoff, filter_by='None',
        force_non_parametric=force_non_parametric)
    stat_cols = list(stat_df.columns)

    all_flux_df = pd.concat([flux_df1, flux_df2], axis=1)
    sample_cols = list(all_flux_df.columns)

    rows = []
    for gene in cobra_model.genes:
        gene_id = gene.id

        if expressed_genes is not None and gene_id not in expressed_genes:
            continue

        for rxn in gene.reactions:
            if not filter_subsystems({rxn.subsystem}):
                continue
            if rxn.id not in all_flux_df.index:
                continue
            row = {
                'Gene': gene_id,
                'Subsystem': rxn.subsystem,
                'Reaction': rxn.id
            }
            if rxn.id not in stat_df.index:
                continue
            for col in stat_cols:
                row[col] = stat_df.loc[rxn.id, col]
            for col in sample_cols:
                row[col] = all_flux_df.loc[rxn.id, col]
            rows.append(row)

    return pd.DataFrame(rows, columns=['Gene', 'Subsystem', 'Reaction'] + stat_cols + sample_cols)


def calculate_flux_sum(cobra_model, flux_file1, flux_file2,
                        flux_fillna=False,
                        remove_zero_flux_sum=False, remove_equal=False,
                        related=False, p_value_cutoff=0.05, filter_by='None',
                        force_non_parametric=False,
                        exclude_currency=True,
                        result_info=['name', 'related reaction', 'flux-sum']):
    stoichiometry_dict, reaction_id_list, metabolite_id_list = get_generic_information(cobra_model)
    flux_df1 = flux_parsing(flux_file1, fillna=flux_fillna)
    flux_df2 = flux_parsing(flux_file2, fillna=flux_fillna)
    flux_sum_df1 = flux_sum(flux_df1, stoichiometry_dict, metabolite_id_list)
    flux_sum_df2 = flux_sum(flux_df2, stoichiometry_dict, metabolite_id_list)

    # Exclude currency metabolites (Kim et al. 2024, Genome Biology 25:66, Table S3)
    if exclude_currency:
        non_curr = [m for m in flux_sum_df1.index if _base_met_id(m) not in CURRENCY_METABOLITES]
        flux_sum_df1 = flux_sum_df1.loc[non_curr]
        flux_sum_df2 = flux_sum_df2.loc[flux_sum_df2.index.isin(non_curr)]

    if remove_zero_flux_sum or remove_equal:
        flux_sum_df1, flux_sum_df2 = remove_zero(flux_sum_df1, flux_sum_df2, remove_equal=remove_equal)
    # statistical analysis
    stat_df = statistical_comparison.two_grouped_data_comparison(flux_sum_df1,
                        flux_sum_df2,
                        p_value_cutoff=p_value_cutoff,
                        filter_by=filter_by,
                        force_non_parametric=force_non_parametric)

    # Sorting: by log2FC (abs desc), then by FDR or P (significant first)
    sorted_index1 = sort_index(input_df=stat_df, by='log2 (condition2/condition1)', method='abs_order', value=False)
    new_stat_df = pd.DataFrame(data=stat_df.to_dict(), index=sorted_index1, columns=stat_df.columns)
    sort_col = 'FDR (adjusted P)' if 'FDR (adjusted P)' in new_stat_df.columns else 'P'
    sorted_index2 = sort_index(input_df=new_stat_df, by=sort_col, method='cutoff_lower', value=p_value_cutoff)
    sorted_index = sorted_index2

    result_dict = stat_df.to_dict()
    columns = list(stat_df.columns.values)
    if 'related reaction' in result_info:
        stoichiometry_dict, reaction_id_list, metabolite_id_list = get_generic_information(cobra_model, absolute_constant=False)
        rel_rxn_df1 = rel_rxn(stoichiometry_dict, stat_df, flux_df1, column_tag='condition1')
        rel_rxn_df2 = rel_rxn(stoichiometry_dict, stat_df, flux_df2, column_tag='condition2')
        rel_rxn_df = pd.concat([rel_rxn_df1, rel_rxn_df2],axis=1)
        columns = list(rel_rxn_df.columns.values)+columns
        rel_rxn_dict = rel_rxn_df.to_dict()
        for i in rel_rxn_dict:
            result_dict[i] = rel_rxn_dict[i]
    if 'name' in result_info:
        name_df = get_metabolite_info(list(stat_df.index.values), cobra_model)
        columns = list(name_df.columns.values)+columns
        name_dict = name_df.to_dict()
        for i in name_dict:
            result_dict[i] = name_dict[i]
    result_df = pd.DataFrame(result_dict, index = sorted_index, columns = columns)
    if 'flux-sum' in result_info:
        result_df = pd.concat([result_df, flux_sum_df1, flux_sum_df2], axis=1).dropna(subset=['Name'])
        result_df = result_df.reindex(sorted_index)
    return result_df

