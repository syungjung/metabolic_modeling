import logging
import os
import glob
import time
import sys
import warnings
import os as _os
def _pkg_files_compat(package):
    import importlib, types
    mod = importlib.import_module(package)
    return _os.path.dirname(_os.path.abspath(mod.__file__))
from cobra.io import read_sbml_model

import numpy as np
import pandas as pd
import tqdm

from tf_metabolism.metabolic_simulation import flux_prediction
from tf_metabolism.metabolic_simulation import flux_sum
from tf_metabolism.omics_integration import tINIT
from tf_metabolism.metabolic_simulation import Simulator

from tf_metabolism.utils import argument_parser
from tf_metabolism.utils import (run_umap, run_moma_results_umap,
                                  visualize_moma_umap,
                                  visualize_pathway_enrichment,
                                  visualize_cohort_dimred)

from tf_metabolism.statistical_analysis import statistical_comparison
from tf_metabolism.statistical_analysis import enrichment

from tf_metabolism.metabolic_model import model_editing
from tf_metabolism import __version__


# ---------------------------------------------------------------------------
# GEM reconstruction & flux prediction
# ---------------------------------------------------------------------------

def reconstruct_GEM(biomass_reaction, generic_model_file, universal_model_file, medium_file,
                    output_dir, omics_file, present_metabolite_file, essential_reaction_file,
                    metabolic_task_file):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    reconstructed_models_dir = os.path.join(output_dir, 'reconstructed_models')
    if os.path.isdir(reconstructed_models_dir) and len(glob.glob(reconstructed_models_dir + '/*/functional_*.xml')) > 0:
        logging.info('GEM reconstruction already done for %s — skipping.' % output_dir)
        return

    generic_cobra_model = read_sbml_model(generic_model_file)
    universal_model = read_sbml_model(universal_model_file)
    generic_cobra_model = model_editing.make_medium_model(generic_cobra_model, medium_file)
    universal_model = model_editing.make_medium_model(universal_model, medium_file)

    tINIT.omics_preprocessing(generic_cobra_model, output_dir, omics_file, use_normalization=False)
    tINIT.reconstruct_GEMs(generic_cobra_model, universal_model, biomass_reaction,
                           'INIT', 'rank', present_metabolite_file, essential_reaction_file,
                           metabolic_task_file, medium_file, output_dir)


def predict_metabolic_fluxes(output_dir, flux_output_dir, model_dir, getpra_file, generic_model_file):
    flux_csv = output_dir + '/%s.csv' % os.path.basename(flux_output_dir)
    if os.path.isfile(flux_csv):
        logging.info('Flux prediction already done (%s) — skipping.' % flux_csv)
        return

    if not os.path.isdir(flux_output_dir):
        os.mkdir(flux_output_dir)

    flux_profiles = {}
    for each_dir in glob.glob('%s/reconstructed_models/*' % model_dir):
        each_model_file = glob.glob(each_dir + '/functional_*.xml')[0]
        sample_name = os.path.basename(each_model_file).split('.xml')[0].split('functional_')[1].strip()
        omics_file = '%s/splited_omics_data/%s.csv' % (model_dir, sample_name)
        predicted_flux = flux_prediction.calculate_flux(
            flux_output_dir, omics_file, getpra_file, False, generic_model_file, each_model_file)
        flux_profiles[sample_name] = predicted_flux if not isinstance(predicted_flux, bool) else {}

    output_folder_basename = os.path.basename(flux_output_dir)
    pd.DataFrame.from_dict(flux_profiles).to_csv(output_dir + '/%s.csv' % output_folder_basename)


# ---------------------------------------------------------------------------
# Enrichment analysis
# ---------------------------------------------------------------------------

def predict_enriched_metabolic_pathways(output_dir, cobra_model, diff_flux_df, background_reactions,
                                         flux1_df=None, flux2_df=None):

    def compute_pathway_logfc(pathway_df, selected_df):
        logfc_map = {}
        logfc_col = 'log2 (condition2/condition1)'
        for pathway in pathway_df.index:
            rxns = [r.id for r in cobra_model.reactions
                    if r.subsystem.split(';')[0].strip() == pathway
                    and r.id in selected_df.index]
            if not rxns:
                logfc_map[pathway] = float('nan')
                continue
            # Mean of per-reaction log2FC (already computed, NaN-safe)
            vals = selected_df.loc[selected_df.index.intersection(rxns), logfc_col].dropna()
            logfc_map[pathway] = float(vals.mean()) if len(vals) > 0 else float('nan')
        pos = list(pathway_df.columns).index('Adjusted P-value') + 1
        pathway_df.insert(pos, 'log2FC', pd.Series(logfc_map))
        return pathway_df

    up_df = diff_flux_df[diff_flux_df['Status'] == 'UP']
    up_pathway_df = enrichment.pathway_enrichment_analysis(cobra_model, list(up_df.index), background_reactions)
    up_pathway_df = compute_pathway_logfc(up_pathway_df, up_df)
    up_pathway_df.to_csv(output_dir + '/Up_regulated_pathways.csv')

    down_df = diff_flux_df[diff_flux_df['Status'] == 'DOWN']
    down_pathway_df = enrichment.pathway_enrichment_analysis(cobra_model, list(down_df.index), background_reactions)
    down_pathway_df = compute_pathway_logfc(down_pathway_df, down_df)
    down_pathway_df.to_csv(output_dir + '/Down_regulated_pathways.csv')


def predict_enriched_transcription_factors(transcript_id_info, trrust, cobra_model, output_dir,
                                            diff_flux_df):
    # Load transcript → gene mapping
    gene_transcript_info = {}
    with open(transcript_id_info, 'r') as fp:
        fp.readline()
        for line in fp:
            sptlist = line.strip().split('\t')
            gene_transcript_info[sptlist[1].strip()] = sptlist[0].strip()

    # Load TRRUST TF → target gene mapping
    up_tf_gene_info, down_tf_gene_info = {}, {}
    with open(trrust, 'r') as fp:
        fp.readline()
        for line in fp:
            sptlist = line.strip().split('\t')
            tf, gene, mode = sptlist[0].strip(), sptlist[2].strip(), line
            if 'Activation' in mode:
                up_tf_gene_info.setdefault(tf, [])
                up_tf_gene_info[tf] = list(set(up_tf_gene_info[tf] + [gene]))
            if 'Repression' in mode:
                down_tf_gene_info.setdefault(tf, [])
                down_tf_gene_info[tf] = list(set(down_tf_gene_info[tf] + [gene]))

    def get_genes(reactions):
        transcripts = []
        for rxn in cobra_model.reactions:
            if rxn.id in reactions:
                transcripts += [g.id for g in rxn.genes]
        return list({gene_transcript_info[t] for t in set(transcripts) if t in gene_transcript_info})

    up_genes = get_genes(diff_flux_df[diff_flux_df['Status'] == 'UP'].index)
    enrichment.tf_enrichment_analysis(up_tf_gene_info, up_genes).to_csv(
        output_dir + '/TF_up_enrichment.csv')

    down_genes = get_genes(diff_flux_df[diff_flux_df['Status'] == 'DOWN'].index)
    enrichment.tf_enrichment_analysis(down_tf_gene_info, down_genes).to_csv(
        output_dir + '/TF_down_enrichment.csv')


# ---------------------------------------------------------------------------
# MOMA targeting simulation
# ---------------------------------------------------------------------------

def run_targeting_simulation(output_dir, targeting_result_dir):
    for each_folder in glob.glob(output_dir + '/condition2/reconstructed_models/*'):
        basename = os.path.basename(each_folder)
        result_file = targeting_result_dir + '/MOMA_target_results_%s.csv' % basename
        if os.path.isfile(result_file):
            logging.info('MOMA targeting already done for %s — skipping.' % basename)
            continue
        model_file = glob.glob(each_folder + '/functional_%s.xml' % basename)[0]
        flux_file = glob.glob(output_dir + '/flux2/Flux_prediction_%s.csv' % basename)[0]

        flux_dist = {}
        with open(flux_file, 'r') as fp:
            for line in fp:
                sptlist = line.strip().split(',')
                flux_dist[sptlist[0].strip()] = float(sptlist[1].strip())

        cobra_model = read_sbml_model(model_file)
        for rxn in cobra_model.reactions:
            flux_dist.setdefault(rxn.id, 0.0)

        obj = Simulator.Simulator()
        obj.load_cobra_model(cobra_model)
        non_boundary = {rxn.id for rxn in cobra_model.reactions if not rxn.boundary}

        results = {}
        for gene in tqdm.tqdm(cobra_model.genes):
            flux_constraints = {rxn.id: [0.0, 0.0] for rxn in gene.reactions}
            _, _, perturbed = obj.run_MOMA(wild_flux=flux_dist, flux_constraints=flux_constraints)
            if perturbed is not False:
                results[gene.id] = {rxn: v for rxn, v in perturbed.items() if rxn in non_boundary}

        pd.DataFrame.from_dict(results).to_csv(
            targeting_result_dir + '/MOMA_target_results_%s.csv' % basename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start = time.time()
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    # Keep cobra/libsbml warnings suppressed throughout the run
    logging.getLogger('cobra').setLevel(logging.ERROR)

    parser = argument_parser(version=__version__)
    options = parser.parse_args()
    output_dir   = options.output_dir
    omics_file1  = options.input_omics_file1
    omics_file2  = options.input_omics_file2

    os.makedirs(output_dir, exist_ok=True)
    output_dir_c1       = '%s/condition1' % output_dir
    output_dir_c2       = '%s/condition2' % output_dir
    output_dir_f1       = '%s/flux1' % output_dir
    output_dir_f2       = '%s/flux2' % output_dir
    output_viz_dir      = '%s/viz' % output_dir
    targeting_result_dir = '%s/targeting_results' % output_dir
    os.makedirs(output_viz_dir, exist_ok=True)
    os.makedirs(targeting_result_dir, exist_ok=True)

    ## Resource files
    def _data(fname):
        return _os.path.join(_pkg_files_compat('tf_metabolism'), 'data', fname)

    # Only used by the disabled TF enrichment step — restore together with the
    # predict_enriched_transcription_factors(...) call below to re-enable it.
    # transcript_id_info      = _data('TranscriptID_info.txt')
    # trrust                  = _data('TRRUST_v2_ensembl.tsv')
    getpra_file             = _data('GeTPRA.txt')
    metabolic_task_file     = _data('MetabolicTasks_bigg.csv')
    medium_file             = _data('RPMI1640_medium.txt')
    present_metabolite_file = _data('essential_metabolites_bigg.txt')
    essential_reaction_file = _data('essential_reactions.txt')
    generic_model_file      = _data('Recon2M.2_Entrez_Gene_BIGG.xml')

    logging.info('Loading metabolic model …')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cobra_model = read_sbml_model(generic_model_file)
    logging.info('Model loaded.')

    ## Differential expression analysis
    omics1_df = pd.read_csv(omics_file1, index_col=0)
    omics2_df = pd.read_csv(omics_file2, index_col=0)
    statistical_comparison.two_grouped_data_comparison(omics1_df, omics2_df, p_value_cutoff=0.05)\
        .to_csv(output_dir + '/Differentially_expressed_transcripts.csv')

    ## Reconstruct GEMs
    for omics_file, out_dir in [(omics_file1, output_dir_c1), (omics_file2, output_dir_c2)]:
        reconstruct_GEM('biomass_reaction', generic_model_file, generic_model_file, medium_file,
                        out_dir, omics_file, present_metabolite_file, essential_reaction_file,
                        metabolic_task_file)

    ## Predict metabolic fluxes
    predict_metabolic_fluxes(output_dir, output_dir_f1, output_dir_c1, getpra_file, generic_model_file)
    predict_metabolic_fluxes(output_dir, output_dir_f2, output_dir_c2, getpra_file, generic_model_file)

    ## Statistical analysis of metabolic flux
    flux_file1 = '%s/flux1.csv' % output_dir
    flux_file2 = '%s/flux2.csv' % output_dir
    flux1_df = pd.read_csv(flux_file1, index_col=0)
    flux2_df = pd.read_csv(flux_file2, index_col=0)

    diff_flux_df = statistical_comparison.two_grouped_data_comparison(flux1_df, flux2_df, p_value_cutoff=0.05, filter_by='raw_p', rbc_cutoff=0.5)
    diff_flux_df.to_csv(output_dir + '/Differential_fluxes.csv')
    statistical_comparison.two_grouped_data_comparison(flux1_df, flux2_df, p_value_cutoff=0.05, filter_by='None')\
        .to_csv(output_dir + '/All_flux_comparison.csv')

    ## Flux-sum analysis
    flux_sum.calculate_flux_sum(cobra_model, flux_file1, flux_file2, flux_fillna=False, p_value_cutoff=0.05)\
        .to_csv('%s/flux_sum_results.csv' % output_dir)

    ## Gene-subsystem flux mapping
    expressed_genes = (set(str(g) for g in pd.read_csv('%s/metabolic_gene_expression.csv' % output_dir_c1, index_col=0).index) |
                       set(str(g) for g in pd.read_csv('%s/metabolic_gene_expression.csv' % output_dir_c2, index_col=0).index))
    flux_sum.generate_gene_subsystem_flux(cobra_model, flux_file1, flux_file2, flux_fillna=False, expressed_genes=expressed_genes)\
        .to_csv('%s/gene_subsystem_flux.csv' % output_dir, index=False)

    if diff_flux_df.empty:
        logging.warning('No significant reactions found with P < 0.05.')
        logging.info(time.strftime("Elapsed time %H:%M:%S", time.gmtime(time.time() - start)))
        sys.exit(0)

    ## Enrichment analysis (background: GPR-annotated reactions observed in flux data)
    gpr_rxns = {r.id for r in cobra_model.reactions if r.gene_reaction_rule.strip()}
    background = [r for r in set(flux1_df.index) | set(flux2_df.index) if r in gpr_rxns]
    diff_flux_df_gpr = diff_flux_df[diff_flux_df.index.isin(gpr_rxns)]
    predict_enriched_metabolic_pathways(output_dir, cobra_model, diff_flux_df_gpr, background, flux1_df, flux2_df)
    # TF enrichment is intentionally disabled (kept available as a function below).
    # predict_enriched_transcription_factors(transcript_id_info, trrust, cobra_model, output_dir, diff_flux_df)

    ## MOMA targeting simulation
    run_targeting_simulation(output_dir, targeting_result_dir)

    ## UMAP visualization + MOMA projection (Python-based PCA → UMAP fit, R-based plotting)
    # Run UMAP/PCA fit in Python for 'all' mode
    run_umap(None, output_dir, output_viz_dir, output_dir + '/Differential_fluxes.csv', mode='all')
    visualize_cohort_dimred(output_viz_dir, mode='all', use_r=True)

    # Run UMAP/PCA fit in Python for 'df' mode
    target_reactions_df = run_umap(None, output_dir, output_viz_dir, output_dir + '/Differential_fluxes.csv', mode='df')
    if target_reactions_df:
        visualize_cohort_dimred(output_viz_dir, mode='df', use_r=True)
        run_moma_results_umap(output_dir, output_viz_dir, target_reactions_df, mode='df', use_r=False)
        visualize_moma_umap(output_viz_dir, mode='df', use_r=True)
    else:
        logging.warning('No DF reactions — skipping MOMA projection.')

    ## Pathway enrichment bar plot
    visualize_pathway_enrichment(output_dir, output_viz_dir, use_r=True)

    logging.info(time.strftime("Elapsed time %H:%M:%S", time.gmtime(time.time() - start)))


if __name__ == '__main__':
    main()
