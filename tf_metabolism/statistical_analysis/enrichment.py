import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def pathway_enrichment_analysis(cobra_model, selected_reactions, background_reactions=None):
    if background_reactions is None:
        all_reactions = [each_reaction.id for each_reaction in cobra_model.reactions]
    else:
        all_reactions = list(background_reactions)
    all_reactions_set = set(all_reactions)
    selected_set = set(selected_reactions)

    pathway_info = {}
    for each_reaction in cobra_model.reactions:
        if each_reaction.id not in all_reactions_set:
            continue
        subsystem = each_reaction.subsystem.split(';')[0].strip()
        if not subsystem or subsystem.strip().lower() == 'unassigned':
            continue
        pathway_info.setdefault(subsystem, []).append(each_reaction.id)

    rows = []
    for pathway, pathway_reactions in pathway_info.items():
        if len(pathway_reactions) < 3:
            continue
        pathway_set = set(pathway_reactions)
        if len(pathway_set & selected_set) < 3:
            continue
        not_selected = all_reactions_set - selected_set

        a = len((all_reactions_set - pathway_set) & not_selected)
        b = len(pathway_set & not_selected)
        c = len((all_reactions_set - pathway_set) & selected_set)
        d = len(pathway_set & selected_set)

        oddsratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
        overlap_reactions = sorted(pathway_set & selected_set)

        rows.append({
            'Pathway': pathway,
            'Overlap': f'{d}/{len(pathway_reactions)}',
            'P-value': p_value,
            'Odds ratio': oddsratio,
            'Reactions': ';'.join(overlap_reactions),
        })

    if not rows:
        return pd.DataFrame(columns=['Pathway', 'Overlap', 'P-value', 'Adjusted P-value', 'Odds ratio', 'Reactions'])

    df = pd.DataFrame(rows).set_index('Pathway')
    df = df[df['Odds ratio'] > 1]

    if df.empty:
        return pd.DataFrame(columns=['Pathway', 'Overlap', 'P-value', 'Adjusted P-value', 'Odds ratio', 'Reactions'])

    _, fdr, _, _ = multipletests(df['P-value'], method='fdr_bh')
    pval_pos = list(df.columns).index('P-value') + 1
    df.insert(pval_pos, 'Adjusted P-value', fdr)

    return df


def tf_enrichment_analysis(tf_info, selected_genes):
    all_genes = list({g for genes in tf_info.values() for g in genes})
    all_genes_set = set(all_genes)
    selected_set = set(selected_genes)

    rows = []
    for tf, tf_genes in tf_info.items():
        tf_set = set(tf_genes)
        not_selected = all_genes_set - selected_set

        a = len((all_genes_set - tf_set) & not_selected)
        b = len(tf_set & not_selected)
        c = len((all_genes_set - tf_set) & selected_set)
        d = len(tf_set & selected_set)

        oddsratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
        overlap_genes = sorted(tf_set & selected_set)

        rows.append({
            'TF': tf,
            'Overlap': f'{d}/{len(tf_genes)}',
            'P-value': p_value,
            'Odds ratio': oddsratio,
            'Genes': ';'.join(overlap_genes),
        })

    if not rows:
        return pd.DataFrame(columns=['TF', 'Overlap', 'P-value', 'Adjusted P-value', 'Odds ratio', 'Genes'])

    df = pd.DataFrame(rows).set_index('TF')
    df = df[df['Odds ratio'] > 1]

    if df.empty:
        return pd.DataFrame(columns=['TF', 'Overlap', 'P-value', 'Adjusted P-value', 'Odds ratio', 'Genes'])

    _, fdr, _, _ = multipletests(df['P-value'], method='fdr_bh')
    pval_pos = list(df.columns).index('P-value') + 1
    df.insert(pval_pos, 'Adjusted P-value', fdr)

    return df
