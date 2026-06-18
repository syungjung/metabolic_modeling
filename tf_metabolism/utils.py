import json
import tempfile
import argparse
import logging
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import joblib
import glob

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cobra.io import read_sbml_model, write_sbml_model

# ---------------------------------------------------------------------------
# R visualization scripts live under tf_metabolism/r_based_viz/
# ---------------------------------------------------------------------------
R_VIZ_DIR = os.path.join(os.path.dirname(__file__), 'r_based_viz')


def _r_script(name):
    """Absolute path to a bundled R visualization script."""
    return os.path.join(R_VIZ_DIR, name)


# ---------------------------------------------------------------------------
# Rscript resolver
# ---------------------------------------------------------------------------
def _find_rscript():
    """Return the Rscript executable path.

    Search order:
    1. RSCRIPT env variable (user override)
    2. shutil.which('Rscript') — searches PATH
    3. Raises RuntimeError if not found
    """
    env = os.environ.get('RSCRIPT')
    if env:
        return env
    found = shutil.which('Rscript')
    if found:
        return found
    raise RuntimeError(
        'Rscript not found. Set the RSCRIPT environment variable or add R to PATH.'
    )


def run_pathway_enrichment_plot(output_dir, output_viz_dir,
                                fdr_cutoff=0.1, title=None, r_script=None):
    """Visualize pathway enrichment results via pathway_enrichment_plot.R.

    Reads Up_regulated_pathways.csv / Down_regulated_pathways.csv from
    ``output_dir`` and writes ``{output_viz_dir}/pathway_barplot.png`` (+ .svg).

    Parameters
    ----------
    output_dir     : directory containing the enrichment CSVs
    output_viz_dir : where the figures are written
    fdr_cutoff     : Adjusted P-value threshold (default 0.1)
    title          : optional plot title suffix (e.g. disease label)
    r_script       : path to pathway_enrichment_plot.R (auto-detected if None)
    """
    import subprocess

    if r_script is None:
        r_script = _r_script('pathway_enrichment_plot.R')
    if not os.path.exists(r_script):
        raise FileNotFoundError(f'R script not found: {r_script}')

    os.makedirs(output_viz_dir, exist_ok=True)

    cmd = [
        _find_rscript(), r_script,
        '--output-dir',     output_dir,
        '--output-viz-dir', output_viz_dir,
        '--fdr-cutoff',     str(fdr_cutoff),
    ]
    if title:
        cmd += ['--title', title]

    logging.info('[R] Running: %s', ' '.join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError('pathway_enrichment_plot.R exited with code %d' % result.returncode)
    logging.info('[R] Pathway enrichment plot done.')


def visualize_pathway_enrichment(output_dir, output_viz_dir, use_r=True,
                                 fdr_cutoff=0.1, title=None):
    """Bar plot of enriched metabolic pathways, with a selectable backend.

    Mirrors the UMAP visualization style: pass ``use_r=True`` to render with the
    R script (``pathway_enrichment_plot.R``, ggplot2) or ``use_r=False`` to render
    with matplotlib. Both backends read Up/Down_regulated_pathways.csv from
    ``output_dir`` and write ``{output_viz_dir}/pathway_barplot.png`` (+ .svg).

    Parameters
    ----------
    output_dir     : directory containing the enrichment CSVs
    output_viz_dir : where the figures are written
    use_r          : True → R/ggplot2 backend, False → matplotlib backend
    fdr_cutoff     : Adjusted P-value threshold (default 0.1)
    title          : optional plot title suffix (e.g. disease label)
    """
    if use_r:
        run_pathway_enrichment_plot(output_dir, output_viz_dir,
                                    fdr_cutoff=fdr_cutoff, title=title)
    else:
        _pathway_barplot_matplotlib(output_dir, output_viz_dir,
                                    fdr_cutoff=fdr_cutoff, title=title)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONDITION_COLORS = {'Control': 'royalblue', 'Test': '#DC143C'}
DATASET_MARKERS  = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '8']


def argument_parser(version=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', required=True, help="Output directory")
    parser.add_argument('-c1', '--input_omics_file1', required=True,
                        help="Input omics file for condition 1")
    parser.add_argument('-c2', '--input_omics_file2', required=True,
                        help="Input omics file for condition 2")
    return parser


def update_cobra_model(cobra_model):
    temp = tempfile.NamedTemporaryFile(prefix='temp_cobra_', suffix='.xml', dir='./', delete=True)
    temp_outfile = temp.name
    temp.close()

    write_sbml_model(cobra_model, temp_outfile)
    cobra_model = read_sbml_model(temp_outfile)
    os.remove(temp_outfile)
    return cobra_model


def find_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    return df.index[outlier_mask.any(axis=1)].tolist()


def set_publication_style(
    base_size: int = 10,
    base_family: str = 'sans-serif',
    width_mm: float = 90.0,
    height_ratio: float = 0.66,
):
    width_in = width_mm / 25.4
    height_in = width_in * height_ratio

    plt.style.use('default')
    sns.set_theme(style='white')

    plt.rcParams.update({
        # Fonts
        'font.size': base_size,
        'font.family': base_family,
        'axes.titlesize': base_size + 1,
        'axes.titleweight': 'medium',
        'axes.titlepad': 8,
        'axes.labelsize': base_size,
        'axes.labelweight': 'medium',
        'xtick.labelsize': max(base_size - 1, 6),
        'ytick.labelsize': max(base_size - 1, 6),
        'legend.fontsize': max(base_size - 1, 6),
        'legend.title_fontsize': base_size,

        # Figure sizing and backgrounds
        'figure.figsize': (width_in, height_in),
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',

        # Axes and spines (minimal)
        'axes.grid': False,
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,

        # Ticks (clear and outward)
        'xtick.bottom': True,
        'ytick.left': True,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        'xtick.color': 'black',
        'ytick.color': 'black',

        # Legend (clean)
        'legend.frameon': False,

        # Titles
        'axes.titlelocation': 'center',

        # Layout and saving
        'figure.autolayout': False,
        'figure.constrained_layout.use': True,
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def run_umap(cobra_model, output_dir, output_viz_dir, flux_comparison_file,
             mode='all', random_state=42):
    """Run UMAP dimensionality reduction.

    Parameters
    ----------
    cobra_model          : COBRApy Model (kept for API compatibility)
    output_dir           : directory containing flux1.csv / flux2.csv
    output_viz_dir       : output directory for UMAP results
    flux_comparison_file : CSV of reactions to include
    mode                 : 'all' or 'df'
    random_state         : RNG seed

    Returns
    -------
    target_reactions : list of reaction IDs used
    """
    label_str = 'All Flux' if mode == 'all' else 'DF reactions only'
    logging.info(f'\n[UMAP-{mode}] {label_str}')

    # Reaction selection: all common reactions from flux1 ∩ flux2
    df1_full = pd.read_csv(f'{output_dir}/flux1.csv', index_col=0)
    df2_full = pd.read_csv(f'{output_dir}/flux2.csv', index_col=0)
    common_all = list(df1_full.index.intersection(df2_full.index))
    logging.info(f'  Common reactions (flux1 ∩ flux2): {len(common_all)}')

    if mode == 'df':
        ref_df = pd.read_csv(flux_comparison_file, index_col=0)
        target_reactions = [r for r in common_all if r in ref_df.index]
        logging.info(f'  Target reactions after DF filter: {len(target_reactions)}')
    else:
        target_reactions = common_all
        logging.info(f'  Target reactions (all): {len(target_reactions)}')

    # Load & merge flux matrices
    df1 = df1_full.reindex(target_reactions).dropna(how='all')
    df2 = df2_full.reindex(target_reactions).dropna(how='all')
    common_rxns = df1.index.intersection(df2.index)
    df1 = df1.loc[common_rxns]
    df2 = df2.loc[common_rxns]

    c_cols = list(df1.columns)
    t_cols = list(df2.columns)
    merged = pd.concat([df1.T, df2.T], axis=0).fillna(0)

    rv     = merged.var(axis=0)
    merged = merged.loc[:, rv > 0]
    logging.info(f'  Samples: {len(c_cols)} Control + {len(t_cols)} Test  |  '
                 f'Reactions after zero-var filter: {merged.shape[1]}')

    conditions = np.array(['Control'] * len(c_cols) + ['Test'] * len(t_cols))

    os.makedirs(output_viz_dir, exist_ok=True)

    # StandardScaler + PCA + UMAP
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(merged.values)

    n_pcs     = min(30, merged.shape[0] - 1, merged.shape[1])
    pca_model = PCA(n_components=n_pcs, random_state=random_state)
    pca_scores = pca_model.fit_transform(X_scaled)

    reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=random_state)
    embedding = reducer.fit_transform(pca_scores)

    out_df = pd.DataFrame(embedding, index=merged.index, columns=['UMAP1', 'UMAP2'])
    out_df['condition'] = conditions
    out_df.to_csv(os.path.join(output_viz_dir, f'UMAP_{mode}_raw_data.csv'))

    # Save models for MOMA projection
    joblib.dump(scaler,    os.path.join(output_viz_dir, f'umap_{mode}_scaler.pkl'))
    joblib.dump(pca_model, os.path.join(output_viz_dir, f'umap_{mode}_pca.pkl'))
    joblib.dump(reducer,   os.path.join(output_viz_dir, f'umap_{mode}_model.pkl'))

    logging.info(f'  Saved: UMAP_{mode}_raw_data.csv | scaler/pca/umap pkl')
    return target_reactions


def run_r_umap(output_dir, output_viz_dir,
               df_file=None, mode='both', r_script=None,
               c1_color=None, c2_color=None,
               c1_label=None, c2_label=None):
    """Call flux_dimred.R via subprocess, then fit sklearn models from R output.

    R handles: PCA → UMAP → visualization → CSV/JSON outputs.
    Python reads R's merged flux matrix + optimal params, fits
    StandardScaler + PCA + UMAP with those exact parameters, and saves pkl
    files for MOMA projection.

    Parameters
    ----------
    output_dir        : directory containing flux1.csv / flux2.csv
    output_viz_dir    : where R writes results (and pkl files are saved)
    df_file           : Differential_fluxes.csv (optional, for df mode)
    mode              : 'all' | 'df' | 'both'
    r_script          : path to flux_dimred.R (auto-detected if None)

    Returns
    -------
    dict  {mode: target_reactions_list}
    """
    import subprocess

    if r_script is None:
        r_script = _r_script('flux_dimred.R')
    if not os.path.exists(r_script):
        raise FileNotFoundError(f'R script not found: {r_script}')

    os.makedirs(output_viz_dir, exist_ok=True)

    cmd = [
        _find_rscript(), r_script,
        '--output-dir',     output_dir,
        '--output-viz-dir', output_viz_dir,
        '--mode',           mode,
    ]
    if df_file and os.path.exists(df_file):
        cmd += ['--df-file', df_file]
    if c1_color:
        cmd += ['--c1-color', c1_color]
    if c2_color:
        cmd += ['--c2-color', c2_color]
    if c1_label:
        cmd += ['--c1-label', c1_label]
    if c2_label:
        cmd += ['--c2-label', c2_label]

    logging.info('[R] Running: %s', ' '.join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError('flux_dimred.R exited with code %d' % result.returncode)

    logging.info('[R] Done.')

    # ------------------------------------------------------------------
    # Fit sklearn models from R outputs (for MOMA projection)
    # ------------------------------------------------------------------
    merged_csv = os.path.join(output_viz_dir, 'merged_flux.csv')
    if not os.path.exists(merged_csv):
        raise FileNotFoundError(f'R did not produce: {merged_csv}')
    mat_corr = pd.read_csv(merged_csv, index_col=0)   # features × samples

    target_reactions = {}
    modes_to_fit = ['all', 'df'] if mode == 'both' else [mode]

    for m in modes_to_fit:
        params_json = os.path.join(output_viz_dir, f'umap_{m}_params.json')
        if not os.path.exists(params_json):
            logging.warning('Params file not found (R skipped mode=%s): %s', m, params_json)
            continue

        with open(params_json) as fp:
            params = json.load(fp)

        n_pcs    = int(params['n_pcs'])
        best_nn  = int(params['best_nn'])
        best_md  = float(params['best_dist'])
        features = params['features']

        target_reactions[m] = features

        # Subset + transpose: samples × features
        mat_m = mat_corr.reindex(features).fillna(0).T   # (n_samples, n_features)

        logging.info('[sklearn-%s] n_samples=%d  n_features=%d  n_pcs=%d  '
                     'n_neighbors=%d  min_dist=%s',
                     m, mat_m.shape[0], mat_m.shape[1], n_pcs, best_nn, best_md)

        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(mat_m.values)

        pca_model = PCA(n_components=min(n_pcs, mat_m.shape[0] - 1, mat_m.shape[1]),
                        random_state=42)
        pca_scores = pca_model.fit_transform(X_scaled)

        reducer   = umap.UMAP(n_neighbors=best_nn, min_dist=best_md,
                              n_components=2, random_state=42)
        embedding = reducer.fit_transform(pca_scores)   # (n_samples, 2)

        joblib.dump(scaler,    os.path.join(output_viz_dir, f'umap_{m}_scaler.pkl'))
        joblib.dump(pca_model, os.path.join(output_viz_dir, f'umap_{m}_pca.pkl'))
        joblib.dump(reducer,   os.path.join(output_viz_dir, f'umap_{m}_model.pkl'))

        # Save Python-space coordinates as a SEPARATE reference file.
        # R's UMAP_*_raw_data.csv is kept intact so UMAP_*_.png (flux viz) is unchanged.
        # MOMA projection uses this Python-space reference for correct coordinate alignment.
        ref_csv = os.path.join(output_viz_dir, f'UMAP_{m}_raw_data.csv')
        python_ref_csv = os.path.join(output_viz_dir, f'UMAP_{m}_python_ref.csv')
        if os.path.exists(ref_csv):
            ref_df = pd.read_csv(ref_csv, index_col=0)
            for i, sample in enumerate(mat_m.index):
                if sample in ref_df.index:
                    ref_df.loc[sample, 'UMAP1'] = embedding[i, 0]
                    ref_df.loc[sample, 'UMAP2'] = embedding[i, 1]
            ref_df.to_csv(python_ref_csv)
            logging.info('[sklearn-%s] Saved Python-space ref: %s',
                         m, os.path.basename(python_ref_csv))

        logging.info('[sklearn-%s] Saved: scaler / pca / umap pkl', m)

    return target_reactions


def _confidence_ellipse(ax, x, y, color, level=0.80, zorder=0):
    """Normal-theory confidence ellipse, matching ggplot stat_ellipse(type='norm').

    Draws a faint filled ellipse plus a dashed outline (same style as the R
    UMAP/PCA panels in flux_dimred.R / redraw_umap_panels.R).
    """
    from matplotlib.patches import Ellipse
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    n_std = np.sqrt(-2.0 * np.log(1.0 - level))   # chi-square radius (df = 2)
    width, height = 2 * n_std * np.sqrt(vals)
    cx, cy = x.mean(), y.mean()
    ax.add_patch(Ellipse((cx, cy), width, height, angle=theta,
                         facecolor=color, edgecolor='none', alpha=0.06, zorder=zorder))
    ax.add_patch(Ellipse((cx, cy), width, height, angle=theta,
                         facecolor='none', edgecolor=color,
                         linestyle='--', linewidth=0.8, zorder=zorder + 1))


def visualize_flux_umap(output_viz_dir, mode='all', c1_color=None, c2_color=None,
                        c1_label=None, c2_label=None, ellipse_level=0.80):
    """Matplotlib UMAP scatter mirroring flux_dimred.R / redraw_umap_panels.R.

    Reads ``UMAP_{mode}_raw_data.csv`` and draws condition-colored points with
    80% normal-theory confidence ellipses (dashed outline + faint fill), a plain
    title, a square panel, and a right-side "Condition" legend. Writes
    ``UMAP_{mode}.png`` (+ .svg) — the same output as the R backend.
    """
    set_publication_style()
    os.makedirs(output_viz_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(output_viz_dir, f'UMAP_{mode}_raw_data.csv'),
                     index_col=0)

    color_map = {
        'Control': c1_color or CONDITION_COLORS['Control'],
        'Test':    c2_color or CONDITION_COLORS['Test'],
    }

    # Legend labels: prefer override, then the R-written 'label' column, then condition
    def _disp(cond, override):
        if override:
            return override
        if 'label' in df.columns:
            vals = df.loc[df['condition'] == cond, 'label'].dropna().unique()
            if len(vals):
                return str(vals[0])
        return cond
    label_map = {'Control': _disp('Control', c1_label),
                 'Test':    _disp('Test', c2_label)}

    fig, ax = plt.subplots(figsize=(5, 5))
    for cond in ['Control', 'Test']:
        m = df['condition'] == cond
        if not m.any():
            continue
        x, y = df.loc[m, 'UMAP1'].values, df.loc[m, 'UMAP2'].values
        _confidence_ellipse(ax, x, y, color_map[cond], level=ellipse_level)
        ax.scatter(x, y, c=color_map[cond], marker='o', s=28, alpha=0.85,
                   linewidths=0, label=label_map[cond], zorder=3)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('UMAP')
    ax.set_box_aspect(1)
    ax.legend(title='Condition', frameon=False,
              loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig.savefig(os.path.join(output_viz_dir, f'UMAP_{mode}.svg'),
                format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(output_viz_dir, f'UMAP_{mode}.png'),
                format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info('Saved: UMAP_%s.svg / .png', mode)


def run_moma_results_umap(output_dir, output_viz_dir, target_reactions,
                          mode='df', use_r=True, moma_subdir=''):
    """Project MOMA targeting results into UMAP space.

    Parameters
    ----------
    output_dir       : directory containing flux2.csv and targeting_results/
    output_viz_dir   : directory containing UMAP models and ComBat CSV
    target_reactions : list of reaction IDs (used by Python path;
                       R reads features from umap_{mode}_params.json)
    mode             : UMAP reference mode ('df' or 'all')
    use_r            : if True, call moma_project.R — coordinates match
                         UMAP_{mode}_raw_data.csv (R/ggplot2 space)
                       if False, use Python sklearn models — coordinates
                         match UMAP_{mode}_python_ref.csv (sklearn space)
    moma_subdir      : subdir under targeting_results/ to read MOMA files,
                       and subdir under output_viz_dir/ to write results.
                       Default '' = root level (original behavior).
    """
    if use_r:
        import subprocess
        r_script = _r_script('moma_project.R')
        if not os.path.exists(r_script):
            raise FileNotFoundError(f'moma_project.R not found: {r_script}')
        cmd = [
            _find_rscript(), r_script,
            '--output-dir',     output_dir,
            '--output-viz-dir', output_viz_dir,
            '--mode',           mode,
        ]
        if moma_subdir:
            cmd += ['--moma-subdir', moma_subdir]
        logging.info('[MOMA-R] Running: %s', ' '.join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError('moma_project.R exited with code %d' % result.returncode)
        logging.info('[MOMA-R] Done.')
        return

    # ---- Python (sklearn) path -----------------------------------------------
    import joblib

    # Load sklearn models
    scaler    = joblib.load(os.path.join(output_viz_dir, f'umap_{mode}_scaler.pkl'))
    pca_model = joblib.load(os.path.join(output_viz_dir, f'umap_{mode}_pca.pkl'))
    reducer   = joblib.load(os.path.join(output_viz_dir, f'umap_{mode}_model.pkl'))

    # Load features from params JSON
    params_json = os.path.join(output_viz_dir, f'umap_{mode}_params.json')
    with open(params_json) as fp:
        params = json.load(fp)
    features = params['features']

    # Control center from Python-space reference
    python_ref_csv = os.path.join(output_viz_dir, f'UMAP_{mode}_python_ref.csv')
    ref_df   = pd.read_csv(python_ref_csv, index_col=0)
    ctrl_df  = ref_df[ref_df['condition'] == 'Control']
    center_x = np.median(ctrl_df['UMAP1'].values)
    center_y = np.median(ctrl_df['UMAP2'].values)

    # Load merged flux baseline and raw flux2 for delta correction
    merged_df = pd.read_csv(os.path.join(output_viz_dir, 'merged_flux.csv'),
                            index_col=0)
    flux2_df  = pd.read_csv(os.path.join(output_dir, 'flux2.csv'),
                            index_col=0).fillna(0)

    # Project each MOMA result file
    moma_dir   = os.path.join(output_dir, 'targeting_results')
    moma_files = glob.glob(os.path.join(moma_dir, 'MOMA_target_results_*.csv'))
    logging.info('[MOMA-Python-%s] Projecting %d files ...', mode, len(moma_files))

    for moma_file in moma_files:
        base_noext  = os.path.splitext(os.path.basename(moma_file))[0]
        sample_name = base_noext.replace('MOMA_target_results_', '')

        df_moma  = pd.read_csv(moma_file, index_col=0)   # reactions × genes
        df_feat  = df_moma.reindex(features).fillna(0)   # features × genes
        moma_mat = df_feat.T.values                       # genes × features

        # Delta correction
        if sample_name in merged_df.columns and sample_name in flux2_df.columns:
            wt_combat = merged_df.reindex(features)[sample_name].fillna(0).values
            wt_raw    = flux2_df.reindex(features)[sample_name].fillna(0).values
            X_input   = (moma_mat - wt_raw[np.newaxis, :]) + wt_combat[np.newaxis, :]
        else:
            logging.warning('[MOMA-Python] %s not in combat/flux2 — raw projection',
                            sample_name)
            X_input = moma_mat

        # Scaler → PCA → UMAP
        X_scaled    = scaler.transform(X_input)
        pca_scores  = pca_model.transform(X_scaled)
        umap_coords = reducer.transform(pca_scores)   # genes × 2

        gene_names = list(df_moma.columns)
        result_df  = pd.DataFrame({
            'X': umap_coords[:, 0],
            'Y': umap_coords[:, 1],
            'distance_to_center': np.sqrt((umap_coords[:, 0] - center_x) ** 2 +
                                          (umap_coords[:, 1] - center_y) ** 2),
        }, index=gene_names)

        out_csv = os.path.join(output_viz_dir,
                               f'UMAP_Results_{base_noext}.csv')
        result_df.to_csv(out_csv)

    logging.info('[MOMA-Python-%s] Done.', mode)


def visualize_moma_umap(output_viz_dir, mode='df', use_r=False):
    """Visualize MOMA simulation results.

    Parameters
    ----------
    output_viz_dir : directory containing UMAP_Results_*.csv
    mode           : UMAP reference mode ('df' or 'all')
    use_r          : if True, delegate to moma_viz.R (ggplot2 + ggrepel)
    """
    if use_r:
        import subprocess
        r_script = _r_script('moma_viz.R')
        cmd = [_find_rscript(), r_script,
               '--output-viz-dir', output_viz_dir,
               '--mode',           mode]
        # Entrez → HGNC symbol map for gene labels (bundled package data)
        gene_map = os.path.join(os.path.dirname(__file__), 'data', 'metabolic_genes.csv')
        if os.path.exists(gene_map):
            cmd += ['--gene-map', gene_map]
        logging.info('[R] Running: %s', ' '.join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError('moma_viz.R exited with code %d' % result.returncode)
        logging.info('[R] MOMA visualization done.')
        return

    # ---- Python (matplotlib) path — mirrors moma_viz.R style --------------
    import io, sys
    import matplotlib.patheffects as pe
    from adjustText import adjust_text

    set_publication_style()

    CTRL_COLOR   = '#009E73'   # Control (Wong bluish green)
    TARGET_COLOR = '#E69F00'   # Top-10 targets (orange)
    SAMPLE_COLOR = '#D55E00'   # Sample / toward-Control arrows (vermilion)
    CENTER_COLOR = '#333333'   # Control Center cross
    AWAY_COLOR   = '#B0B0B0'   # arrows away from Control / background MOMA points

    # Entrez → HGNC symbol map (bundled package data)
    gene_map_file = os.path.join(os.path.dirname(__file__), 'data', 'metabolic_genes.csv')
    entrez2symbol = {}
    if os.path.exists(gene_map_file):
        gm = pd.read_csv(gene_map_file, dtype=str)
        if {'entrez', 'symbol'}.issubset(gm.columns):
            entrez2symbol = dict(zip(gm['entrez'], gm['symbol']))

    def _to_symbol(g):
        s = entrez2symbol.get(str(g))
        return s if isinstance(s, str) and s else str(g)

    # Reference UMAP (python_ref preferred, else R raw data)
    python_ref_csv = os.path.join(output_viz_dir, f'UMAP_{mode}_python_ref.csv')
    raw_ref_csv    = os.path.join(output_viz_dir, f'UMAP_{mode}_raw_data.csv')
    ref_csv        = python_ref_csv if os.path.exists(python_ref_csv) else raw_ref_csv
    umap_ref_df    = pd.read_csv(ref_csv, index_col=0)

    ctrl_mask = umap_ref_df['condition'].values == 'Control'
    ctrl_x    = umap_ref_df.loc[ctrl_mask, 'UMAP1'].values
    ctrl_y    = umap_ref_df.loc[ctrl_mask, 'UMAP2'].values
    center_x  = np.median(ctrl_x)
    center_y  = np.median(ctrl_y)

    result_files = glob.glob(os.path.join(output_viz_dir, 'UMAP_Results_*.csv'))
    target_rows = []

    for result_file in result_files:
        basename    = os.path.basename(result_file).replace('UMAP_Results_', '').split('.')[0]
        sample_name = basename.replace('MOMA_target_results_', '')
        tmp_df      = pd.read_csv(result_file, index_col=0)

        # Rank top-10 by recovery toward Control (recovery_proj_pca desc if present,
        # else fall back to distance_to_center asc)
        if 'recovery_proj_pca' in tmp_df.columns:
            top = tmp_df.sort_values('recovery_proj_pca', ascending=False).head(10)
        else:
            top = tmp_df.sort_values('distance_to_center', ascending=True).head(10)

        sample_xy = None
        if sample_name in umap_ref_df.index:
            sample_xy = (umap_ref_df.loc[sample_name, 'UMAP1'],
                         umap_ref_df.loc[sample_name, 'UMAP2'])

        fig, ax = plt.subplots(figsize=(7.2, 6.2))

        _confidence_ellipse(ax, ctrl_x, ctrl_y, CTRL_COLOR, level=0.80)
        ax.scatter(ctrl_x, ctrl_y, color=CTRL_COLOR, s=18, alpha=0.7,
                   linewidths=0, label='Control', zorder=2)
        # All MOMA gene KO results (gray cloud)
        ax.scatter(tmp_df['X'], tmp_df['Y'], color=AWAY_COLOR, s=10, alpha=0.3,
                   linewidths=0, zorder=1)

        # Arrows: sample → KO position for top-10 (vermilion = toward Control)
        if sample_xy is not None:
            sx, sy = sample_xy
            d_sample = np.hypot(sx - center_x, sy - center_y)
            for gid, r in top.iterrows():
                if 'recovery_proj_pca' in top.columns:
                    toward = r['recovery_proj_pca'] > 0
                else:
                    toward = np.hypot(r['X'] - center_x, r['Y'] - center_y) < d_sample
                ax.annotate('', xy=(r['X'], r['Y']), xytext=(sx, sy),
                            arrowprops=dict(arrowstyle='->',
                                            color=SAMPLE_COLOR if toward else AWAY_COLOR,
                                            alpha=0.55 if toward else 0.4, lw=0.6),
                            zorder=3)

        ax.scatter(top['X'], top['Y'], color=TARGET_COLOR, s=40, alpha=0.9,
                   linewidths=0, label='Top-10 targets', zorder=4)
        # Cross markers: Control Center + Sample
        ax.scatter(center_x, center_y, color=CENTER_COLOR, marker='+',
                   s=120, linewidths=1.6, zorder=6, label='Control Center')
        if sample_xy is not None:
            ax.scatter(sample_xy[0], sample_xy[1], color=SAMPLE_COLOR, marker='+',
                       s=120, linewidths=1.6, zorder=6, label=f'Sample({sample_name})')

        # Gene labels (HGNC symbols, italic, white halo + leader lines via adjustText)
        texts = [ax.text(r['X'], r['Y'], _to_symbol(gid), fontsize=9, fontstyle='italic',
                         path_effects=[pe.withStroke(linewidth=2, foreground='white')])
                 for gid, r in top.iterrows()]

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('MOMA Simulation results')
        ax.set_box_aspect(1)
        ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

        _old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='#8C8C8C', lw=0.4))
        finally:
            sys.stdout = _old_stdout

        fig.savefig(os.path.join(output_viz_dir, f'UMAP_{basename}.svg'),
                    format='svg', bbox_inches='tight')
        fig.savefig(os.path.join(output_viz_dir, f'UMAP_{basename}.png'),
                    format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        for gid in top.index:
            target_rows.append((str(gid), _to_symbol(gid), sample_name))

    pd.DataFrame(target_rows, columns=['Gene', 'Symbol', 'Sample']).to_csv(
        os.path.join(output_viz_dir, 'target_genes.csv'), index=False)
    return


def _pathway_barplot_matplotlib(output_dir, output_viz_dir, fdr_cutoff=0.1, title=None):
    """Matplotlib backend for visualize_pathway_enrichment (use_r=False).

    Draws a single combined bar plot of -log10(FDR) for up- and down-regulated
    metabolic pathways, mirroring the R figure (same colors, ratio labels,
    publication style). Writes ``{output_viz_dir}/pathway_barplot.png`` (+ .svg)
    — the same output as the R backend.
    """
    import textwrap

    set_publication_style()
    os.makedirs(output_viz_dir, exist_ok=True)

    UP_COLOR, DOWN_COLOR = '#E41A1C', '#377EB8'

    def _load(direction):
        path = f'{output_dir}/{direction}_regulated_pathways.csv'
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0)
        if df.empty:
            return None
        df = df.copy()
        df['Pathway']   = df.index
        df['Direction'] = direction.capitalize()   # 'Up' / 'Down'
        return df

    parts = [d for d in (_load('Up'), _load('Down')) if d is not None]
    if not parts:
        logging.warning('No pathway enrichment results found — skipping bar plot.')
        return

    combined = pd.concat(parts, ignore_index=True)
    combined = combined[combined['Adjusted P-value'] < fdr_cutoff]
    if combined.empty:
        logging.warning('No pathways pass FDR < %s — skipping bar plot.', fdr_cutoff)
        return

    combined['neg_log10_p'] = -np.log10(combined['Adjusted P-value'])
    overlap = combined['Overlap'].astype(str).str.split('/', expand=True)
    combined['k']       = overlap[0].astype(float)
    combined['n_total'] = overlap[1].astype(float)
    combined['enrich_r'] = (combined['k'] / combined['n_total']).round(2)
    combined['bar_label'] = (combined['enrich_r'].astype(str) + ' (' +
                             combined['k'].astype(int).astype(str) + '/' +
                             combined['n_total'].astype(int).astype(str) + ')')

    # Order: Up group then Down group, each ascending by significance (R: arrange(Direction, neg_log10_p))
    combined['Direction'] = pd.Categorical(combined['Direction'], categories=['Up', 'Down'], ordered=True)
    combined = combined.sort_values(['Direction', 'neg_log10_p']).reset_index(drop=True)

    n = len(combined)
    y_pos  = np.arange(n)
    colors = [UP_COLOR if d == 'Up' else DOWN_COLOR for d in combined['Direction']]
    fig_height = max(n * 0.45 + 2.8, 4)

    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(y_pos, combined['neg_log10_p'], color=colors,
            edgecolor='black', linewidth=0.5, height=0.7)
    for yi, (val, lab) in enumerate(zip(combined['neg_log10_p'], combined['bar_label'])):
        ax.text(val / 2, yi, lab, ha='center', va='center', color='white', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([textwrap.fill(p, 45) for p in combined['Pathway']])
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlim(0, combined['neg_log10_p'].max() * 1.08)
    ax.set_xlabel(r'$-\log_{10}$(FDR)')
    ax.set_ylabel('Metabolic pathway')
    ax.set_title(title or 'Enriched metabolic pathways', pad=30)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=UP_COLOR, edgecolor='black', label='Up'),
                       Patch(facecolor=DOWN_COLOR, edgecolor='black', label='Down')],
              title='Direction', loc='lower center', bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False)

    fig.savefig(f'{output_viz_dir}/pathway_barplot.png', format='png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_viz_dir}/pathway_barplot.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    logging.info('Saved: %s/pathway_barplot.png (+ .svg)', output_viz_dir)
    return
