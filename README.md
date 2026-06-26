# tf_metabolism

**Characterizing metabolic reprogramming between two conditions**

A framework that characterizes metabolic reprogramming between two conditions by integrating
transcriptomic data with a genome-scale metabolic model (GEM). It reconstructs condition-specific
models, predicts metabolic fluxes, identifies differentially active reactions and the metabolic
pathways that are reprogrammed, and ranks candidate metabolic intervention targets via MOMA
single-gene knockout simulation.

## Overview

The pipeline (`tf_metabolism/__main__.py`) runs end to end:

1. **Differential expression** — two-group comparison of input transcriptomes.
2. **GEM reconstruction** — condition-specific models built from Recon2M.2 via tINIT
   (omics integration in `tf_metabolism/omics_integration/`).
3. **Flux prediction** — context-specific fluxes per sample (`metabolic_simulation/`,
   Gurobi-based LP/MOMA).
4. **Differential flux & flux-sum analysis** — statistically compares fluxes and
   metabolite turnover between conditions (`statistical_analysis/`).
5. **Pathway enrichment** — enrichment analysis over the differentially active metabolic
   reactions to highlight reprogrammed pathways (`statistical_analysis/enrichment.py`).
6. **MOMA targeting simulation** — single gene knockouts on each condition-specific GEM,
   scored by flux perturbation via MOMA (`run_targeting_simulation`, writes `targeting_results/`).
7. **Visualization** — PCA/UMAP embedding fitting and MOMA projection computations are performed in Python (using scikit-learn and umap-learn). Publication-quality figures (cohort PCA/UMAP panel plots and MOMA targeting plots) are generated in R using the updated visualization scripts (`flux_dimred_plot.R` and `moma_viz.R`). The enriched-pathway bar plot has a selectable backend: R/ggplot2 or matplotlib (`visualize_pathway_enrichment(..., use_r=True|False)`, both write `viz/pathway_barplot.png`).

## Requirements

- **Python 3.6** and **R 4.2** (the pipeline calls R scripts via `Rscript`)
- **Gurobi 9.1** with a valid license — free academic licenses at
  <https://www.gurobi.com/academia/>
- Developed and tested on Linux (x86-64).

## Installation

### Option A — Conda (recommended, one command)

Installs Python packages, R, R packages, and Gurobi together:

```bash
conda env create -f environment.yml
conda activate gems
```

> Gurobi still requires you to activate a license (`grbgetkey <your-key>`).

### Option B — pip + R (manual)

```bash
# Python
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# R packages (see the commented list in requirements.txt)
Rscript -e 'install.packages(c("cluster","dplyr","ggplot2","ggrepel","jsonlite",
  "optparse","patchwork","scales","stringr","svglite","umap"))'
```

Make sure `Rscript` is on your `PATH` (or set the `RSCRIPT` environment variable).

## Usage

```bash
python ./tf_metabolism.py -o <output_dir> -c1 <condition1.csv> -c2 <condition2.csv>
```

Example (the configuration in `run.sh`, healthy control vs. Crohn's disease):

```bash
python ./tf_metabolism.py -o ./output_CD \
    -c1 ./input_data/HC_entrez_tpm.csv \
    -c2 ./input_data/CD_entrez_tpm.csv
```

### Arguments

| Flag | Argument | Description |
|------|----------|-------------|
| `-o`  / `--output_dir`        | path | Output directory (created if missing) |
| `-c1` / `--input_omics_file1` | CSV  | Transcriptome for condition 1 (genes × samples) |
| `-c2` / `--input_omics_file2` | CSV  | Transcriptome for condition 2 (genes × samples) |

### Input format

- `-c1` / `-c2`: CSV with gene identifiers (Entrez) as the row index and samples as columns.
  See `input_data/HC_entrez_tpm.csv`, `input_data/CD_entrez_tpm.csv`,
  or the smaller `input_data/sample1.csv` / `sample2.csv` examples.

### Outputs

Written under `<output_dir>/`:

| File / directory | Contents |
|------------------|----------|
| `condition1/`, `condition2/`   | Reconstructed condition-specific GEMs and split omics |
| `flux1.csv`, `flux2.csv`       | Predicted flux profiles per condition |
| `Differentially_expressed_transcripts.csv` | DEGs between the two conditions |
| `Differential_fluxes.csv`, `All_flux_comparison.csv` | Differential metabolic fluxes |
| `flux_sum_results.csv`         | Flux-sum (metabolite turnover) comparison |
| `gene_subsystem_flux.csv`      | Gene-to-subsystem flux mapping |
| `Up_regulated_pathways.csv`, `Down_regulated_pathways.csv` | Enriched metabolic pathways |
| `targeting_results/`           | MOMA single-gene knockout flux perturbations |
| `viz/`                         | UMAP embeddings, `pathway_barplot.png` (+ .svg), figures |

## Repository layout

```
tf_metabolism/
├── __main__.py                # pipeline entry point
├── metabolic_model/           # GEM editing, GPR manipulation
├── metabolic_simulation/      # flux prediction, MOMA, gap filling, tasks
├── omics_integration/         # tINIT / INIT model reconstruction
├── statistical_analysis/      # differential comparison, enrichment
├── utils.py                   # R-script orchestration, UMAP helpers
└── r_based_viz/               # R visualization scripts (ggplot2)
    ├── flux_dimred_plot.R      #   plot cohort PCA and UMAP figures from Python coordinates
    ├── moma_viz.R              #   plot MOMA UMAP figures + target ranking from Python coordinates
    └── pathway_enrichment_plot.R  # enriched-pathway bar plot
input_data/                    # example transcriptomes, model, media, tasks
environment.yml                # conda environment (Python + R + Gurobi)
requirements.txt               # Python deps (+ R deps as comments)
```
