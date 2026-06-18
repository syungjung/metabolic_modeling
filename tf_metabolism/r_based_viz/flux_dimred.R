#!/usr/bin/env Rscript
# =============================================================================
# flux_dimred.R — Generic flux dimensionality reduction (PCA + UMAP)
#
# Usage:
#   Rscript flux_dimred.R \
#       --output-dir    <path>          # contains flux1.csv, flux2.csv, DE csv
#       --output-viz-dir <path>         # where all outputs are written
#       [--df-file      <path>]         # Differential_fluxes.csv
#       [--mode         all|df|both]    # default: both
#
# Python-facing outputs (read by run_r_umap() in utils.py):
#   {output_viz_dir}/merged_flux.csv        — merged flux matrix (rxn × sample)
#   {output_viz_dir}/umap_{mode}_params.json     — n_pcs, best_nn, best_dist, sample_ids, feature_ids
#   {output_viz_dir}/UMAP_{mode}_raw_data.csv    — 2D UMAP coords + condition
#   {output_viz_dir}/PCA_{mode}_raw_data.csv     — 2D PCA coords + condition
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(ggplot2)
  library(umap)
  library(patchwork)
  library(jsonlite)
})
set.seed(42)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
option_list <- list(
  make_option("--output-dir",     type = "character", help = "dir with flux1.csv / flux2.csv"),
  make_option("--output-viz-dir", type = "character", help = "where to write results"),
  make_option("--df-file",        type = "character", default = NULL,
              help = "path to Differential_fluxes.csv"),
  make_option("--mode",           type = "character", default = "both",
              help = "all | df | both  (default: both)"),
  make_option("--c1-color",       type = "character", default = "#009E73",
              help = "Control group color (default: #009E73)"),
  make_option("--c2-color",       type = "character", default = "#D55E00",
              help = "Test group color (default: #D55E00)"),
  make_option("--c1-label",       type = "character", default = "Control",
              help = "Control group display label (default: Control)"),
  make_option("--c2-label",       type = "character", default = "Test",
              help = "Test group display label (default: Test)")
)

opt  <- parse_args(OptionParser(option_list = option_list))
output_dir     <- opt[["output-dir"]]
output_viz_dir <- opt[["output-viz-dir"]]
df_file        <- opt[["df-file"]]
run_mode       <- tolower(opt[["mode"]])   # all | df | both
c1_label       <- opt[["c1-label"]]
c2_label       <- opt[["c2-label"]]

cond_name <- c2_label
message(sprintf("Condition name: %s", cond_name))

dir.create(output_viz_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Settings — fixed parameters (label-agnostic, literature defaults)
# ---------------------------------------------------------------------------
N_PCS       <- 30    # n_pcs:        30 PCs (appropriate for bulk-level flux data)
N_NEIGHBORS <- 15    # n_neighbors:  15 (UMAP default; suitable for 98–147 samples)
MIN_DIST    <- 0.1   # min_dist:     0.1 (UMAP default, no artificial clustering)

# Colorblind-safe (Wong 2011) defaults; overridden by --c1-color / --c2-color args
# Keys use display labels (c1_label / c2_label) for plot legends
condition_colors <- setNames(c(opt[["c1-color"]], opt[["c2-color"]]),
                             c(c1_label, c2_label))

# ---------------------------------------------------------------------------
# Publication theme
# ---------------------------------------------------------------------------
theme_pub <- function(base_size = 10, base_family = "Liberation Sans") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      plot.background   = element_rect(fill = "white"),
      panel.background  = element_rect(fill = "white"),
      axis.line         = element_line(color = "black", linewidth = 0.75),
      axis.ticks        = element_line(color = "black", linewidth = 0.75),
      axis.ticks.length = unit(6, "pt"),
      axis.title.x      = element_text(margin = margin(t = 5)),
      axis.title.y      = element_text(margin = margin(r = 5)),
      plot.title        = element_text(size = 12, face = "plain", hjust = 0.5),
      axis.title        = element_text(size = 12),
      axis.text         = element_text(size = 12, color = "black"),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key        = element_rect(fill = "white", color = NA),
      legend.text       = element_text(size = 12),
      legend.title      = element_text(size = 12, face = "plain"),
      legend.position   = "right",
      plot.margin       = unit(c(0.15, 0.15, 0.15, 0.15), "cm")
    )
}

# ---------------------------------------------------------------------------
# Helper: run one mode (all or df)
# ---------------------------------------------------------------------------
run_dimred <- function(mat_input, meta, mode, cond_name, output_viz_dir,
                       c1_label = "Control", c2_label = "Test") {

  subtitle <- if (mode == "all") "All reactions" else "DF reactions only"

  message(sprintf("\n%s", strrep("=", 60)))
  message(sprintf("[%s] Dimensionality reduction (%s)", toupper(mode), subtitle))
  message(strrep("=", 60))

  # -- PCA ----------------------------------------------------------------
  message("\n[PCA] Running PCA (n_pcs = ", N_PCS, " fixed)...")
  pca_full <- prcomp(t(mat_input), scale. = TRUE, center = TRUE)
  var_exp  <- summary(pca_full)$importance[2, ] * 100
  cum_var  <- cumsum(var_exp)
  n_pcs_opt <- min(N_PCS, ncol(pca_full$x))

  message(sprintf("  n_pcs=%d (fixed) | cumvar=%.1f%%", n_pcs_opt, cum_var[n_pcs_opt]))

  # Scree plot (reference only — n_pcs line marks fixed value)
  n_show   <- min(n_pcs_opt + 5, length(var_exp))
  df_scree <- data.frame(PC     = seq_len(n_show),
                         VarExp = var_exp[seq_len(n_show)],
                         CumVar = cum_var[seq_len(n_show)])

  p_scree <- ggplot(df_scree, aes(x = PC, y = VarExp)) +
    geom_line(color = "#0072B2", linewidth = 0.3) +
    geom_point(color = "#0072B2", size = 1.5) +
    geom_vline(xintercept = n_pcs_opt, linetype = "dashed",
               color = "#D55E00", linewidth = 0.3) +
    annotate("text", x = n_pcs_opt + 0.5, y = max(df_scree$VarExp) * 0.95,
             label = sprintf("PC%d\n(fixed)", n_pcs_opt),
             hjust = 0, size = 3.2, color = "#D55E00") +
    labs(title = "Scree Plot", x = "PC", y = "Variance Explained (%)") +
    theme_pub()

  p_cumvar <- ggplot(df_scree, aes(x = PC, y = CumVar)) +
    geom_line(color = "#D55E00", linewidth = 0.3) +
    geom_point(color = "#D55E00", size = 1.5) +
    geom_hline(yintercept = c(80, 90), linetype = "dashed",
               color = "grey60", linewidth = 0.3) +
    geom_vline(xintercept = n_pcs_opt, linetype = "dashed",
               color = "#D55E00", linewidth = 0.3) +
    annotate("text", x = n_show, y = c(81.5, 91.5),
             label = c("80%", "90%"), hjust = 1, size = 3, color = "grey50") +
    labs(title = "Cumulative Variance",
         subtitle = sprintf("Fixed: %d PCs (cumvar=%.1f%%)", n_pcs_opt, cum_var[n_pcs_opt]),
         x = "PC", y = "Cumulative Variance (%)") +
    theme_pub()

  p_scree_panel <- (p_scree + p_cumvar) +
    plot_annotation(tag_levels = "A") &
    theme(plot.tag = element_text(size = 12, face = "bold"))
  ggsave(file.path(output_viz_dir, sprintf("pca_scree_cumvar_%s.png", mode)),
         p_scree_panel, width = 10, height = 4.5, dpi = 300)
  ggsave(file.path(output_viz_dir, sprintf("pca_scree_cumvar_%s.svg", mode)),
         p_scree_panel, width = 10, height = 4.5)
  message(sprintf("  Saved: pca_scree_cumvar_%s.png", mode))

  # -- UMAP (fixed parameters) --------------------------------------------
  best_nn <- N_NEIGHBORS
  best_md <- MIN_DIST
  message(sprintf("\n[UMAP] Fixed parameters: n_neighbors=%d, min_dist=%.1f",
                  best_nn, best_md))

  # -- Final UMAP ---------------------------------------------------------
  message("\n[UMAP] Final embedding...")
  cfg_final              <- umap.defaults
  cfg_final$n_neighbors  <- best_nn
  cfg_final$min_dist     <- best_md
  cfg_final$n_components <- 2
  cfg_final$random_state <- 42
  cfg_final$input        <- "data"
  set.seed(42)
  umap_result <- umap(pca_full$x[, seq_len(n_pcs_opt)], config = cfg_final)
  umap_final  <- umap_result$layout

  # Save full R models for MOMA projection (moma_project.R)
  saveRDS(pca_full,    file.path(output_viz_dir, sprintf("pca_%s_model.rds",  mode)))
  saveRDS(umap_result, file.path(output_viz_dir, sprintf("umap_%s_model.rds", mode)))
  message(sprintf("  Saved R models: pca_%s_model.rds  umap_%s_model.rds", mode, mode))

  umap2d <- data.frame(
    UMAP1     = umap_final[, 1],
    UMAP2     = umap_final[, 2],
    condition = meta$condition,
    dataset   = meta$GSE_ID,
    GSM_ID    = meta$GSM_ID,
    stringsAsFactors = FALSE
  )
  rownames(umap2d) <- meta$GSM_ID

  # PCA 2D for reference
  pca2d <- data.frame(
    PC1       = pca_full$x[, 1],
    PC2       = pca_full$x[, 2],
    condition = meta$condition,
    dataset   = meta$GSE_ID,
    GSM_ID    = meta$GSM_ID,
    stringsAsFactors = FALSE
  )
  rownames(pca2d) <- meta$GSM_ID

  # Display label column (Control→c1_label, Test→c2_label) for plot legends
  # condition column keeps "Control"/"Test" for downstream CSV compatibility
  label_map    <- c("Control" = c1_label, "Test" = c2_label)
  umap2d$label <- factor(label_map[umap2d$condition], levels = c(c1_label, c2_label))
  pca2d$label  <- factor(label_map[pca2d$condition],  levels = c(c1_label, c2_label))
  condition_colors <- setNames(unname(condition_colors), c(c1_label, c2_label))

  # -- Visualization (format matches redraw_umap_panels.R) ------------------
  make_plot <- function(df, x, y, xtitle, ytitle, ptitle) {
    ggplot(df, aes(x = .data[[x]], y = .data[[y]], color = label, fill = label)) +
      stat_ellipse(aes(group = label), type = "norm", level = 0.80,
                   linetype = "dashed", linewidth = 0.3,
                   geom = "polygon", alpha = 0.06, show.legend = FALSE) +
      stat_ellipse(aes(group = label), type = "norm", level = 0.80,
                   linetype = "dashed", linewidth = 0.3, show.legend = FALSE) +
      geom_point(size = 2.2, alpha = 0.85) +
      scale_color_manual(values = condition_colors, name = "Condition") +
      scale_fill_manual(values  = condition_colors, name = "Condition") +
      labs(x = xtitle, y = ytitle, title = ptitle) +
      theme_pub() +
      theme(aspect.ratio = 1)
  }

  p_pca  <- make_plot(pca2d,  "PC1",   "PC2",
                      sprintf("PC1 (%.1f%%)", var_exp[1]),
                      sprintf("PC2 (%.1f%%)", var_exp[2]), "PCA")
  p_umap <- make_plot(umap2d, "UMAP1", "UMAP2", "UMAP 1", "UMAP 2", "UMAP")

  # Individual plots
  ggsave(file.path(output_viz_dir, sprintf("PCA_%s.png",  mode)), p_pca,
         width = 4.5, height = 4.5, dpi = 300, bg = "white")
  ggsave(file.path(output_viz_dir, sprintf("PCA_%s.svg",  mode)), p_pca,
         width = 4.5, height = 4.5, bg = "white")
  ggsave(file.path(output_viz_dir, sprintf("UMAP_%s.png", mode)), p_umap,
         width = 4.5, height = 4.5, dpi = 300, bg = "white")
  ggsave(file.path(output_viz_dir, sprintf("UMAP_%s.svg", mode)), p_umap,
         width = 4.5, height = 4.5, bg = "white")

  # Combined panel (PCA + UMAP), shared legend on the right
  p_panel <- (p_pca + p_umap + plot_layout(guides = "collect")) &
    theme(legend.position = "right")
  ggsave(file.path(output_viz_dir, sprintf("PCA_UMAP_panel_%s.png", mode)),
         p_panel, width = 7.0, height = 2.9, dpi = 300, bg = "white")
  ggsave(file.path(output_viz_dir, sprintf("PCA_UMAP_panel_%s.svg", mode)),
         p_panel, width = 7.0, height = 2.9, bg = "white")
  message(sprintf("  Saved: UMAP_%s.png/.svg  PCA_%s.png  PCA_UMAP_panel_%s.png",
                  mode, mode, mode))

  # -- Save raw 2D CSVs (Python reads these) --------------------------------
  umap_csv <- file.path(output_viz_dir, sprintf("UMAP_%s_raw_data.csv", mode))
  pca_csv  <- file.path(output_viz_dir, sprintf("PCA_%s_raw_data.csv",  mode))
  write.csv(umap2d, umap_csv, row.names = TRUE)
  write.csv(pca2d,  pca_csv,  row.names = TRUE)
  message(sprintf("  Saved CSV: %s", basename(umap_csv)))

  # -- Save params JSON (Python reads these to fit sklearn models) ----------
  params <- list(
    n_pcs     = n_pcs_opt,
    best_nn   = best_nn,
    best_dist = best_md,
    features  = rownames(mat_input),
    samples   = meta$GSM_ID
  )
  params_json <- file.path(output_viz_dir, sprintf("umap_%s_params.json", mode))
  write(toJSON(params, auto_unbox = TRUE, pretty = TRUE), params_json)
  message(sprintf("  Saved params: %s", basename(params_json)))

  invisible(list(umap2d = umap2d, pca2d = pca2d, n_pcs = n_pcs_opt,
                 best_nn = best_nn, best_md = best_md))
}

# =============================================================================
# Main
# =============================================================================

# 1. Load flux matrices
message("Loading flux data from: ", output_dir)
flux1 <- read.csv(file.path(output_dir, "flux1.csv"),
                  row.names = 1, check.names = FALSE)
flux2 <- read.csv(file.path(output_dir, "flux2.csv"),
                  row.names = 1, check.names = FALSE)

message(sprintf("  flux1 (Control): %d reactions x %d samples",
                nrow(flux1), ncol(flux1)))
message(sprintf("  flux2 (%s):  %d reactions x %d samples",
                cond_name, nrow(flux2), ncol(flux2)))

# 2. Merge (all common reactions)
common_rxns <- intersect(rownames(flux1), rownames(flux2))
mat_merged  <- cbind(flux1[common_rxns, ], flux2[common_rxns, ])
mat_merged[is.na(mat_merged)] <- 0
rv          <- apply(mat_merged, 1, var)
mat_merged  <- mat_merged[rv > 0, ]
message(sprintf("  Common reactions after zero-var filter: %d", nrow(mat_merged)))

# 3. Sample metadata
all_samples <- colnames(mat_merged)
meta <- data.frame(
  GSM_ID    = all_samples,
  condition = ifelse(all_samples %in% colnames(flux1), "Control", "Test"),
  GSE_ID    = "Unknown",
  stringsAsFactors = FALSE
)

message(sprintf("  Control: %d | Test (%s): %d",
                sum(meta$condition == "Control"),
                cond_name,
                sum(meta$condition == "Test")))

# 4. Merged flux matrix (no batch correction), samples in metadata order
mat_corr <- as.matrix(mat_merged)[, meta$GSM_ID]

# 5. Save matrix for Python (shared by all + df modes); filename kept for
#    downstream compatibility (read by run_r_umap() and moma_project.R)
corr_csv <- file.path(output_viz_dir, "merged_flux.csv")
write.csv(as.data.frame(mat_corr), corr_csv, row.names = TRUE)
message(sprintf("Saved: merged_flux.csv (%d features x %d samples)",
                nrow(mat_corr), ncol(mat_corr)))

# 6. Run for requested mode(s)
run_all <- run_mode %in% c("all", "both")
run_df  <- run_mode %in% c("df",  "both")

if (run_all) {
  run_dimred(mat_corr, meta, "all", cond_name, output_viz_dir,
             c1_label = c1_label, c2_label = c2_label)
}

if (run_df) {
  if (is.null(df_file) || !file.exists(df_file)) {
    df_file <- file.path(output_dir, "Differential_fluxes.csv")
  }
  if (!file.exists(df_file)) {
    message("WARNING: DF file not found — skipping DF mode: ", df_file)
  } else {
    df_csv  <- read.csv(df_file, row.names = 1, check.names = FALSE)
    df_rxns <- intersect(rownames(df_csv), rownames(mat_corr))
    message(sprintf("DF reactions matched: %d / %d",
                    length(df_rxns), nrow(df_csv)))
    if (length(df_rxns) >= 3) {
      run_dimred(mat_corr[df_rxns, ], meta, "df", cond_name, output_viz_dir,
                 c1_label = c1_label, c2_label = c2_label)
    } else {
      message("WARNING: Too few DF reactions — skipping DF mode.")
    }
  }
}

message("\nAll done. Output in: ", output_viz_dir)
