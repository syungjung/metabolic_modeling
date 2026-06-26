#!/usr/bin/env Rscript
# =============================================================================
# flux_dimred_plot.R — Plot cohort PCA and UMAP from Python-generated CSVs
# =============================================================================
suppressPackageStartupMessages({
  library(optparse)
  library(ggplot2)
  library(patchwork)
  library(jsonlite)
})

option_list <- list(
  make_option("--output-viz-dir", type = "character", help = "where raw data CSVs are and plots are written"),
  make_option("--mode",           type = "character", default = "df", help = "all | df"),
  make_option("--c1-color",       type = "character", default = "#009E73"),
  make_option("--c2-color",       type = "character", default = "#D55E00"),
  make_option("--c1-label",       type = "character", default = "Control"),
  make_option("--c2-label",       type = "character", default = "Test")
)

opt <- parse_args(OptionParser(option_list = option_list))
output_viz_dir <- opt[["output-viz-dir"]]
mode           <- opt[["mode"]]
c1_color       <- opt[["c1-color"]]
c2_color       <- opt[["c2-color"]]
c1_label       <- opt[["c1-label"]]
c2_label       <- opt[["c2-label"]]

condition_colors <- setNames(c(c1_color, c2_color), c(c1_label, c2_label))

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

pca_csv  <- file.path(output_viz_dir, sprintf("PCA_%s_raw_data.csv",  mode))
umap_csv <- file.path(output_viz_dir, sprintf("UMAP_%s_raw_data.csv", mode))

if (!file.exists(pca_csv) || !file.exists(umap_csv)) {
  stop("Input CSV files not found.")
}

pca2d  <- read.csv(pca_csv,  row.names = 1, check.names = FALSE)
umap2d <- read.csv(umap_csv, row.names = 1, check.names = FALSE)

label_map    <- c("Control" = c1_label, "Test" = c2_label)
umap2d$label <- factor(label_map[umap2d$condition], levels = c(c1_label, c2_label))
pca2d$label  <- factor(label_map[pca2d$condition],  levels = c(c1_label, c2_label))

# Read PCA explained variance from params JSON if available
params_json <- file.path(output_viz_dir, sprintf("umap_%s_params.json", mode))
var_exp <- c(0, 0)
if (file.exists(params_json)) {
  params <- fromJSON(params_json)
  if ("pca_explained_variance_ratio" %in% names(params)) {
    var_exp <- params$pca_explained_variance_ratio * 100
  }
}

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

# Save plots
ggsave(file.path(output_viz_dir, sprintf("PCA_%s.png",  mode)), p_pca,
       width = 4.5, height = 4.5, dpi = 300, bg = "white")
ggsave(file.path(output_viz_dir, sprintf("PCA_%s.svg",  mode)), p_pca,
       width = 4.5, height = 4.5, bg = "white")
ggsave(file.path(output_viz_dir, sprintf("UMAP_%s.png", mode)), p_umap,
       width = 4.5, height = 4.5, dpi = 300, bg = "white")
ggsave(file.path(output_viz_dir, sprintf("UMAP_%s.svg", mode)), p_umap,
       width = 4.5, height = 4.5, bg = "white")

p_panel <- (p_pca + p_umap + plot_layout(guides = "collect")) &
  theme(legend.position = "right")
ggsave(file.path(output_viz_dir, sprintf("PCA_UMAP_panel_%s.png", mode)),
       p_panel, width = 7.0, height = 2.9, dpi = 300, bg = "white")
ggsave(file.path(output_viz_dir, sprintf("PCA_UMAP_panel_%s.svg", mode)),
       p_panel, width = 7.0, height = 2.9, bg = "white")

message("Cohort plots saved.")
