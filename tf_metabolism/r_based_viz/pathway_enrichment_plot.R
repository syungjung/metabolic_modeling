#!/usr/bin/env Rscript
# =============================================================================
# pathway_enrichment_plot.R — Enriched metabolic pathway bar plot
#
# Reads the pathway enrichment outputs produced by
# statistical_analysis/enrichment.py and draws a publication-style bar plot of
# -log10(FDR) for up- and down-regulated pathways.
#
# Usage:
#   Rscript pathway_enrichment_plot.R \
#       --output-dir     <path>        # contains Up_regulated_pathways.csv / Down_regulated_pathways.csv
#       --output-viz-dir <path>        # where the figures are written
#       [--fdr-cutoff    0.1]          # Adjusted P-value threshold (default 0.1)
#       [--title         "<label>"]    # plot title suffix (e.g. "CD vs HC")
#
# Outputs:
#   {output_viz_dir}/pathway_barplot.png  (+ .svg)
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(ggplot2)
  library(stringr)
})

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
option_list <- list(
  make_option("--output-dir",     type = "character",
              help = "dir with Up_regulated_pathways.csv / Down_regulated_pathways.csv"),
  make_option("--output-viz-dir", type = "character",
              help = "where to write the figures"),
  make_option("--fdr-cutoff",     type = "double", default = 0.1,
              help = "Adjusted P-value cutoff [default %default]"),
  make_option("--title",          type = "character", default = "",
              help = "plot title suffix (e.g. disease label)")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt[["output-dir"]]) || is.null(opt[["output-viz-dir"]])) {
  stop("--output-dir and --output-viz-dir are required")
}
output_dir     <- opt[["output-dir"]]
output_viz_dir <- opt[["output-viz-dir"]]
fdr_cutoff     <- opt[["fdr-cutoff"]]
title_suffix   <- opt[["title"]]

dir.create(output_viz_dir, showWarnings = FALSE, recursive = TRUE)

BAR_H      <- 0.45
FIG_MARGIN <- 2.8

# ---------------------------------------------------------------------------
# Publication theme
# ---------------------------------------------------------------------------
theme_publication <- function(base_size = 12, base_family = "Liberation Sans") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      plot.background       = element_rect(fill = "white"),
      panel.background      = element_rect(fill = "white"),
      panel.grid.major      = element_blank(),
      panel.grid.minor      = element_blank(),
      panel.border          = element_rect(color = "black", fill = NA, linewidth = 0.75),
      axis.line             = element_blank(),
      axis.ticks            = element_line(color = "black", linewidth = 0.75),
      axis.ticks.length     = unit(6, "pt"),
      axis.title.x          = element_text(margin = margin(t = 5)),
      axis.title.y          = element_text(margin = margin(r = 5)),
      plot.title            = element_text(size = 12, face = "plain", hjust = 0.5),
      axis.title            = element_text(size = 12),
      axis.text             = element_text(size = 12, color = "black"),
      legend.background     = element_rect(fill = "white", color = NA),
      legend.key            = element_rect(fill = "white", color = NA),
      legend.text           = element_text(size = 12),
      legend.title          = element_text(size = 12, face = "plain"),
      legend.position       = "right",
      plot.margin           = unit(c(0.15, 0.15, 0.15, 0.15), "cm"),
      strip.background      = element_blank(),
      strip.text            = element_text(face = "plain", size = 12),
      plot.title.position   = "plot"
    )
}

# ---------------------------------------------------------------------------
# Load enrichment results
# ---------------------------------------------------------------------------
up_file   <- file.path(output_dir, "Up_regulated_pathways.csv")
down_file <- file.path(output_dir, "Down_regulated_pathways.csv")

read_pathways <- function(path, direction) {
  if (!file.exists(path)) return(NULL)
  df <- read.csv(path, row.names = 1, check.names = FALSE)
  if (nrow(df) == 0) return(NULL)
  df$Pathway   <- rownames(df)
  df$Direction <- direction
  df
}

up_df   <- read_pathways(up_file,   "Up")
down_df <- read_pathways(down_file, "Down")

if (is.null(up_df) && is.null(down_df)) {
  message("No pathway enrichment results found in ", output_dir, " — nothing to plot.")
  quit(save = "no", status = 0)
}

combined <- bind_rows(up_df, down_df) %>%
  filter(`Adjusted P-value` < fdr_cutoff)

if (nrow(combined) == 0) {
  message("No pathways pass FDR < ", fdr_cutoff, " — nothing to plot.")
  quit(save = "no", status = 0)
}

combined <- combined %>%
  mutate(
    neg_log10_p = -log10(`Adjusted P-value`),
    k           = as.numeric(sub("/.*", "", Overlap)),
    n_total     = as.numeric(sub(".*/", "", Overlap)),
    Direction   = factor(Direction, levels = c("Up", "Down"))
  ) %>%
  arrange(Direction, neg_log10_p) %>%
  mutate(
    row_id    = factor(seq_len(n())),
    enrich_r  = round(k / n_total, 2),
    bar_label = paste0(enrich_r, " (", k, "/", n_total, ")"),
    text_x    = neg_log10_p / 2
  )

pathway_labels <- setNames(combined$Pathway, combined$row_id)
x_upper        <- max(combined$neg_log10_p) * 1.08
fig_height     <- nrow(combined) * BAR_H + FIG_MARGIN

plot_title <- if (nzchar(title_suffix)) {
  paste0("Enriched metabolic pathways (", title_suffix, ")")
} else {
  "Enriched metabolic pathways"
}

# ---------------------------------------------------------------------------
# Build plot
# ---------------------------------------------------------------------------
p <- ggplot(combined, aes(x = neg_log10_p, y = row_id, fill = Direction)) +
  geom_col(color = "black", linewidth = 0.5, width = 0.7) +
  geom_text(aes(label = bar_label, x = text_x), color = "white", size = 3.8) +
  scale_y_discrete(labels = function(x) str_wrap(pathway_labels[x], width = 45)) +
  scale_x_continuous(limits = c(0, x_upper), expand = c(0, 0)) +
  scale_fill_manual(
    values = c("Up" = "#E41A1C", "Down" = "#377EB8"),
    breaks = c("Up", "Down"),
    name   = "Direction"
  ) +
  labs(
    y     = "Metabolic pathway",
    x     = expression(-log[10]("FDR")),
    title = plot_title
  ) +
  theme_publication() +
  guides(fill = guide_legend(nrow = 1, title.position = "left")) +
  theme(
    legend.position      = "top",
    legend.direction     = "horizontal",
    legend.justification = "center"
  )

out_png <- file.path(output_viz_dir, "pathway_barplot.png")
ggsave(out_png, plot = p, width = 9, height = fig_height, dpi = 300, bg = "white")
ggsave(sub("\\.png$", ".svg", out_png), plot = p,
       width = 9, height = fig_height, bg = "white")
message("Saved: ", out_png, " (+ .svg)")
